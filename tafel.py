import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def tp_files(path):
    """Get all TP files in directory"""
    return [f for f in os.listdir(path) if f.endswith('.mpt') and 'TP' in f]


def extrapolate(x_bf, y_bf, x_int):
    # Unpack input variables
    x_0, x_1 = x_bf
    y_0, y_1 = y_bf

    # Calculate equation of line
    grad = (y_0 - y_1) / (math.log(x_0, 10) - math.log(x_1, 10))
    k = y_0 - (grad * math.log(x_0, 10))

    # Get limits of extrapolation
    x_max = 0.05  # Minimum x limit
    x_min = math.pow(10, (math.log(x_int, 10) - 0.25))

    # Extrapolate
    y_min = (grad * math.log(x_min, 10)) + k
    y_max = (grad * math.log(x_max, 10)) + k
    return [x_min, x_max], [y_min, y_max]


def regression(X, Y):
    denom = X.dot(X) - X.mean() * X.sum()
    m = (X.dot(Y) - Y.mean() * X.sum()) / denom
    b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denom
    y_pred = (m*X + b)
    res = Y - y_pred
    tot = Y - Y.mean()
    R_sq = 1 - res.dot(res) / tot.dot(tot)
    return y_pred, R_sq


def linear_regression(x_values, y_values, width):
    # Smooth data with LOWESS algorithm
    lowess = sm.nonparametric.lowess(y_values, x_values, frac=0.1)
    x_arr = lowess[:, 0]
    y_arr = lowess[:, 1]

    # Crop data for I > 0.1 mA
    x_arr = np.flip(x_arr)
    y_arr = np.flip(y_arr)
    while x_arr[0] > 0.1:
        x_arr = np.delete(x_arr, 0)
        y_arr = np.delete(y_arr, 0)
    x_arr = np.flip(x_arr)
    y_arr = np.flip(y_arr)
    while x_arr[0] < 0.001:
        x_arr = np.delete(x_arr, 0)
        y_arr = np.delete(y_arr, 0)

    # Convert from linear-log to linear plot
    x_log_arr = np.log10(x_arr)
    X = np.asarray(x_log_arr)
    Y = np.asarray(y_arr)

    print(len(X))
    print(len(Y))

    x_pred = 0
    y_pred = 0
    R = 0
    idx = 0
    for i in range(len(X)):
        if i == len(X) - width:
            break
        X_i = X[i:i+width]
        Y_i = Y[i:i+width]
        Y_R, R_R = regression(X_i, Y_i)
        if R_R > R:
            x_pred = X_i
            y_pred = Y_R
            R = R_R
            idx = i

    print(R)

    plt.plot(X, Y)
    plt.plot(x_pred, y_pred, 'r')
    plt.show()

    x_0 = x_arr[idx]
    x_1 = x_arr[idx+width-1]
    y_0 = y_pred[0]
    y_1 = y_pred[-1]
    return [x_0, x_1], [y_0, y_1]


def plot(path, save_fig, save_pdf, width_n, width_p):
    # Get files
    fn = tp_files(path)

    # Use pandas to read the data file
    headers = ["mode", "ox", "error", "control", "time", "controlV", "ewe", "I", "P"]
    df = pd.read_csv(fn[0], '\t', names=headers, encoding="ISO-8859-1")
    idx = df.index[df['mode'] == "mode"].tolist()   # Find index of data header
    df = df.iloc[idx[0]+1:]                         # Delete rows before data
    df = df.astype(float)                           # Convert to float

    # Arrays of raw data
    current = df["I"]
    volts = df["ewe"] * 1000

    # Split data in to positive and negative currents for plotting
    idx_ary = np.sign(current)
    idx_split = np.where(idx_ary == 1)[0][0]
    curr_split = np.split(current, [idx_split])
    volt_split = np.split(volts, [idx_split])
    curr_n, curr_p = curr_split[0], curr_split[1]
    volt_n, volt_p = volt_split[0], volt_split[1]

    # Plot the raw positive and negative data
    plt.plot(curr_n*-1, volt_n, 'C0', curr_p, volt_p, 'C0')
    plt.xscale('log')
    plt.xlabel("I (mA)")
    plt.ylabel("Ewe (mV)")
    plt.grid('true')
    fig = plt.gcf()
    fig.set_size_inches(11.69, 8.27)
    plt.savefig('py_tafel_graph.svg', bbox_inches='tight') if save_fig else False
    plt.savefig('py_tafel_graph.pdf') if save_pdf else False
    plt.show()


    # Work out the lines of best fit
    # Linear Regression method
    x_n, y_n = linear_regression(curr_n * -1, volt_n, width_n)
    x_p, y_p = linear_regression(curr_p, volt_p, width_p)

    # x_n, y_n = bestfit_line(curr_n * -1, volt_n, 10)
    # x_p, y_p = bestfit_line(curr_p, volt_p, 10)

    # Calculate the intersect
    # Equations for best fit lines
    m1 = (y_n[1] - y_n[0]) / (math.log(x_n[1], 10) - math.log(x_n[0], 10))
    b = y_n[1] - (m1 * math.log(x_n[1], 10))
    m2 = (y_p[1] - y_p[0]) / (math.log(x_p[1], 10) - math.log(x_p[0], 10))
    c = y_p[1] - (m2 * math.log(x_p[1], 10))
    # Calculate coordinates of intersect
    x = math.pow(10, ((-(b-c))/(m1-m2)))
    y = (m1 * math.log(x, 10)) + b

    # Extrapolate the bestfit lines to intersect
    x_n_e, y_n_e = extrapolate(x_n, y_n, x)
    x_p_e, y_p_e = extrapolate(x_p, y_p, x)

    # Plot intersecting graph
    plt.plot(curr_n*-1, volt_n, 'C0o', curr_p, volt_p, 'C0o',
             x_n_e, y_n_e, 'C1', x_p_e, y_p_e, 'C1',
             x_n, y_n, 'C2', x_p, y_p, 'C2', markersize=2)
    plt.xscale('log')
    plt.xlabel("I (mA)")
    plt.ylabel("Ewe (mV)")
    plt.grid('true')
    # Plot horizontal line from intersection and annotate
    x_lims = plt.gca().get_xlim()
    x_span = math.log(x_lims[1], 10) - math.log(x_lims[0], 10)
    x_dif = math.log(x, 10) - math.log(x_lims[0], 10)
    x_loc = x_dif / x_span
    plt.axhline(y=y, xmax=x_loc, color='C3')
    plt.annotate(f'$E_{{corr}}={y:.4f} mV$', xy=(min(curr_p), y - 25), xycoords='data', size=12)
    # Plot vertical line from intersection and annotate
    y_lims = plt.gca().get_ylim()
    y_span = y_lims[1] - y_lims[0]
    y_dif = y - y_lims[0]
    y_loc = y_dif / y_span
    plt.axvline(x=x, ymax=y_loc, color='C3')
    plt.annotate(f'$I_{{corr}}={x:.3e} mA$', xy=(x + 0.0005, min(volt_n)), xycoords='data', size=12)
    fig = plt.gcf()
    fig.set_size_inches(11.69, 8.27)
    plt.savefig('py_tafel_graph_intersecting.svg', bbox_inches='tight')
    plt.show()


# TEST CODE
#path = 'C:\\Users\\Jamie\\OneDrive - University of Surrey\\Samples\\Corewire HY-100\\Sample 1B\\Corrosion Cell\\Test 7\\'
path = 'C:\\Users\\Jamie\\OneDrive - University of Surrey\\Samples\\Dstl Q1N\\Sample 1B\\Corrosion Cell\\Test 5\\'
os.chdir(path)
save_fig = False
save_pdf = False
width_n = 10
width_p = 30
plot(path, save_fig, save_pdf, width_n, width_p)





def bestfit_line(x_values, y_values, width):
    """Calculate start and end points for linear line"""

    # Smooth data with LOWESS algorithm
    lowess = sm.nonparametric.lowess(y_values, x_values, frac=0.1)
    x_arr = lowess[:, 0]
    y_arr = lowess[:, 1]

    # Crop data for I > 0.1 mA
    x_arr = np.flip(x_arr)
    y_arr = np.flip(y_arr)
    while x_arr[0] > 0.1:
        x_arr = np.delete(x_arr, 0)
        y_arr = np.delete(y_arr, 0)
    x_arr = np.flip(x_arr)
    y_arr = np.flip(y_arr)
    while x_arr[0] < 0.004:
        x_arr = np.delete(x_arr, 0)
        y_arr = np.delete(y_arr, 0)

    # Convert from linear-log to linear plot
    x_log_arr = np.log10(x_arr)

    # Plot data and setup for second derivative
    fig, ax1 = plt.subplots()
    ax1.plot(x_log_arr, y_arr, 'bo')
    ax2 = ax1.twinx()

    # First derivative
    x_diff_arr = x_log_arr[1:] - x_log_arr[:-1]
    y_diff_arr = y_arr[1:] - y_arr[:-1]
    y_1d_arr = y_diff_arr / x_diff_arr
    lowess = sm.nonparametric.lowess(y_1d_arr, x_log_arr[:-1], frac=0.2)
    x_1d = lowess[:, 0]
    y_1d = lowess[:, 1]

    # Second derivative
    x_diff_arr = x_1d[1:] - x_1d[:-1]
    y_diff_arr = y_1d[1:] - y_1d[:-1]
    y_2d_arr = y_diff_arr / x_diff_arr
    lowess = sm.nonparametric.lowess(y_2d_arr, x_log_arr[:-2], frac=0.2)
    x_2d = lowess[:, 0]
    y_2d = lowess[:, 1]
    ax2.plot(x_log_arr[:-2], y_2d_arr, 'C3')
    ax2.plot(x_2d, y_2d, 'C4')
    plt.show()

    # Find the inflection point between the min and max peaks of smoothed second derivative
    # idx_min = np.where(y_2d == min(y_2d))[0][0]
    # idx_max = np.where(y_2d == max(y_2d))[0][0]
    # idx = [idx_min, idx_max]
    # idx_min = min(idx)
    # idx_max = max(idx)

    # Get indexes for inflection and apply with best fit data width
    # y_2d_crop = y_2d[idx_min:idx_max]
    # idx_inlf = np.where(np.absolute(y_2d_crop) == min(np.absolute(y_2d_crop)))[0][0]
    # y_2d_crop = y_2d_crop[idx_inlf-width:idx_inlf+width+1]  # Sets window either side of inflection point
    # idx_min = np.where(y_2d == y_2d_crop[0])[0][0]
    # idx_max = np.where(y_2d == y_2d_crop[-1])[0][0]

    # Overwrite crop between min and max second derivative
    y_2d_crop = y_2d
    idx_inlf = np.where(np.absolute(y_2d_crop) == min(np.absolute(y_2d_crop)))[0][0]
    y_2d_crop = y_2d_crop[idx_inlf-width:idx_inlf+width+1]  # Sets window either side of inflection point
    idx_min = np.where(y_2d == y_2d_crop[0])[0][0]
    idx_max = np.where(y_2d == y_2d_crop[-1])[0][0]

    # Find x limits for linear line
    lowess = sm.nonparametric.lowess(y_values, x_values, frac=0.1)
    x_lin = np.asarray(lowess[:, 0])
    x_0 = x_lin[idx_min]
    x_1 = x_lin[idx_max]

    # Create y points for linear negative line
    y_lin = np.asarray(lowess[:, 1])
    y_0 = y_lin[idx_min]
    y_1 = y_lin[idx_max]
    return [x_0, x_1], [y_0, y_1]
