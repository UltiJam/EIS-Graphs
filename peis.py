import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def peis_data(path):
    """Get all PEIS files in directory"""
    fn = [f for f in os.listdir(path) if f.endswith('.mpt') and 'PEIS' in f]
    # [freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm, |Z|/Ohm, Phase(Z)/deg, time/s, <Ewe>/V, <I>/mA, Cs/µF,
    # Cp/µF, cycle number, I Range, |Ewe|/V, |I|/A, Re(Y)/Ohm-1, Im(Y)/Ohm-1, |Y|/Ohm-1, Phase(Y)/deg]

    # Use pandas to read the data file
    headers = ["freq", "reZ", "imZ", "Z", "phaseZ", "time", "ewe", "I", "Cs", "Cp", "cycle",
               "Irange", "modewe", "modI", "reY", "imY", "modY", "phaseY"]
    df = pd.read_csv(fn[0], '\t', names=headers, encoding="ISO-8859-1")
    idx = df.index[df['freq'] == "freq/Hz"].tolist()   # Find index of data header
    df = df.iloc[idx[0]+1:]                         # Delete rows before data
    return df.astype(float)     # Convert to float


def bode(data, save_fig, save_pdf):
    """Bode plot"""

    freq = data["freq"].to_numpy()
    z = data["Z"].to_numpy()
    phase = data["phaseZ"].to_numpy()

    print(max(z))

    # Crop data for freq > log4
    while freq[0] > 10000:
        freq = np.delete(freq, 0)
        z = np.delete(z, 0)
        phase = np.delete(phase, 0)

    fig, ax1 = plt.subplots()
    color = 'C0'
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (\u03A9$\mathregular{cm^2}$)', color=color)
    ax1.set_yscale('log')
    ax1.plot(freq, z, 'x-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'C1'
    ax2.set_ylabel('Phase Angle (\u00B0)', color=color)  # we already handled the x-label with ax1
    ax2.semilogx(freq, phase, 'x-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ax2.get_ylim()[::-1]) # flips y axis
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig = plt.gcf()
    fig.set_size_inches(11.69, 8.27)
    plt.savefig('py_bode_graph.svg', bbox_inches='tight') if save_fig else False
    plt.savefig('py_bode_graph.pdf') if save_pdf else False
    plt.show()


def nyquist(data, save_fig, save_pdf):
    """Nyquist plot"""

    re_z = data["reZ"]
    im_z = data["imZ"]

    plt.plot(re_z, im_z)
    plt.xlabel("Z' (\u03A9$\mathregular{cm^2}$)")
    plt.ylabel("-Z'' (\u03A9$\mathregular{cm^2}$)")
    plt.grid('true')
    fig = plt.gcf()
    fig.set_size_inches(11.69, 8.27)
    plt.savefig('py_nyquist_graph.svg', bbox_inches='tight') if save_fig else False
    plt.savefig('py_nyquist_graph.pdf') if save_pdf else False
    plt.show()
    return


def plot(path, save_fig, save_pdf):
    """Bode and Nyquist plots from functions"""

    data = peis_data(path)
    bode(data, save_fig, save_pdf)
    nyquist(data, save_fig, save_pdf)
    return
