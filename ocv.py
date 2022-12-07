import os
import pandas as pd
import matplotlib.pyplot as plt


def ocv_files(path):
    # Returns all OCV files in directory
    return [f for f in os.listdir(path) if f.endswith('.mpt') and 'OCV' in f]


def plot(path, save_fig, save_pdf):
    # Get files
    fn = ocv_files(path)

    # Use pandas to read the data file
    df = pd.read_csv(fn[0], '\t', names=["mode", "error", "time", "ewe"], encoding="ISO-8859-1")
    idx = df.index[df['mode'] == "mode"].tolist()   # Find index of data header
    df = df.iloc[idx[0]+1:]                         # Delete rows before data
    df = df.astype(float)                           # Convert to float

    time = df["time"]
    volts = df["ewe"] * 1000

    plt.plot(time, volts)
    plt.xlabel("Time (s)")
    plt.ylabel("Ewe (mV)")
    plt.grid('true')
    fig = plt.gcf()
    fig.set_size_inches(11.69, 8.27)
    plt.savefig('py_ocv_graph.svg', bbox_inches='tight') if save_fig else False
    plt.savefig('py_ocv_graph.pdf') if save_pdf else False
    plt.show()
