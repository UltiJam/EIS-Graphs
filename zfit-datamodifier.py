import os
import pandas as pd


def peis_modify(path):
    """Get all PEIS files in directory"""
    fn = [f for f in os.listdir(path) if f.endswith('.mpt') and 'PEIS' in f]

    # freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm, |Z|/Ohm, Phase(Z)/deg, time/s, <Ewe>/V, <I>/mA, Cs/µF, Cp/µF, cycle number,
    # I Range, |Ewe|/V, |I|/A, (Q - Qo)/mA.h, Re(Y)/Ohm-1, Im(Y)/Ohm-1, |Y|/Ohm-1, Phase(Y)/deg, dq/mA.h

    # Use pandas to read the data file
    headers = ["freq", "reZ", "imZ", "Z", "phaseZ", "time", "ewe", "I", "Cs", "Cp", "cycle",
               "Irange", "modewe", "modI", "qqo", "reY", "imY", "modY", "phaseY", "dq"]
    df1 = pd.read_csv(fn[0], '\t', names=headers, encoding="ISO-8859-1", skip_blank_lines=False)
    df2 = pd.read_csv(fn[0], '\t', names=headers, encoding="ISO-8859-1", skip_blank_lines=False)

    # df1 is start info of file
    idx = df1.index[df1['freq'] == "freq/Hz"].tolist()   # Find index of data header
    df1 = df1.iloc[:idx[0]]                         # Delete rows after data
    df1 = df1.drop(columns=["imZ", "Z", "phaseZ", "time", "ewe", "I", "Cs", "Cp", "cycle",
               "Irange", "modewe", "modI", "qqo", "reY", "imY", "modY", "phaseY", "dq"])
    df1.loc[df1['reZ'].notnull(), 'reZ'] = '\t' + df1['reZ']
    df1.loc[df1['freq'].isnull(),'freq'] = df1['reZ']
    df1 = df1.drop(columns=["reZ"])

    print(df1)

    # df2 is the data table
    # Headers to delete:
    # (Q - Qo)/mA.h, dq/mA.h
    df2 = df2.iloc[idx[0]:]                         # Delete rows before data
    df2 = df2.drop(columns=["qqo", "dq"])

    with open(f'{path}zfit\\{fn[0][:-4]}_Modified.mpt', 'w', newline='') as f:
        df1.to_csv(f, header=None, index=None, sep='\t', mode='a')
        df2.to_csv(f, header=None, index=None, sep='\t', mode='a')
    return


# TEST CODE
# path = 'C:\\Users\\Jamie\\OneDrive - University of Surrey\\Samples\\Corewire HY-100\\Sample 1A\\Corrosion Cell\\Test 9\\'
path = 'C:\\Users\\Jamie\\OneDrive - University of Surrey\\Samples\\Dstl Q1N\\Sample 1B\\Corrosion Cell\\Test 1\\'
os.chdir(path)
peis_modify(path)
