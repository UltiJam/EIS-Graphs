import os
from PyEIS import *

def peis_files(path):
    """Get all PEIS files in directory"""
    return [f for f in os.listdir(path) if f.endswith('.mpt') and 'PEIS' in f]


def fit(path):
    # Get files
    fn = peis_files(path)

    ex1 = EIS_exp(path=path, data=fn)
    print(ex1.df_raw.head())

    ex1.EIS_plot(legend='potential', bode='log_im', savefig=f'{path}fig1.png')

    ex1.Lin_KK(legend='potential', bode='log_im', plot='w_data', savefig=f'{path}fig2.png')

    # w = Angular frequency[1 / s]
    # Rs = Series resistance[Ohm]
    # R = Resistance[Ohm]
    # Q = Constant phase element[s ^ n / ohm]
    # n = Constant phase elelment exponent[-]
    # fs = summit frequency of RQ circuit[Hz]
    Rs_guess = 50
    R_guess = 10000
    n_guess = 0.9
    fs_guess = 0.001

    params = Parameters()
    params.add('Rs', value=Rs_guess, min=Rs_guess * .01, max=Rs_guess * 100)
    params.add('R', value=R_guess, min=R_guess * .001, max=R_guess * 100)
    params.add('n', value=n_guess, min=n_guess * .1, max=n_guess * 10)
    params.add('fs', value=fs_guess, min=fs_guess * .01, max=fs_guess * 100000)
    # params.add('Q', value=fs_guess, min=fs_guess*.01, max=fs_guess*100000)

    ex1.EIS_fit(params=params, circuit='R-RQ', weight_func='modulus')

    ex1.EIS_plot(legend='potential', fitting='on', savefig=f'{path}fig3.png')


# TEST CODE
#path = 'C:\\Users\\Jamie\\OneDrive - University of Surrey\\Samples\\Corewire HY-100\\Sample 1A\\Corrosion Cell\\Test 9\\'
path = 'C:\\Users\\Jamie\\OneDrive - University of Surrey\\Samples\\Dstl Q1N\\Sample 1A\\Corrosion Cell\\Test 4\\zfit\\'
os.chdir(path)
save_fig = False
save_pdf = False
fit(path)
