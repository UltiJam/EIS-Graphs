import os
from PyPDF2 import PdfFileMerger

import ocv
import peis
import tafel


def directory(sample, test):
    """Set new working directory"""
    fileDir = os.path.dirname(os.path.abspath(__file__))  # Directory of the Module
    parentDir = os.path.dirname(fileDir)  # Directory of the Module directory
    parentDir = os.path.dirname(parentDir)  # Directory of the Module directory
    newPath = os.path.join(parentDir, 'Samples', sample, 'Corrosion Cell', test)
    os.chdir(newPath)  # Change working directory
    return newPath


def graphMaker():
    """Merge PDF files"""
    pdfs = ['py_ocv_graph.pdf', 'py_bode_graph.pdf', 'py_nyquist_graph.pdf', 'py_tafel_graph.pdf']
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write("py_combined.pdf")
    merger.close()
    for pdf in pdfs:
        os.remove(pdf)


# Test file info
samp = "Corewire HY-100\\Sample 1B"
#samp = "Dstl Q1N\\Sample 1A"
test = "Test 6"
path = directory(samp, test)
save_fig = True
save_pdf = True

ocv.plot(path, save_fig, save_pdf)
peis.plot(path, save_fig, save_pdf)
tafel.plot(path, save_fig, save_pdf)
graphMaker() if save_pdf else False
