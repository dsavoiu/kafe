#!/usr/bin/env python
# example8_BreitWigner 
'''
Fitting a Breit-Wigner Resonance
--------------------------------

    In this example, a Breit-Wigner shape is fitted to measurements of the 
    hadronic cross section at the electron-positron collider LEP around 
    the Z resonance.

    Illustrates the usage of ``buildFit_fromFile``.
'''

#
##--------------------------------------------------------------------
# Author:      G. Quast   Jul. 2014
# dependencies: PYTHON v2.7, sys, numpy, matplotlib.pyplot, 
#               kafe
# last modified: 27-JUL-31 <initial version>
#                06-AUG-14 changed to use buildFit_fromFile()
#---------------------------------------------------------------------

# import everything we need from kafe
from kafe import *
# import helper function to parse the input file
from kafe.file_tools import buildFit_fromFile
#
# ---------------------------------------------------------

fname = 'LEP-Data.dat'
# initialize fit object from file
BWfit = buildFit_fromFile(fname)
BWfit.do_fit()
#
BWplot = Plot(BWfit)
BWplot.plot_all()
BWplot.save("kafe_BreitWignerFit.pdf")
BWplot.show()
