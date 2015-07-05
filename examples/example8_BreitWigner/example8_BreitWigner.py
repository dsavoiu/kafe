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
#                09-OCT-14 added section to plot contour
#                05-Dec-14 more comfortable contour plotting
#---------------------------------------------------------------------

# import everything we need from kafe
import kafe
# import helper function to parse the input file
from kafe.file_tools import buildFit_fromFile
#
import matplotlib.pyplot as plt
#
# ---------------------------------------------------------

fname = 'LEP-Data.dat'
# initialize fit object from file
BWfit = buildFit_fromFile(fname)
BWfit.do_fit()
#
BWplot = kafe.Plot(BWfit)
BWplot.plot_all()
BWplot.save("kafe_BreitWignerFit.pdf")

# plot contours and profiles
BWfit.plot_correlations()

# show everything
plt.show()
