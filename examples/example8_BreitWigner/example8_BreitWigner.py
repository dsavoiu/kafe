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
#---------------------------------------------------------------------

# import everything we need from kafe
from kafe import *
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
BWplot = Plot(BWfit)
BWplot.plot_all()
BWplot.save("kafe_BreitWignerFit.pdf")
BWplot.show()

# plot 1-sigma contour of first two parameters into a separate figure
x, y = BWfit.minimizer.get_contour(0, 1, n_points=100)  # get contour
cont_fig = plt.figure()  # create new figure for contour
cont_ax = cont_fig.gca()  # get/create axes object for current figure
# set axis labels
cont_ax.set_xlabel('$%s$' % (BWfit.latex_parameter_names[0],))
cont_ax.set_ylabel('$%s$' % (BWfit.latex_parameter_names[1],))
# plot the actual contour
cont_ax.fill(x, y, alpha=0.25, color='red')
# save to file
cont_fig.savefig("kafe_BreitWignerFit_contour12.pdf")
# show the contour
#cont_fig.show()

