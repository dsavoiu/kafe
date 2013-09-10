'''
W boson mass
============

    This example shows the averaging of several experiments' results to
find the mass of the W boson. It also shows the effect of including co-
variance matrices into one's calculations instead of just statistical
errors.

'''
from kafe import *

import matplotlib.pyplot as plt
import numpy as np

@FitFunction
def constant_1par(x, mean=80):    
    '''Constant Function'''
    return mean

# Load the Datasets from a file
myDataset_CM = parse_column_data('w_mittelung.dat',
                              field_order='x,y',
                              cov_mat_files=(None, 'w_mass.cov'),
                              title="W-Boson-Mass (mit KMen)")

myDataset_YE = parse_column_data('w_mittelung.dat',
                              field_order='x,y,yabsstat',
                              title="W-Boson-Masse (ohne KMen)")

# Create the Fits
myFit_CM = Fit(myDataset_CM, constant_1par, function_label="Mittelwert mit KMen")
myFit_CM.do_fit()

myFit_YE = Fit(myDataset_YE, constant_1par, function_label="Mittelwert ohne KMen")
myFit_YE.do_fit()

# Plot the Fits
myPlot = Plot(myFit_CM, myFit_YE)

# Label the axes
myPlot.axis_labels = ("Experiment Nr.", "W-Boson-Masse [GeV]")

# Draw the Fits
myPlot.plot_all()

# Save
myPlot.save("plot.pdf")

# Show
myPlot.show()
