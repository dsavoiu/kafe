'''
W boson mass
------------

    This example shows the averaging of several experiments' results to
find the mass of the W boson. It also shows the effect of including co-
variance matrices into one's calculations instead of just statistical
errors.

'''

###########
# Imports #
###########

# import everything we need from kafe
from kafe import *

# additionally, import the model function we
# want to fit:
from kafe.function_library import constant_1par

############
# Workflow #
############

# Load the Datasets from a file (one-datapoint-per-row format)
myDataset_CM = parse_column_data('w_mittelung.dat',
                                 field_order='x,y',
                                 cov_mat_files=(None, 'w_mass.cov'),
                                 title="W-Boson-Mass (mit CovMats)",
                                 axis_labels=("Experiment Nr.",
                                              "W-Boson-Masse"),
                                 axis_units=(None, 'GeV'))

myDataset_YE = parse_column_data('w_mittelung.dat',
                                 field_order='x,y,yabserr',
                                 title="W-Boson-Masse (ohne CovMats)",
                                 axis_labels=("Experiment Nr.",
                                              "W-Boson-Masse"),
                                 axis_units=(None, 'GeV'))

# Create and do the Fits
myFit_CM = Fit(myDataset_CM, constant_1par,
               fit_label="Mittelwert mit CovMats")
myFit_CM.do_fit()

myFit_YE = Fit(myDataset_YE, constant_1par,
               fit_label="Mittelwert ohne CovMats")
myFit_YE.do_fit()

# Plot the Fits
myPlot = Plot(myFit_CM, myFit_YE)

# Label the axes
myPlot.axis_labels = ("Experiment Nr.", "W-Boson-Masse [GeV]")

# Draw the Fits
myPlot.plot_all()

###############
# Plot output #
###############

# Save
myPlot.save("plot.pdf")

# Show
myPlot.show()
