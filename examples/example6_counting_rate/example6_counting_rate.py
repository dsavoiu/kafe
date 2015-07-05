'''
Fit of a counting rate
----------------------

   This simple example demonstrates the fitting of a function to
   a counting rate with - possibly - a small number of entries
   or even bins with zero entries, where the errors are given
   a the sqare-root of the number of entries.
   In this example, however, the independent variable is the cosine
   of the angle, so here we show how to modify the name of the
   independent variable of the fit function to to reflect that.
'''

###########
# Imports #
###########

# import everything we need from kafe
import kafe
from kafe.file_tools import parse_column_data
from kafe.function_library import poly4
import numpy as np

# modify function's independent variable name to reflect its nature:
poly4.x_name = 'x=cos(t)'
poly4.latex_x_name = 'x=\\cos(\\theta)'

# import some functions from numpy
from numpy import exp, cos


############
# Workflow #
############

# load the experimental data from a file
my_dataset = parse_column_data(
    'counting_rate.dat',
    field_order="x,y,yabserr",
    title="Counting Rate per Angle"
)

### pre-fit
# error for bins with zero contents is set to 1.
covmat = my_dataset.get_cov_mat('y')

for i in range(0, len(covmat)):
    if covmat[i, i] == 0.:
        covmat[i, i] = 1.
my_dataset.set_cov_mat('y', covmat)  # write it back

# Create the Fit
my_fit = kafe.Fit(my_dataset, poly4)
#                 fit_label="Linear Regression " + dataset.data_label[-1])

# perform an initial fit with temporary errors (minimal output)
my_fit.call_minimizer(final_fit=False, verbose=False)

# set errors using model at pre-fit parameter values: sigma_i^2=cov[i,i]=n(x_i)
fdata = my_fit.fit_function.evaluate(my_fit.xdata,
                                     my_fit.current_parameter_values)
np.fill_diagonal(covmat, fdata)
my_fit.current_cov_mat = covmat  # use modified covariance matrix

#
### end pre-fit

# Do the Fit
my_fit.do_fit()

# Create the plots
my_plot = kafe.Plot(my_fit)
# -- set the axis labels
my_plot.axis_labels = ['$\\cos(\\theta)$', 'counting rate']
# -- set scale linear / log
my_plot.axes.set_yscale('linear')

# Draw the plots
my_plot.plot_all()

###############
# Plot output #
###############

# Save the plots
my_plot.save('plot.pdf')

#my_fit.plot_correlations()

# Show the plots
my_plot.show()
