'''
A Tale of Two Fits 
------------------

    This simple example fits a linear function to two Datasets
and plots both Fits into the same Plot.
'''

from kafe import *
import numpy as np

@FitFunction
def linear_2par(x, slope=1, y_intercept=0):
    return slope * x + y_intercept

# The following block generates the Datasets and writes a file for
# each of them.

# my_datasets = []
#
# n_p = 10
# xmin, xmax = 3, 4
# slope, y_intercept = 3.44, 0.04
# sigma_x, sigma_y = 0.1, 0.3
# xdata = np.linspace(xmin,xmax,n_p) + np.random.normal(0.0, sigma_x, n_p)
# ydata = slope * xdata + [y_intercept]*n_p
# ydata += np.random.normal(0.0, sigma_y, n_p)
#  
# my_datasets.append(build_dataset(xdata, ydata, xabsstat=sigma_x, yabsstat=sigma_y)) 
#  
# n_p = 10
# xmin, xmax = 2, 3
# slope, y_intercept = 2.81, 0.13
# sigma_x, sigma_y = 0.05, 0.5
# xdata = np.linspace(xmin,xmax,n_p) + np.random.normal(0.0, sigma_x, n_p)
# ydata = slope * xdata + [y_intercept]*n_p
# ydata += np.random.normal(0.0, sigma_y, n_p)
#  
# my_datasets.append(build_dataset(xdata, ydata, xabsstat=sigma_x, yabsstat=sigma_y))
#  
# my_datasets[0].write_formatted('dataset1.dat')
# my_datasets[1].write_formatted('dataset2.dat')

# Load the Datasets from files
my_datasets = [
               Dataset(input_file='dataset1.dat', title="Example Dataset 1"),
               Dataset(input_file='dataset2.dat', title="Example Dataset 2")
              ]

# Create the Fits
my_fits = [Fit(dataset, linear_2par, function_label="Linear Regression "+dataset.data_label[-1]) for dataset in my_datasets]

# Do the Fits
[fit.do_fit() for fit in my_fits]

# Create the plots
my_plot = Plot(my_fits[0], my_fits[1])

# Draw the plots
my_plot.plot_all()

# Save the plots
my_plot.save('plot.pdf')

# Show the plots
my_plot.show()
