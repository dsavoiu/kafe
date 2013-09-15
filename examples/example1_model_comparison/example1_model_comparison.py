'''
Comparison of two models for the same data
------------------------------------------

    In this example, two models (exponential and linear) are fitted
to data from a single Dataset.
'''

###########
# Imports #
###########

# import everything we need from kafe
from kafe import *

# additionally, import the two model functions we
# want to fit:
from kafe.function_library import linear_2par, exp_2par


####################
# Helper functions #
####################

def generate_dataset(output_file_path):
    '''The following block generates the Datasets and writes a file for
    each of them.'''

    import numpy as np  # need some functions from numpy

    n_p = 10
    xmin, xmax = 1, 10
    growth, constant = 0.12, 1.34
    sigma_x, sigma_y = 0.1, 0.1
    xdata = np.linspace(xmin, xmax, n_p) + np.random.normal(0.0, sigma_x, n_p)
    ydata = map(lambda x: exp_2par(x, growth, constant), xdata)
    ydata += np.random.normal(0.0, sigma_y, n_p)

    my_dataset = build_dataset(xdata, ydata,
                               xabsstat=sigma_x, yabsstat=sigma_y)
    my_dataset.write_formatted(output_file_path)


############
# Workflow #
############

# Generate the Dataset and store it in a file
#generate_dataset('dataset.dat')

# Load the Dataset from the file
my_dataset = Dataset(input_file='dataset.dat', title="Example Dataset")

# Create the Fits
my_fits = [Fit(my_dataset, exp_2par),
           Fit(my_dataset, linear_2par)]

# Do the Fits
for fit in my_fits:
    fit.do_fit()

# Create the plots
my_plot = Plot(my_fits[0], my_fits[1])

# Draw the plots
my_plot.plot_all(show_data_for=0)  # only show data once (it's the same data)

###############
# Plot output #
###############

# Save the plots
my_plot.save('plot.pdf')

# Show the plots
my_plot.show()
