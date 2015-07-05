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
import kafe
from kafe import ASCII, LaTeX, FitFunction
from numpy import exp
import matplotlib.pyplot as plt

#############################
# Model function definition #
#############################

# Set an ASCII expression for this function
@ASCII(x_name="t", expression="A0*exp(-t/tau)")
# Set some LaTeX-related parameters for this function
@LaTeX(name='A', x_name="t",
       parameter_names=('A_0', '\\tau{}'),
       expression="A_0\\,\\exp(\\frac{-t}{\\tau})")
@FitFunction
def exponential(t, A0=1, tau=1):
    return A0 * exp(-t/tau)


####################
# Helper functions #
####################

def generate_dataset(output_file_path):
    '''The following block generates the Datasets and writes a file for
    each of them.'''

    import numpy as np  # need some functions from numpy

    n_p = 10
    xmin, xmax = 1, 5
    sigma_x, sigma_y = 0.3, 0.4
    xdata = np.linspace(xmin, xmax, n_p) + np.random.normal(0.0, sigma_x, n_p)

    A0 ,tau = 1., 1.
    ydata = map(lambda x: exponential(x, A0, tau), xdata)
    ydata *= np.random.normal(1.0, sigma_y, n_p)

    my_datasets.append(kafe.Dataset(data=(xdata, ydata)))
    my_datasets[-1].add_error_source('x', 'simple', sigma_x)
    my_datasets[-1].add_error_source('y', 'simple', sigma_y, relative=True)

    my_dataset.write_formatted(output_file_path)


############
# Workflow #
############

# Generate the Dataset and store it in a file
#generate_dataset('dataset.dat')

# Initialize the Dataset
my_dataset = kafe.Dataset(title="Example Dataset",
                          axis_labels=['t', 'A'] )

# Load the Dataset from the file
my_dataset.read_from_file(input_file='dataset.dat')


# Create the Fit
my_fit = kafe.Fit(my_dataset, exponential)

# Do the Fit
my_fit.do_fit()

# Create the plots
my_plot = kafe.Plot(my_fit, yscale='log', yscalebase=10)
#OR:
#my_plot.set_axis_scale('y', 'log', basey=2)

# Draw the plots
my_plot.plot_all()

###############
# Plot output #
###############

# Save the plots
my_plot.save('kafe_example3.pdf')

cor_fig = my_fit.plot_correlations()
cor_fig.savefig('kafe_example3_correlations.pdf')

# Show the plots
my_plot.show()
