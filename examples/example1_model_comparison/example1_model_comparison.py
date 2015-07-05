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
    growth, constant = 0.15, 1.3
    sigma_x, sigma_y = 0.3, 0.2
    xdata = np.linspace(xmin, xmax, n_p) + np.random.normal(0.0, sigma_x, n_p)
    ydata = map(lambda x: exp_2par(x, growth, constant), xdata)
    ydata += np.random.normal(0.0, sigma_y, n_p)

    my_dataset = kafe.Dataset(data=(xdata, ydata))
    my_dataset.add_error_source('x', 'simple', sigma_x)
    my_dataset.add_error_source('y', 'simple', sigma_y)
    
    my_dataset.write_formatted(output_file_path)


############
# Workflow #
############

# Generate the Dataset and store it in a file
#generate_dataset('dataset.dat')

# Initialize Dataset
my_dataset = kafe.Dataset(title="Example Dataset")

# Load the Dataset from the file
my_dataset.read_from_file('dataset.dat')

#print my_dataset.get_cov_mat(0)
#print my_dataset.get_cov_mat(1)

# Create the Fits
my_fits = [kafe.Fit(my_dataset, exp_2par),
           kafe.Fit(my_dataset, linear_2par)]

# Do the Fits
for fit in my_fits:
    fit.do_fit()

# Create the plots
my_plot = kafe.Plot(my_fits[0], my_fits[1])

# Draw the plots
my_plot.plot_all(show_data_for=0)  # only show data once (it's the same data)

###############
# Plot output #
###############

# Save the plots
my_plot.save('kafe_example1.pdf')

# check contours
contour1 = my_fits[0].plot_contour(0, 1, dchi2=[1.,2.3])
profile00=my_fits[0].plot_profile(0)
profile01=my_fits[0].plot_profile(1)
contour2 = my_fits[1].plot_contour(0, 1, dchi2=[1.,2.3])

#contour1.savefig('kafe_example1_contour1.pdf')
#contour2.savefig('kafe_example1_contour2.pdf')
#profile00.savefig('kafe_example1_profile00.pdf')
#profile01.savefig('kafe_example1_profile01.pdf')

# Show the plots
my_plot.show()
