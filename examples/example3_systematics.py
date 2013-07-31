'''
Visual of systematic error influence
-------------------------------------

This example compares a linear fit with statistical errors, the same fit with systematics, 
and a third time with statistics instead of systematics.

'''
from kafe import *
import numpy as np

def linear_2par(x, slope=1, y_intercept=0):
    '''Linear function with parameters of slope and y intercept'''
    return slope * x + y_intercept    

def exp_3par(x, damping=0.03, scale=1, y_offset=0):
    '''Exponential function with parameters of damping, scale and y offset'''
    return scale * np.exp(damping * x) + y_offset


# Define x-axis data
my_x_data  = np.linspace(0,20,15)    # fifty evenly-spaced points on the x axis

# Set the "true" model parameters (model: linear_3par)
true_slope = 0.706253
true_y_intercept = 1.1120 

# Generate y-axis data from model
my_y_data = map(lambda x: linear_2par(x, true_slope, true_y_intercept), my_x_data)

# Set "true" statistical error
y_stat_error = .1

# Set "true" systematic error (slightly larger than the statistical error)
y_syst_error = .3

# Scatter data according to y random error (add a normally distributed random value to each data point)
my_y_data += np.random.normal(0.0, y_stat_error, len(my_x_data))

# Scatter data according to y random error (add a normally distributed random value to all data point)
my_y_data = my_y_data + [np.random.normal(0.0, y_syst_error)] * len(my_x_data)

# Construct the Datasets
my_dataset_stat = build_dataset(xdata=my_x_data, 
                    ydata=my_y_data,
                    yabsstat=y_stat_error,
                    title="With $\\sigma_y = %g$" % y_stat_error)

my_dataset_statsyst = build_dataset(xdata=my_x_data, 
                    ydata=my_y_data,
                    yabsstat=y_stat_error,
                    yabssyst=y_syst_error,
                    title="With $\\sigma_y = %g$ and $\\Delta y = %f$" % (y_stat_error, y_syst_error))

my_dataset_statstat = build_dataset(xdata=my_x_data, 
                    ydata=my_y_data,
                    yabsstat=np.sqrt(y_stat_error**2 + y_syst_error**2),
                    title="With $\\sigma_{y,1} = %g$ and $\\sigma_{y,2} = %f$" % (y_stat_error, y_syst_error))

# Fit the exponential model to the data
my_fits = [ Fit(my_dataset_stat, linear_2par, function_label='Fit for $\\sigma_y$'),
            Fit(my_dataset_statsyst, linear_2par, function_label='Fit for $\\sigma_y$ + $\\Delta y$'),
            Fit(my_dataset_statstat, linear_2par, function_label='Fit for $\\sigma_{y,1}$ + $\\sigma_{y,2}$')
          ]

for fit in my_fits:
    fit.do_fit(quiet=True)

# Plot both fits in the same Plot
myPlot = Plot(my_fits[0], my_fits[1], my_fits[2])
myPlot.plot_all(show_info_for='all',    # include every fit in the parameter info box
                show_data_for=0)        # only show data once
 
# Show/Save the Plot
myPlot.show()