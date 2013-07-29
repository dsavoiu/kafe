'''
Compare two separate models for a fit
-------------------------------------

This example shows how

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
my_x_data  = np.linspace(0,100,50)    # fifty evenly-spaced points on the x axis

# Set the "true" model parameters (model: exp_3par)
true_damping = 0.01627
true_scale = 0.706253
true_y_offset = 1.1120 

# Generate y-axis data from model
my_y_data = map(lambda x: exp_3par(x, true_damping, true_scale, true_y_offset), my_x_data)

# Set "true" error
y_error = 0.1

# Scatter data according to y error (add a normally distributed random value to the data)
my_y_data += np.random.normal(0.0, y_error, len(my_x_data))

# Construct the Dataset
my_dataset = build_dataset(xdata=my_x_data, 
                    ydata=my_y_data,
                    yabsstat=y_error,
                    title="From exponential model")


# Fit the linear and the exponential model to the data
my_fit_linear = Fit(my_dataset, linear_2par, function_label='Linear fit')
my_fit_linear.do_fit()

my_fit_exponential = Fit(my_dataset, exp_3par, function_label='Exponential fit')
my_fit_exponential.do_fit()

# Plot both fits in the same Plot
myPlot = Plot(my_fit_linear, my_fit_exponential)
myPlot.plot_all(show_info_for='all',    # include every fit in the parameter info box
                show_data_for=0)        # only show data once

# Show/Save the Plot
myPlot.show()