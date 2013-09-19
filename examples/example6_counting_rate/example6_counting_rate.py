'''
Counting rate
------------------

    This example demonstrated the fitting of a 4th-degree polynomial,
    which is in the function library provided by kafe.
    For this example, however, the independent variable is the cosine
    of the angle, so here we show that modification of the function
    is possible to reflect that.
    
'''

###########
# Imports #
###########

# import everything we need from kafe
from kafe import *
from kafe.function_library import poly4

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
    field_order="x,y,yabsstat",
    title="Counting Rate per Angle"
)

# Create the Fit
my_fit = Fit(my_dataset,
             poly4)
#            fit_label="Linear Regression " + dataset.data_label[-1])

# Do the Fits
my_fit.do_fit()

# Create the plots
my_plot = Plot(my_fit)

# Set the axis labels
my_plot.axis_labels = ['$\\cos(\\theta)$', '$y$']

# Draw the plots
my_plot.plot_all()

###############
# Plot output #
###############

# Save the plots
my_plot.save('plot.pdf')

# Show the plots
my_plot.show()
