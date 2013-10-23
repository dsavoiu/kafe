'''
Double slit experiment
----------------------

    This example fits a modified quadratic sinc function to data from a
    double slit experiment.
'''

###########
# Imports #
###########

# import everything we need from kafe
from kafe import *

# import some functions from numpy
from numpy import cos, sin

#############################
# Model function definition #
#############################

# Set an ASCII expression for this function
@ASCII(x_name="x", expression="I*(sin(k/2*b*sin(x))/(k/2*b*sin(x))"
                              "*cos(k/2*g*sin(x)))^2")
# Set some LaTeX-related parameters for this function
@LaTeX(name='f', x_name="\\alpha{}", 
       parameter_names=('I', 'b', 'g', 'k'),
       expression="I\\,\\left(\\frac{\\sin(\\frac{k}{2}\\,b\\,\\sin{\\alpha})}"
                  "{\\frac{k}{2}\\,b\\,\\sin{\\alpha}}"
                  "\\cos(\\frac{k}{2}\\,g\\,\\sin{\\alpha})\\right)^2")
@FitFunction
def double_slit(alpha, I=1, b=1, g=1, k=1):
    k_half_sine_alpha = k/2*sin(alpha)  # helper variable
    k_b = k_half_sine_alpha * b
    k_g = k_half_sine_alpha * g
    return I * (sin(k_b)/(k_b) * cos(k_g))**2


############
# Workflow #
############

# load the experimental data from a file
my_dataset = parse_column_data(
    'double_slit.dat',
    field_order="x,y,xabserr,yabserr",
    title="Double Slit Data"
)

# Create the Fit
my_fit = Fit(my_dataset,
             double_slit)
#            fit_label="Linear Regression " + dataset.data_label[-1])

# Set the initial values for the fit
#                      I   b      g      k 
my_fit.set_parameters((1., 15e-6, 45e-6, 12.57e6))

# Do the Fits
my_fit.do_fit()

# Create the plots
my_plot = Plot(my_fit)

# Draw the plots
my_plot.plot_all()

###############
# Plot output #
###############

# Save the plots
my_plot.save('plot.pdf')

# Show the plots
my_plot.show()
