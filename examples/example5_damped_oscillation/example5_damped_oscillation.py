'''
Damped Oscillation
------------------

    This example shows the fitting of a more complicated model function
    to data collected from a damped oscillation experiment.
    This also demonstrates how to set the initial values of the
    parameters to something other than defined in the function
    definition.
'''

###########
# Imports #
###########

# import everything we need from kafe
from kafe import *

# import some functions from numpy
from numpy import exp, cos


#############################
# Model function definition #
#############################

# Set an ASCII expression for this function
@ASCII(x_name="t", expression="A0*exp(-t/tau)*cos(omega*t+phi)")
# Set some LaTeX-related parameters for this function
@LaTeX(name='A', x_name="t",
       parameter_names=('a_0', '\\tau{}', '\\omega{}', '\\varphi{}'),
       expression="a_0\\,\\exp(-\\frac{t}{\\tau})\,"
                  "\cos(\\omega{}\\,t+\\varphi{})")
@FitFunction
def damped_oscillator(t, a0=1, tau=1, omega=1, phi=0):
    return a0 * exp(-t/tau) * cos(omega*t + phi)


############
# Workflow #
############

# load the experimental data from a file
my_dataset = parse_column_data(
    'damped_oscillation.dat',
    field_order="x,y,xabserr,yabserr",
    title="Damped Oscillator",
    axis_labels=['Time $t$','Amplitude'])

# Create the Fit
my_fit = Fit(my_dataset,
             damped_oscillator)
#            fit_label="Linear Regression " + dataset.data_label[-1])

# Set the initial values for the fit:
#                      a_0 tau omega phi
my_fit.set_parameters((1., 3., 6.28, 0.))

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
my_plot.save('kafe_example5.pdf')

# Show the plots
my_plot.show()
