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
@ASCII(x_name="x", expression="I0*(sin(k/2*b*sin(x))/(k/2*b*sin(x))"
                              "*cos(k/2*g*sin(x)))^2")
# Set some LaTeX-related parameters for this function
@LaTeX(name='I', x_name="\\alpha{}",
       parameter_names=('I_0', 'b', 'g', 'k'),
       expression="I_0\\,\\left(\\frac{\\sin(\\frac{k}{2}\\,b\\,\\sin{\\alpha})}"
                  "{\\frac{k}{2}\\,b\\,\\sin{\\alpha}}"
                  "\\cos(\\frac{k}{2}\\,g\\,\\sin{\\alpha})\\right)^2")
@FitFunction
def double_slit(alpha, I0=1, b=10e-6, g=20e-6, k=1.e7):
    k_half_sine_alpha = k/2*sin(alpha)  # helper variable
    k_b = k_half_sine_alpha * b
    k_g = k_half_sine_alpha * g
    return I0 * (sin(k_b)/(k_b) * cos(k_g))**2

############
# Workflow #
############

# load the experimental data from a file
my_dataset = parse_column_data('double_slit.dat',
                field_order="x,y,xabserr,yabserr",
                title="Double Slit Data",
                axis_labels=['$\\alpha$','Intensity'],
                basename='double_slit_withConstraint' )

# Create the Fit
my_fit = Fit(my_dataset,
             double_slit, minimizer_to_use='ROOT')
#            fit_label="Linear Regression " + dataset.data_label[-1])

# Set the initial values for the fit
#                      I   b      g        k
my_fit.set_parameters([1., 20e-6, 50e-6, 1e7])
# g, b and k cannot all be determined simultaneously from data,
my_fit.constrain_parameters(['k'],[9.67e6], [1.e4])

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

my_fit.plot_correlations()


# Show the plots
my_plot.show()
