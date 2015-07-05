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
import kafe
from kafe import ASCII, LaTeX, FitFunction
from kafe.file_tools import parse_column_data

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
def damped_oscillator(t, a0=1., tau=1., omega=1., phi=0.):
    return a0 * exp(-t/tau) * cos(omega*t + phi)

###################
# Helper function #
###################
def generate_dataset(outfile):
    '''The following block generates the data'''

    import numpy as np  # need some functions from numpy

    n_p = 20
    tmin, tmax = 0.2, 10.
    sigt=0.1
    sigArel=0.3
    tdat = np.linspace(tmin, tmax, n_p) + np.random.normal(0.0, sigt, n_p)
    a0=1.
    tau=2.
    omega=6.28
    phi=3.14/4.
    terr=[sigt for i in range(n_p)]
    Adat = map(lambda x: damped_oscillator(x, a0, tau, omega, phi), tdat)
    Aerr=sigArel*np.fabs(np.array(Adat))
    Adat *= np.random.normal(1.0, sigArel, n_p)
    np.savetxt(outfile,np.column_stack( (tdat, Adat, terr, Aerr) ))

############
# Workflow #
############

# Generate the Dataset and store it in a file
#generate_dataset('oscillation.dat')

# load the experimental data from a file
my_dataset = parse_column_data(
    'oscillation.dat',
    field_order="x,y,xabserr,yabserr",
    title="Damped Oscillator",
    axis_labels=['Time $t$','Amplitude'])

# Create the Fit
my_fit = kafe.Fit(my_dataset,
                  damped_oscillator)
#                 fit_label="Linear Regression " + dataset.data_label[-1])

# Set the initial values for the fit:
#                      a_0 tau omega phi
my_fit.set_parameters((1., 2., 6., 0.8))

# Do the Fits
my_fit.do_fit()

# Create the plots
my_plot = kafe.Plot(my_fit)

# Draw the plots
my_plot.plot_all()

###############
# Plot output #
###############

# Save the plots
my_plot.save('kafe_example5.pdf')

my_fit.plot_correlations()

# Show the plots
my_plot.show()
