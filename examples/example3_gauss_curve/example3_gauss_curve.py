'''
Plotting a Gaussian curve without data points
---------------------------------------------

    This example creates a dummy Dataset object whose points lie exactly
on a Gaussian curve. The Fit will then converge toward that very same
Gaussian. When plotting, the data points used to "support" the curve
can be skipped.
    This example shows how to use matplotlib further to annotate plots.

'''

###########
# Imports #
###########

# import everything we need from kafe
from kafe import *

# some additional things we'll need
from numpy import sqrt, pi, exp, linspace
import matplotlib.pyplot as plt

###########################
# Fit function definition #
###########################


# Set an ASCII expression for this function
@ASCII(expression="1/(sqrt(2*pi)*sigma)*exp(-(x-mu)^2/(2*sigma^2))")
# Set some LaTeX-related parameters for this function
@LaTeX(name='\mathcal{N}', parameter_names=('\mu{}', '\sigma{}'),
       expression="\\frac{1}{\\sigma\\sqrt{2\\pi}}\\exp"
                  "(-\\frac{(x-\\mu)^2}{2\\sigma^2})")
# Declare that this is a fit function
@FitFunction
def gauss_2par(x, mu=0.0, sigma=1.0):
    '''Gaussian distribution'''
    norm_factor = 1.0 / (sqrt(2 * pi) * sigma)
    return exp(-((x - mu)**2 / (2 * sigma**2))) * norm_factor

############
# Workflow #
############

# Define x-axis data
my_x_data = linspace(-3, 3, 20)  # twenty evenly-spaced points on
                                 # the x axis, from -3 to 3

# Generate y-axis data from model
my_y_data = map(lambda x: gauss_2par(x, 0, 1), my_x_data)

# Construct the Datasets
my_dataset = build_dataset(xdata=my_x_data,
                           ydata=my_y_data,
                           title="Standard-Normalverteilung")

# Fit the model to the data
my_fit = Fit(my_dataset, gauss_2par,
             fit_label='Standard-Normalverteilung')

# Don't call do_fit for this Fit.

# Plot the Fit
myPlot = Plot(my_fit, show_legend=True)

# Instruct LaTeX to use the EulerVM package (optional, uncomment if needed)
#plt.rcParams.update({'text.latex.preamble': ['\\usepackage{eulervm}']})

# Draw the Plots
myPlot.plot_all(show_info_for='all',  # include every fit in the info box
                show_data_for=None)   # don't show the points, just the curve

#########################
# Further customization #
#########################

# Use the axes property of Plot objects to do some
# additional matplotlib things (annotations, etc.)

# First, fill a portion of the area beneath
# the curve with a distrinctive color
section = linspace(-1, 1, 200)  # choose an interval to highlight
myPlot.axes.fill_between(
    section,
    map(my_fit.get_current_fit_function(), section),  # upper bound of fill
    alpha='0.2',  # fill with a transparency of 20%
    color=myPlot.plot_style.get_linecolor(0)  # same fill color as the line
)

# Draw lines to the left and right of the highlighted region
myPlot.axes.axvline(x=0., color='k', linewidth=2, ls='dashed', ymin=0.4)
myPlot.axes.axvline(x=0., color='k', linewidth=2, ls='dashed', ymax=0.35)

# Add some annotations to the plot
myPlot.axes.annotate("$\\mu=0$", xy=(0.0, 0.13), size=20, ha='center')
myPlot.axes.annotate("$\,\\sigma=1$", xy=(0.5, 0.04), size=20,
                     ha='center', va='center')

# Add some arrows to the plot (to mark the 1-sigma-width of the curve)
myPlot.axes.arrow(0, 0,  1, 0, length_includes_head=True,
                  head_length=0.2, head_width=0.02, color='k')
myPlot.axes.arrow(1, 0, -1, 0, length_includes_head=True,
                  head_length=0.2, head_width=0.02, color='k')

###############
# Plot output #
###############

# Save the Plot
myPlot.save('plot.pdf')

# Show the Plot
myPlot.show()
