'''
Plotting a Gaussian curve without data points
---------------------------------------------

	This example creates a dummy Dataset object whose points lie exactly
on a Gaussian curve. The Fit will then converge toward that very same
Gaussian. When plotting, the data points used to "support" the curve
can be skipped.
	This example shows how to use matplotlib further to annotate plots.

'''
from kafe import *
import numpy as np
import matplotlib.pyplot as plt
import math

def gauss_2par(x, mu=0.0, sigma=1.0):
    '''Gaussian distribution'''
    return 1.0 / (np.sqrt(2 * math.pi) * sigma) * np.exp( -( (x - mu) ** 2 / (2 * sigma ** 2)) )    


# Define x-axis data
my_x_data  = np.linspace(-3,3,20)    # twenty evenly-spaced points on the x axis, from -3 to 3

# Generate y-axis data from model
my_y_data = map(lambda x: gauss_2par(x, 0, 1), my_x_data)

# Construct the Datasets
my_dataset = build_dataset(
                    xdata=my_x_data, 
                    ydata=my_y_data,
                    title="Standard-Normalverteilung")

# Fit the model to the data
my_fit = Fit(my_dataset, gauss_2par, function_label='Standard-Normalverteilung')

# Don't call do_fit for this Fit. 
 
# Plot the Fit
myPlot = Plot(my_fit, show_legend=True)

# Instruct LaTeX to use the EulerVM package (optional, uncomment if needed)
#plt.rcParams.update({'text.latex.preamble': ['\\usepackage{eulervm}']})

myPlot.plot_all(show_info_for='all',   # include every fit in the parameter info box
                show_data_for=None)    # don't show the data points, just the curve

# choose an interval to highlight
section = np.linspace(-1, 1, 200)

# Use the axes peoperty of Plot objects to do some additional matplotlib things (annotations, etc.)
myPlot.axes.fill_between(section, map(my_fit.get_current_fit_function(), section), alpha='0.2', color=myPlot.plot_style.get_linecolor(0))
myPlot.axes.axvline(x=0., color='k', linewidth=2, ls='dashed', ymin=0.4)
myPlot.axes.axvline(x=0., color='k', linewidth=2, ls='dashed', ymax=0.35)
myPlot.axes.annotate("$\\mu=0$", xy=(0.0, 0.13), size=20, ha='center')
myPlot.axes.arrow(0,0, 1,0,length_includes_head=True, head_length=0.2, head_width=0.02, color='k')
myPlot.axes.arrow(1,0,-1,0,length_includes_head=True, head_length=0.2, head_width=0.02, color='k')
myPlot.axes.annotate("$\,\\sigma=1$", xy=(0.5, 0.04), size=20, ha='center', va='center')

# Save the Plot
myPlot.save('plot.pdf')

# Show the Plot
myPlot.show()
