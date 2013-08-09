'''
Visual of systematic error influence
-------------------------------------

This example compares a linear fit with statistical errors, the same fit with systematics, 
and a third time with statistics instead of systematics.

'''
from kafe import *
import numpy as np
import matplotlib.pyplot as plt
import math

def gauss_2par(x, mu=0.0, sigma=1.0):
    '''Gaussian distribution'''
    return 1.0 / (np.sqrt(2 * math.pi) * sigma) * np.exp( -( (x - mu) ** 2 / (2 * sigma ** 2)) )    


# Define x-axis data
my_x_data  = np.linspace(-3,3,20)    # fifty evenly-spaced points on the x axis

# Generate y-axis data from model
my_y_data = map(lambda x: gauss_2par(x, 0, 1), my_x_data)

# Construct the Datasets
my_dataset = build_dataset(
                    xdata=my_x_data, 
                    ydata=my_y_data,
                    title="Standard-Normalverteilung")

# Fit the exponential model to the data
my_fits = [
           Fit(my_dataset, gauss_2par, function_label='Standard-Normalverteilung')
          ]

# for fit in my_fits:
#     fit.do_fit(quiet=True)
 
# Plot both fits in the same Plot

myPlot = Plot(my_fits[0], show_legend=False)
plt.rcParams.update({'text.latex.preamble': ['\\usepackage{eulervm}']})
myPlot.plot_all(show_info_for=None,    # include every fit in the parameter info box
                show_data_for=None)    # only show data once

section = np.linspace(-1, 1, 200)

myPlot.axes.fill_between(section, map(my_fits[0].get_current_fit_function(), section), alpha='0.2', color=myPlot.plot_style.get_linecolor(0))
myPlot.axes.axvline(x=0., color='k', linewidth=2, ls='dashed', ymin=0.4)
myPlot.axes.axvline(x=0., color='k', linewidth=2, ls='dashed', ymax=0.35)
myPlot.axes.annotate("$\\mu=0$", xy=(0.0, 0.13), size=20, ha='center')#, xytext=(3.0, 0.3))
myPlot.axes.arrow(0,0, 1,0,length_includes_head=True, head_length=0.2, head_width=0.02, color='k')
myPlot.axes.arrow(1,0,-1,0,length_includes_head=True, head_length=0.2, head_width=0.02, color='k')
myPlot.axes.annotate("$\,\\sigma=1$", xy=(0.5, 0.04), size=20, ha='center', va='center')
# Show/Save the Plot
myPlot.show()