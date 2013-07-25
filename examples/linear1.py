from kafe import Dataset, Fit, Plot
import numpy as np


def linear_2par2(x, slope=1, x_offset=0):
    '''Linear function with parameters of slope and x offset'''
    return slope * (x - x_offset)    


# Define x-axis data
myXData  = np.asarray([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

# Define two sets of y-axis data
myYData  = np.asarray([0.227216,    -0.159875,    0.0924984,    -0.299377,    0.307159,    1.15718,    0.701532,    0.661879,    0.536548,    0.874998,    1.02603])
myYData2 = np.asarray([-0.254749307136, -0.0983575484803, -0.108455145055, -0.317740870797, 0.112078239069, 0.238810848121, 0.285778651146, 0.498801157207, 0.220837168526, 0.587468379781, 0.517299353294])

# Define uncertainties of the data points
tstFloat = 0.3      # Uncertainty of each point in myYData
tstFloat2 = 0.15    # Uncertainty of each point in myYData2


# Constuct the Datasets
myDataset = Dataset(xdata=myXData, 
                    ydata=myYData,
                    yabsstat=tstFloat,
                    title="Example dataset")

myDataset2 = Dataset(xdata=myXData, 
                    ydata=myYData2,
                    yabsstat=tstFloat2,
                    title="Another example")

# Construct and do the fits
myFit = Fit(myDataset, linear_2par2, function_label='Linear fit for example data')
myFit.do_fit()

myFit2 = Fit(myDataset2, linear_2par2, function_label='Linear fit for other example data')
myFit2.do_fit()

# Plot both fits in the same Plot
myPlot = Plot(myFit, myFit2)
myPlot.plot_all()

# Show/Save the Plot
myPlot.show()
#myPlot.save('/home/daniel/Desktop/testfig.pdf')