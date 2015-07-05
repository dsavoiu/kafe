#!/usr/bin/env python
'''
Histogram Fit
--------------

   This simple example demonstrates the fitting of a function to
   a `numpy` histogram.
'''

###########
# Imports #
###########

# import everything we need from kafe
import kafe
from kafe.function_library import gauss
import numpy as np

###################
# Helper function #
###################

def generate_histogram(output_path, bins=50, N=500):
    '''The following block generates a histogram with data'''

    ### one-dimensional histogram ###
    def hist(data, bins=50, xlabel='x', ylabel='number of entries'):
        #import matplotlib.pyplot as plt
        bc, be = np.histogram(data, bins)  # histogram data
        bincent = (be[:-1] + be[1:])/2.
        #w=0.9*(be[1]-be[0])
        #plt.bar(bincent,bc,align='center',width=w,facecolor='b',alpha=0.75) #
        #plt.xlabel(xlabel) # ... for x ...
        #plt.ylabel(ylabel) # ... and y axes
        #plt.show()
        return bc, be

#  generate random normal-distributed numbers
    hdata = np.random.normal(size=N)

#  histogram data using numpy historgram, get constants & edges
    hconts, hedges = hist(hdata, bins=bins)
    bincenters = (hedges[:-1] + hedges[1:])/2.  # centres of bins

#  generate dataset with statistical errors of bin entries
    #hdataset = build_dataset(bincenters, hconts, yabserr=np.sqrt(hconts))
    hdataset = kafe.Dataset([bincenters, hconts])
    hdataset.add_error_source('y', 'simple', np.sqrt(hconts))
    hdataset.write_formatted(output_path)

############
# Workflow #
############

# Generate the Dataseta and store them in files
#generate_histogram('hdataset.dat',N=250)

# Initialize the Dataset
hdataset = kafe.Dataset(title="Data for example 9",
                        axis_labels=['x', 'entries'])

# Load the Datasets from file
hdataset.read_from_file('hdataset.dat')


# error for bins with zero contents is set to 1.
covmat = hdataset.get_cov_mat('y')
for i in range(0, len(covmat)):
    if covmat[i, i] == 0.:
        covmat[i, i] = 1.
hdataset.set_cov_mat('y', covmat)  # write it back

# Create the Fit instance
hfit = kafe.Fit(hdataset, gauss, fit_label="Fit of a Gaussian to histogram data")
#
# perform an initial fit with temporary errors (minimial output)
hfit.call_minimizer(final_fit=False, verbose=False)
#
# set errors using model at pre-fit parameter values: sigma_i^2=cov[i,i]=n(x_i)
fdata = hfit.fit_function.evaluate(hfit.xdata, hfit.current_parameter_values)
np.fill_diagonal(covmat, fdata)
hfit.current_cov_mat = covmat  # write it back new covariance matrix
#
# now do final fit with full output
hfit.do_fit()

# Create, draw, save and show plot
hplot = kafe.Plot(hfit)
hplot.plot_all()
hplot.save('kafe_example9.pdf')
hplot.show()
