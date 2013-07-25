'''
.. module:: constants
   :platform: Unix
   :synopsis: A submodule defining some constants used by the fit package.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@ekp.kit.edu>
'''

# (G) Constants related to the graphical output
#################################################

G_PADDING_FACTOR_X = 1.2    #: factor by which to expand `x` data range
G_PADDING_FACTOR_Y = 1.2    #: factor by which to expand `y` data range
G_PLOT_POINTS = 200         #: number of plot points for plotting the function

# (M) Related to the behavior of TMinuit
#################################################

M_TOLERANCE = 0.1           #: `Minuit` tolerance level
M_MAX_ITERATIONS = 6000     #: Maximum `Minuit` iterations until aborting the process
M_MAX_X_FIT_ITERATIONS = 2  #: Number of maximal additional iterations for `x` fit (0 disregards `x` errors)
M_CONFIDENCE_LEVEL = 0.05   #: Confidence level for hypythesis test. A fit is rejected it :math:`\chi^2_\text{prob}` is smaller than this constant

# (F) Output number format preferences
#################################################

F_SIGNIFICANCE = 2
'''
Set significance for returning results and errors
N = rounding error to N significant digits and value
to the same order of magnitude as the error.
'''
                            
# Other constants
#################################################
