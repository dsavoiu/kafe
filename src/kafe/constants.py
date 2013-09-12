'''
.. module:: constants
   :platform: Unix
   :synopsis: A submodule defining some constants used by the fit package.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''

# (G) Constants related to the graphical output
#################################################

G_PADDING_FACTOR_X = 1.2    #: factor by which to expand `x` data range
G_PADDING_FACTOR_Y = 1.2    #: factor by which to expand `y` data range
G_PLOT_POINTS = 200         #: number of plot points for plotting the function

# (M) Related to the behavior of TMinuit
#################################################

M_TOLERANCE = 0.1           #: `Minuit` tolerance level
M_MAX_ITERATIONS = 6000     #: Maximum `Minuit` iterations until abort
M_MAX_X_FIT_ITERATIONS = 2  #: Maximum additional iterations for `x` fit
M_CONFIDENCE_LEVEL = 0.05   #: Confidence level for hypothesis test

# A fit is rejected if :math:`\chi^2_\text{prob}`
# is smaller than this constant.

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

D_DEBUG_MODE = False
