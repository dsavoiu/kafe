'''
.. module:: function_library
   :platform: Unix
   :synopsis: A submodule containing a collection of model functions.
.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''

from numpy import exp, log, sqrt, pi
from scipy.special import gamma

##########################################
# Constant, Linear and Polynomial Models #
##########################################

# Constant Models

@FitFunction
def constant_1par(x, constant=1.0):
    return constant

# Linear Models

@FitFunction
def linear_1par(x, slope=1.0):
    return slope * x

@FitFunction
def linear_2par(x, slope=1.0, y_intercept=0.0):
    return slope * x + y_intercept

@FitFunction
def linear_2par2(x, slope=1.0, x_offset=0.0):
    return slope * (x - y_intercept)

# Quadratic

@FitFunction
def quadratic_1par(x, quad_coeff=1.0):
    return quad_coeff * x ** 2

@FitFunction
def quadratic_2par(x, quad_coeff=1.0, constant=0.0):
    return quad_coeff * x ** 2 + constant

@FitFunction
def quadratic_2par2(x, quad_coeff=1.0, x_offset=0.0):
    return quad_coeff * (x - x_offset) ** 2

@FitFunction
def quadratic_3par(x, quad_coeff=1.0, lin_coeff=0.0, constant=0.0):
    return quad_coeff * x ** 2 + lin_coeff * x + constant

@FitFunction
def quadratic_3par2(x, quad_coeff=1.0, x_offset=0.0, constant=0.0):
    return quad_coeff * (x - x_offset) ** 2 + constant

# Other Polynomials

@FitFunction
def poly3(x, coeff3=1.0, coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff3 * x ** 3 + coeff2 * x ** 2 + coeff1 * x + coeff0

@FitFunction
def poly4(x, coeff4=1.0, coeff3=0.0, coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff4 * x ** 4 + coeff3 * x ** 3 + coeff2 * x ** 2 + coeff1 * x + coeff0

@FitFunction
def poly5(x, coeff5=1.0, coeff4=0.0, coeff3=0.0, coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff5 * x ** 5 + coeff4 * x ** 4 + coeff3 * x ** 3 + coeff2 * x ** 2 + coeff1 * x + coeff0

######################
# Exponential Models #
######################

@FitFunction
def exp_2par(x, growth=1.0, constant_factor=0.0):
    return exp(growth * x) * constant_factor

@FitFunction
def exp_3par(x, growth=1.0, constant_factor=0.0, y_offset=0.0):
    return exp(growth * x) * constant_factor + y_offset

@FitFunction
def exp_3par2(x, growth=1.0, constant_factor=0.0, x_offset=0.0):
    return exp(growth * (x - x_offset)) * constant_factor

@FitFunction
def exp_4par(x, growth=1.0, constant_factor=0.0, x_offset=0.0, y_offset=0.0):
    return exp(growth * (x - x_offset)) * constant_factor + y_offset

###########################
# Other Non-Linear Models #
###########################

@FitFunction
def gauss(x, mean=0.0, sigma=1.0, scale=1.0):
    return 1.0 * scale / (sigma * sqrt(2*pi)) * exp(-(x-mean)**2/(2 * sigma ** 2))

@FitFunction
def poisson(x, mean=0.0, scale=1.0):
    return scale * mean ** x * exp(-mean) / gamma(x+1)

# Breit-Wigner needs work...
# def breit_wigner(x, mean=0.0, width=1.0, scale=1.0):
#     return scale * (x * width) ** 2 / ( (x ** 2 - mean ** 2) ** 2 + (x ** 2 * width / mean) ** 2 )
