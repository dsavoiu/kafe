'''
.. module:: function_library
    :platform: Unix
    :synopsis: A submodule containing a collection of model functions.
.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''

from function_tools import FitFunction, LaTeX, ASCII
from numpy import exp
from scipy.special import gamma

##########################################
# Constant, Linear and Polynomial Models #
##########################################


# Constant Models
@ASCII(expression='constant')
@LaTeX(name='f', parameter_names=('c',), expression='c')
@FitFunction
def constant_1par(x, constant=1.0):
    return constant


# Linear Models
@ASCII(expression='slope * x')
@LaTeX(name='f', parameter_names=('m',), expression='m\\,x')
@FitFunction
def linear_1par(x, slope=1.0):
    return slope * x


@ASCII(expression='slope * x + y_intercept')
@LaTeX(name='f', parameter_names=('m', 'n'), expression='m\\,x+n')
@FitFunction
def linear_2par(x, slope=1.0, y_intercept=0.0):
    return slope * x + y_intercept


@ASCII(expression='slope * (x - x_offset)')
@LaTeX(name='f', parameter_names=('m', 'x_0'), expression='m(x-x_0)')
@FitFunction
def linear_2par2(x, slope=1.0, x_offset=0.0):
    return slope * (x - x_offset)


# Quadratic
@ASCII(expression='quad_coeff * x^2')
@LaTeX(name='f', parameter_names=('a',), expression='a\\,x^2')
@FitFunction
def quadratic_1par(x, quad_coeff=1.0):
    return quad_coeff * x**2


@ASCII(expression='quad_coeff * x^2 + constant')
@LaTeX(name='f', parameter_names=('a', 'c'), expression='a\\,x^2+c')
@FitFunction
def quadratic_2par(x, quad_coeff=1.0, constant=0.0):
    return quad_coeff * x**2 + constant


@ASCII(expression='quad_coeff * (x - x_offset)^2')
@LaTeX(name='f', parameter_names=('a', 'x_0'), expression='a\\,(x-x_0)^2')
@FitFunction
def quadratic_2par2(x, quad_coeff=1.0, x_offset=0.0):
    return quad_coeff * (x - x_offset) ** 2


@ASCII(expression='quad_coeff * x^2 + lin_coeff * x + constant')
@LaTeX(name='f', parameter_names=('a', 'b', 'c'), expression='a\\,x^2+b\\,x+c')
@FitFunction
def quadratic_3par(x, quad_coeff=1.0, lin_coeff=0.0, constant=0.0):
    return quad_coeff * x**2 + lin_coeff * x + constant


@ASCII(expression='quad_coeff * (x - x_offset)^2 + constant')
@LaTeX(name='f', parameter_names=('a', 'x_0', 'c'),
       expression='a\\,(x-x_0)^2+c')
@FitFunction
def quadratic_3par2(x, quad_coeff=1.0, x_offset=0.0, constant=0.0):
    return quad_coeff * (x - x_offset) ** 2 + constant


# Other Polynomials
@ASCII(expression='coeff3 * x^3 + coeff2 * x^2 + coeff1 * x + coeff0')
@LaTeX(name='f', parameter_names=('a', 'b', 'c', 'd'),
       expression='a\\,x^3+b\\,x^2+c\\,x+d')
@FitFunction
def poly3(x, coeff3=1.0, coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff3 * x**3 + coeff2 * x**2 + coeff1 * x + coeff0


@ASCII(expression='coeff4 * x^4 + coeff3 * x^3 + coeff2 * x^2 + \
                   scoeff1 * x + coeff0')
@LaTeX(name='f', parameter_names=('a', 'b', 'c', 'd', 'e'),
       expression='a\\,x^4+b\\,x^3+c\\,x^2+d\\,x+e')
@FitFunction
def poly4(x, coeff4=1.0, coeff3=0.0, coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff4 * x**4 + coeff3 * x**3 + coeff2 * x**2 + coeff1 * x + coeff0


@ASCII(expression='coeff5 * x^5 + coeff4 * x^4 + coeff3 * x^3 + \
                   coeff2 * x^2 + coeff1 * x + coeff0')
@LaTeX(name='f', parameter_names=('a', 'b', 'c', 'd', 'e', 'f'),
       expression='a\\,x^5+b\\,x^4+c\\,x^3+d\\,x^2+e\\,x+f')
@FitFunction
def poly5(x, coeff5=1.0, coeff4=0.0, coeff3=0.0,
          coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff5 * x**5 + coeff4 * x**4 + coeff3 * x**3 + coeff2 * x**2 + \
        coeff1 * x + coeff0


######################
# Exponential Models #
######################

@ASCII(expression='exp(growth * x) * constant_factor')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0'),
       expression='a_0\\,\\exp(\\lambda x)')
@FitFunction
def exp_2par(x, growth=1.0, constant_factor=0.0):
    return exp(growth * x) * constant_factor


@ASCII(expression='exp(growth * x) * constant_factor + y_offset')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0', 'y_0'),
       expression='a_0\\,\\exp(\\lambda x)+y_0')
@FitFunction
def exp_3par(x, growth=1.0, constant_factor=0.0, y_offset=0.0):
    return exp(growth * x) * constant_factor + y_offset


@ASCII(expression='exp(growth * (x - x_offset)) * constant_factor')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0', 'x_0'),
       expression='a_0\\,\\exp(\\lambda(x-x_0))')
@FitFunction
def exp_3par2(x, growth=1.0, constant_factor=0.0, x_offset=0.0):
    return exp(growth * (x - x_offset)) * constant_factor


@ASCII(expression='exp(growth * (x - x_offset)) * constant_factor + y_offset')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0', 'x_0', 'y_0'),
       expression='a_0\\,\\exp(\\lambda(x-x_0))+y_0')
@FitFunction
def exp_4par(x, growth=1.0, constant_factor=0.0, x_offset=0.0, y_offset=0.0):
    return exp(growth * (x - x_offset)) * constant_factor + y_offset


###########################
# Other Non-Linear Models #
###########################

@ASCII(expression='scale * exp(-(x-mean)^2/(2 * sigma^2))')
@LaTeX(name='f', parameter_names=('\\mu{}', '\\sigma{}', 'a_0'),
       expression='a_0\\,\\exp(-\\frac{(x-\\mu)^2}{2\\sigma^2})')
@FitFunction
def gauss(x, mean=0.0, sigma=1.0, scale=1.0):
    return scale * exp(-(x-mean)**2/(2 * sigma ** 2))


@ASCII(expression='scale * mean^x * exp(-mean) / Gamma(x+1)')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0'),
       expression='a_0\\,\\lambda^x\\,\\frac{\\exp(-\\lambda)}{\\Gamma(x+1)}')
@FitFunction
def poisson(x, mean=0.0, scale=1.0):
    return scale * mean ** x * exp(-mean) / gamma(x+1)


# Breit-Wigner needs work...
# @FitFunction
# def breit_wigner(x, mean=0.0, width=1.0, scale=1.0):
#     return scale * (x * width)**2 / \
#        ((x**2 - mean**2)**2 + (x**2 * width / mean)**2)
