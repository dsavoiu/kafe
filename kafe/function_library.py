'''
.. module:: function_library
   :platform: Unix
   :synopsis: A submodule containing a collection of model functions.
.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>>

Collection of model functions
'''


from .function_tools import FitFunction, LaTeX, ASCII
from numpy import exp, sqrt, pi
from scipy.special import gamma, wofz

#######################################################################
# Change-log:
# GQ: 20-Aug-14: added relativistic Breit-Wigner, Lorentz, Voigt and
#                and nomalized Gauss
#######################################################################


##########################################
# Constant, Linear and Polynomial Models #
##########################################

'''
- Constant Models
'''

@ASCII(expression='constant')
@LaTeX(name='f', parameter_names=('c',), expression='c')
@FitFunction
def constant_1par(x, constant=1.0):
    return constant

'''
- Linear Models
'''
@ASCII(expression='slope * x')
@LaTeX(name='f', parameter_names=('m',), expression='m\,x')
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


'''
- Quadratic models
'''
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


'''
- Other Polynomials
'''
@ASCII(expression='coeff3 * x^3 + coeff2 * x^2 + coeff1 * x + coeff0')
@LaTeX(name='f', parameter_names=('a', 'b', 'c', 'd'),
       expression='a\\,x^3+b\\,x^2+c\\,x+d')
@FitFunction
def poly3(x, coeff3=1.0, coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff3 * x**3 + coeff2 * x**2 + coeff1 * x + coeff0


@ASCII(expression='coeff4 * x^4 + coeff3 * x^3 + coeff2 * x^2 + '
                  'coeff1 * x + coeff0')
@LaTeX(name='f', parameter_names=('a', 'b', 'c', 'd', 'e'),
       expression='a\\,x^4+b\\,x^3+c\\,x^2+d\\,x+e')
@FitFunction
def poly4(x, coeff4=1.0, coeff3=0.0, coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff4 * x**4 + coeff3 * x**3 + coeff2 * x**2 + coeff1 * x + coeff0


@ASCII(expression='coeff5 * x^5 + coeff4 * x^4 + coeff3 * x^3 + '
                  'coeff2 * x^2 + coeff1 * x + coeff0')
@LaTeX(name='f', parameter_names=('a', 'b', 'c', 'd', 'e', 'f'),
       expression='a\\,x^5+b\\,x^4+c\\,x^3+d\\,x^2+e\\,x+f')
@FitFunction
def poly5(x, coeff5=1.0, coeff4=0.0, coeff3=0.0,
          coeff2=0.0, coeff1=0.0, coeff0=0.0):
    return coeff5 * x**5 + coeff4 * x**4 + coeff3 * x**3 + coeff2 * x**2 + \
        coeff1 * x + coeff0


'''
- Exponential Models #
'''

@ASCII(expression='exp(growth * x) * constant_factor')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0'),
       expression='a_0\\,\\exp(\\lambda x)')
@FitFunction
def exp_2par(x, growth=1.0, constant_factor=1.0):
    return exp(growth * x) * constant_factor


@ASCII(expression='exp(growth * x) * constant_factor + y_offset')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0', 'y_0'),
       expression='a_0\\,\\exp(\\lambda x)+y_0')
@FitFunction
def exp_3par(x, growth=1.0, constant_factor=1.0, y_offset=0.0):
    return exp(growth * x) * constant_factor + y_offset


@ASCII(expression='exp(growth * (x - x_offset)) * constant_factor')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0', 'x_0'),
       expression='a_0\\,\\exp(\\lambda(x-x_0))')
@FitFunction
def exp_3par2(x, growth=1.0, constant_factor=1.0, x_offset=0.0):
    return exp(growth * (x - x_offset)) * constant_factor


@ASCII(expression='exp(growth * (x - x_offset)) * constant_factor + y_offset')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0', 'x_0', 'y_0'),
       expression='a_0\\,\\exp(\\lambda(x-x_0))+y_0')
@FitFunction
def exp_4par(x, growth=1.0, constant_factor=1.0, x_offset=0.0, y_offset=0.0):
    return exp(growth * (x - x_offset)) * constant_factor + y_offset


'''
- Other Non-Linear Models
'''

@ASCII(expression='scale * exp(-(x-mean)^2/(2 * sigma^2))')
@LaTeX(name='f', parameter_names=('\\mu{}', '\\sigma{}', 'a_0'),
       expression='a_0\\,\\exp(-\\frac{(x-\\mu)^2}{2\\sigma^2})')
@FitFunction
def gauss(x, mean=0.0, sigma=1.0, scale=1.0):
    return scale * exp(-(x-mean)**2/(2 * sigma ** 2))

# normalized Gauss distribution
@ASCII(expression='scale/(sqrt(2pi)*sigma)*exp(-(x-mean)^2/(2 * sigma^2))')
@LaTeX(name='f', parameter_names=('\\mu{}', '\\sigma{}', 'a_0'),
  expression='\\frac{a_0}{\\sqrt{2\\pi}\\,\\sigma}\\,\\exp(-\\frac{(x-\\mu)^2}{2\\sigma^2})')
@FitFunction
def normgauss(x, mean=0.0, sigma=1.0, scale=1.0):
    return scale / sqrt(2.*pi)/sigma * exp(-(x-mean)**2/(2 * sigma ** 2))


@ASCII(expression='scale * mean^x * exp(-mean) / Gamma(x+1)')
@LaTeX(name='f', parameter_names=('\\lambda{}', 'a_0'),
       expression='a_0\\,\\lambda^x\\,\\frac{\\exp(-\\lambda)}{\\Gamma(x+1)}')
@FitFunction
def poisson(x, mean=0.0, scale=1.0):
    return scale * mean ** x * exp(-mean) / gamma(x+1)



# Lorentz curve (or Cauchy-Distribution)
#
@ASCII(expression='scale * gamma / (pi *((x-x0)^2 + gamma^2)')
@LaTeX(name='f', parameter_names=('x_0','\\gamma','scale'),
              expression='\\frac{\\gamma}'
              '{ \\pi\\,((x-x_0)^2 + \\gamma^2)}' )
@FitFunction
def lorentz(x, x0=0., gamma=1., scale=1.):
  return scale * gamma / (pi * ((x-x0)*(x-x0) + gamma*gamma))

# relativistic Breit-Wigner:
#
@ASCII(expression='s*M^2*G^2/[(x^2-M^2)^2+(G^2*M^2)]')
@LaTeX(name='f', parameter_names=('\\sigma_0', 'M_Z','\\Gamma_Z'),
expression='\\frac{\\sigma_0\\, M_Z^2\\Gamma^2}'
           '{ ((x^2-M_Z^2)^2+(\\Gamma^2 \\cdot M_Z^2))}' )
@FitFunction
def breit_wigner(x, M=91.0, G=2.0, s0=40.0):
   return s0*M*M*G*G/((x*x-M*M)**2+(G*G*M*M))

# relativistic Breit-Wigner with s-dependent width
#
@ASCII(expression='s0*x^2*G^2/[(x^2-M^2)^2+(x^4*G^2/M^2)]')
@LaTeX(name='f', parameter_names=('\\sigma_0', 'M_Z','\\Gamma_Z'),
expression='\\frac{\\sigma_0\\, M_Z^2\\Gamma^2}'
                 '{((x^2-M_Z^2)^2+(x^4\\Gamma^2 / M_Z^2))}')
@FitFunction
def breit_wigner2(x, M=91.2, G=2.5, s0=41.0):
   return s0*x*x*G*G/((x*x-M*M)**2+(x**4*G*G/(M*M)))

# Voigt function (Lorentz folded with Gauss)
@ASCII(expression='Lorentz(x,x0,gamma) folded w. Gauss(x,0,sigma)')
@LaTeX(name='f', parameter_names=('scale', 'x_0','\\gamma,\\sigma'),
       expression='{Lorentz(x,x_0,\\gamma) \\, \\oplus \\, Gauss(x\',0,\\sigma)}')
@FitFunction
def voigt(x, x0=0., gamma=1., sigma=0.01, scale=1.):
  if gamma!=0. and sigma/gamma < 1.e-16:
      return lorentz(x, x0, gamma, scale)
  else:
    return scale*wofz(((x - x0)+1.j*gamma)/sqrt(2.)/sigma).real/(sqrt(2.*pi)*sigma)

