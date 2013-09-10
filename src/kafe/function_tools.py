'''
.. module:: function_tools
   :platform: Unix
   :synopsis: This submodule contains several useful tools for getting
   information about a function, including the number, names and default
   values of its parameters and its derivatives with respect to the independent
   variable or the parameters.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

'''

import numpy as np

# get a numerical derivative calculating function from SciPy
from scipy.misc import derivative as scipy_der
from latex_tools import ascii_to_latex_math
from inspect import getsourcelines

##################
# Useful methods #
##################


def derivative(func, derive_by_index, variables_tuple, derivative_spacing):
    r'''
    Gives :math:`\frac{\partial f}{\partial x_k}` for :math:`f = f(x_0, x_1,
    \ldots)`. `func` is :math:`f`, `variables_tuple` is :math:`\{x_i\}` and
    `derive_by_index` is :math:`k`.
    '''

    # define a dummy function, so that the variable
    # by which f is to be derived is the only variable
    def tmp_func(derive_by_var):
        argument_list = []
        for arg_nr, arg_val in enumerate(variables_tuple):
            if arg_nr == derive_by_index:
                argument_list.append(derive_by_var)
            else:
                argument_list.append(arg_val)
        return func(*argument_list)

    # return the derivative of that function
    return scipy_der(tmp_func, variables_tuple[derive_by_index],
                     dx=derivative_spacing)


def outer_product(input_array):
    r'''
    Takes a `NumPy` array and returns the outer (dyadic, Kronecker) product
    with itself. If `input_array` is a vector :math:`\mathbf{x}`, this returns
    :math:`\mathbf{x}\mathbf{x}^T`.
    '''
    la = len(input_array)

    # return outer product as numpy array
    return np.kron(input_array, input_array).reshape(la, la)

##############
# Decorators #
##############


# Main decorator for fit functions
class FitFunction:
    """
    Decorator class for fit functions. If a function definition is decorated
    using this class, some information is collected about the function which is
    relevant to the fitting process, such as the number of parameters, their
    names and default values. Some details pertaining to display and
    representation are also set, such as :math:`\LaTeX{}` representations of
    the parameter names and the function name. Other decorators can be applied
    to a function object to specify things such as a :math:`\LaTeX{}` or
    plain-text expression for the fit function.
    """

    def __init__(self, f):
        self.f = f

        # Numeric properties of the function
        #: The number of parameters
        self.number_of_parameters = f.func_code.co_argcount-1

        #: The default values of the parameters
        self.parameter_defaults = f.func_defaults

        #: string object holding the source code for the fit-function

        self.sourcelines = getsourcelines(f)  # (for REALLY explicit docs)

        # String properties in ASCII/plaintext format
        self.name = f.__name__  #: The name of the function
        #: The names of the parameters
        self.parameter_names = f.func_code.co_varnames[
            1:f.func_code.co_argcount
        ]
        #: The name given to the independent variable
        self.x_name = f.func_code.co_varnames[0]

        #: a math expression (string) representing the function's result
        self.expression = None

        # String properties in LaTeX format
        #: The function's name in :math:`\LaTeX{}`
        self.latex_name = ascii_to_latex_math(self.name)
        #: A list of parameter names in :math:`\LaTeX{}`
        self.latex_parameter_names = [ascii_to_latex_math(name)
                                      for name in self.parameter_names]
        #: A :math:`\LaTeX{}` symbol for the independent variable.
        self.latex_x_name = ascii_to_latex_math(self.x_name)
        #: a :math:`\LaTeX{}` math expression, the function's result
        self.latex_expression = None

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def derive_by_x(self, x_0, derivative_spacing, parameter_list):
        r'''
        If `x_0` is iterable, gives the array of derivatives of a function
        :math:`f(x, par_1, par_2, \ldots)` around :math:`x = x_i` at every
        :math:`x_i` in :math:`\vec{x}`. If `x_0` is not iterable, gives the
        derivative of a function :math:`f(x, par_1, par_2, \ldots)` around
        :math:`x = \verb!x_0!`.
        '''
        try:
            iterator_over_x_0 = iter(x_0)  # try to get an iterator object
        except TypeError:
            # object is not iterable, return the derivative in x_0 (float)
            return scipy_der(self.f, x_0,
                             args=parameter_list, dx=derivative_spacing)
        else:
            # object is iterable, go through it and derive at each x_0 in it
            output_list = []
            for x in iterator_over_x_0:
                # call recursively
                output_list.append(
                    self.derive_by_x(x, derivative_spacing,
                                     parameter_list)
                )

            return np.asarray(output_list)

    def derive_by_parameters(self, x_0, derivative_spacing, parameter_list):
        r'''
        Returns the gradient of `func` with respect to its parameters, i.e.
        with respect to every variable of `func` except the first one.
        '''
        output_list = []

        # compile all function arguments into a variables tuple
        variables_tuple = tuple([x_0] + list(parameter_list))

        # go through all arguments except the first one
        for derive_by_index in xrange(1, self.f.func_code.co_argcount):
            output_list.append(
                derivative(self.f, derive_by_index,
                           variables_tuple, derivative_spacing)
            )

        return np.asarray(output_list)

    def get_function_equation(self, equation_format='latex',
                              equation_type='full', ensuremath=True):
        r'''
        Returns a string representing the function equation. Supported formats
        are :math:`\LaTeX{}` and ASCII inline math. Note that :math:`\LaTeX{}`
        math is wrapped by default in an ``\ensuremath{}`` expression. If this
        is not desired behaviour, the flag ``ensuremath`` can be set to
        ``False``.

        *equation_format* : string (optional)
            Can be either "latex" (default) or "ascii".

        *equation_type* : string (optional)
            Can be either "full" (default) or "short". A "short"-type equation
            limits itself to the function name and variables::

                f(x, par1, par2)

            A "full"-type equation includes the expression which the function
            calculates::

                f(x, par1, par2) = par1 * x + par2

        *ensuremath* : boolean (optional)
            If a :math:`\LaTeX{}` math equation is requested, ``True``
            (default) will wrap the resulting expression in an
            ``\ensuremath{}`` tag. Otherwise no wrapping is done.
        '''

        if equation_format == 'latex':
            tmp_dict = {'name': self.latex_name, 'xname': self.latex_x_name,
                        'paramstring': ",~".join(self.latex_parameter_names),
                        'expr': self.latex_expression}
        elif equation_format == 'ascii':
            tmp_dict = {'name': self.name, 'xname': self.x_name,
                        'paramstring': ", ".join(self.parameter_names),
                        'expr': self.expression}
        else:
            raise Exception("Unknown function equation format: \
            Expected `latex` or `ascii`, got `%s`" % (equation_format,))

        if equation_type == 'full':
            if tmp_dict['expr'] is None:
                tmp_dict['expr'] = "<expression not specified>"
            return "%(name)s(%(xname)s, %(paramstring)s) = %(expr)s" % tmp_dict
        elif equation_type == 'short':
            return "%(name)s(%(xname)s, %(paramstring)s)" % tmp_dict
        else:
            raise Exception("Unknown function equation type: \
            Expected `full` or `short`, got `%s`." % (equation_type,))
