'''
.. module:: function_tools
   :platform: Unix
   :synopsis: This submodule contains several useful tools for getting
     information about a function, including the number, names and default
     values of its parameters and its derivatives with respect to the
     independent variable or the parameters.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>
'''
# -----------------------------------------------------------------
# Changes:
# GQ 140803: precision for derive_by_parameteres now takes
#            as spacing 1% of error, or 1e-7 if error is 0;
#            same done for derive_by_x (to account for large
#             variations in input values and errors)
# GQ 140815  removed `getsourcelines`, as this was not compatible
#              with module import from a string
# GQ 140817 addes method `evaluate` to FitFunction

import numpy as np

# get a numerical derivative calculating function from SciPy
from scipy.misc import derivative as scipy_der
from .latex_tools import ascii_to_latex_math
#GQ from inspect import getsourcelines

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')


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
    #  by which f is to be derived is the only variable
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
                      dx=derivative_spacing )


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
    using this class, some information is collected about the function which
    is relevant to the fitting process, such as the number of parameters,
    their names and default values. Some details pertaining to display and
    representation are also set, such as :math:`LaTeX` representations of
    the parameter names and the function name. Other decorators can be applied
    to a function object to specify things such as a :math:`LaTeX` or
    plain-text expression for the fit function.
    """

    def __init__(self, f):
        self.f = f

        # Numeric properties of the function
        #: The number of parameters
        try:
            self.number_of_parameters = f.func_code.co_argcount - 1
        except AttributeError:
            self.number_of_parameters = f.__code__.co_argcount - 1

        #: The default values of the parameters
        try:
            self.parameter_defaults = f.func_defaults
        except AttributeError:
            self.parameter_defaults = f.__defaults__

        #: string object holding the source code for the fit-function
        #GQ self.sourcelines = getsourcelines(f)  # (for REALLY explicit docs)

        # String properties in ASCII/plaintext format
        self.name = f.__name__  #: The name of the function

        # Check if all parameters have default values
        if self.parameter_defaults is None:
            # no parameters have a default value
            logger.warn("Default values not provided for any parameters "
                        "of function <%s>. Setting all default parameter "
                        "values to 1.0." % (self.name,))
            self.parameter_defaults = tuple(self.number_of_parameters * [1.0])
        elif len(self.parameter_defaults) != self.number_of_parameters:
            # only the last few parameters have a default value
            # -> pad the default parameter tuple with zeros
            logger.warn("Number of default parameters given for "
                        "function <%s> does not match total "
                        "parameter number " % (self.name,))
            logger.info("Setting default parameter values for function <%s> "
                        "to 1.0 where omitted." % (self.name,))
            N = self.number_of_parameters  # shortcut
            # get default parameters as list
            defpar = list(self.parameter_defaults)
            # pad the list with 1.0 to the left
            defpar = [1.0]*(N-len(defpar))+defpar
            self.parameter_defaults = tuple(defpar)  # cast back to tuple

        #: The names of the parameters
        try:
            self.parameter_names = f.func_code.co_varnames[
                1:f.func_code.co_argcount
            ]
            #: The name given to the independent variable
            self.x_name = f.func_code.co_varnames[0]
        except AttributeError:
            self.parameter_names = f.__code__.co_varnames[
                1:f.__code__.co_argcount
            ]
            #: The name given to the independent variable
            self.x_name = f.__code__.co_varnames[0]


        #: a math expression (string) representing the function's result
        self.expression = None

        # String properties in LaTeX format
        #: The function's name in :math:`LaTeX`
        self.latex_name = ascii_to_latex_math(self.name)
        #: A list of parameter names in :math:`LaTeX`
        self.latex_parameter_names = [ascii_to_latex_math(name)
                                      for name in self.parameter_names]
        #: A :math:`LaTeX` symbol for the independent variable.
        self.latex_x_name = ascii_to_latex_math(self.x_name, monospace=False)
        #: a :math:`LaTeX` math expression, the function's result
        self.latex_expression = None

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def evaluate(self, x_0, parameter_list):
        r'''
        Evaluate the fit function at an x-value or at an array of
        x-values for the parameter values in `prarameter_list`.

          **x_0** float or array of floats

          **parameter_list** values of function parameters

          **returns** function value(s)
        '''

        try:
            iterator_over_x_0 = iter(x_0)  # try to get an iterator object
        except TypeError:
            # object is not iterable, return the function value at x_0 (float)
            return self.f(x_0, *parameter_list)
        else:
            # object is iterable, return array
            def tempf(x): # define helper function of x only to apply map()
              return self.f(x, *parameter_list)
            # use python map to calculate function values at each x
            return np.asarray(list(map(tempf, x_0) ))


    def derive_by_x(self, x_0, precision_list, parameter_list):
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
                             args=parameter_list, dx=precision_list)
        else:
            # object is iterable, go through it and derive at each x_0 in it
            output_list = []
            for i, x in enumerate(x_0):
                output_list.append(
                    self.derive_by_x(x, precision_list[i], parameter_list) )

            return np.asarray(output_list)

    def derive_by_parameters(self, x_0, precision_spec, parameter_list):
        r'''
        Returns the gradient of `func` with respect to its parameters, i.e.
        with respect to every variable of `func` except the first one.

        **precision_spec** : ``float`` or iterable of ``floats``
            An array of floats indicating the initial point spacing for
            numerically evaluating the derivative. Can be a single float
            value to use the same spacing for every derivation.
        '''
        output_list = []

        try:
            iter(precision_spec)
        except TypeError:
            # precision_spec is not iterable, create array
            try:
                precision_spec = np.ones(self.f.func_code.co_argcount - 1) * precision_spec
            except AttributeError:
                precision_spec = np.ones(self.f.__code__.co_argcount - 1) * precision_spec

        # compile all function arguments into a variables tuple
        variables_tuple = tuple([x_0] + list(parameter_list))

        # go through all arguments except the first one
        try:
            no_of_args = self.f.func_code.co_argcount
        except AttributeError:
            no_of_args = self.f.__code__.co_argcount
        for derive_by_index in range(1, no_of_args):
            precision = precision_spec[derive_by_index-1]
            if not precision:
                precision = 1.e-7
            output_list.append(
                derivative(self.f, derive_by_index,
                           variables_tuple, precision)
            )

        return np.asarray(output_list)

    def get_function_equation(self, equation_format='latex',
                              equation_type='full', ensuremath=True):
        r'''
        Returns a string representing the function equation. Supported formats
        are :math:`LaTeX` and ASCII inline math. Note that :math:`LaTeX`
        math is wrapped by default in an ``\ensuremath{}`` expression. If this
        is not desired behaviour, the flag ``ensuremath`` can be set to
        ``False``.

        *equation_format* : string (optional)
            Can be either "latex" (default) or "ascii".

        *equation_type* : string (optional)
            Can be either "full" (default), "short" or "name". A "name"-type
            equation returns a representation of the function name::

                f

            A "short"-type equation limits itself to the function name and
            variables::

                f(x, par1, par2)

            A "full"-type equation includes the expression which the function
            calculates::

                f(x, par1, par2) = par1 * x + par2

        *ensuremath* : boolean (optional)
            If a :math:`LaTeX` math equation is requested, ``True``
            (default) will wrap the resulting expression in an
            ``\ensuremath{}`` tag. Otherwise, no wrapping is done.
        '''

        if equation_format == 'latex':
            tmp_dict = {'name': self.latex_name, 'xname': self.latex_x_name,
                        'paramstring': ",\,".join(self.latex_parameter_names),
                        'expr': self.latex_expression,
                        'hspc': "\\,"}
        elif equation_format == 'ascii':
            tmp_dict = {'name': self.name, 'xname': self.x_name,
                        'paramstring': ", ".join(self.parameter_names),
                        'expr': self.expression,
                        'hspc': " "}
        else:
            raise Exception("Unknown function equation format: "
                            "Expected `latex` or `ascii`, got `%s`"
                            % (equation_format,))

        if equation_type == 'full':
            if tmp_dict['expr'] is None and equation_format == 'ascii':
                tmp_dict['expr'] = "<expression not specified>"
            elif tmp_dict['expr'] is None and equation_format == 'latex':
                tmp_dict['expr'] = "\,?"
            if equation_format == 'latex' and ensuremath:
                return "\\ensuremath{%(name)s(%(xname)s;%(hspc)s"\
                       "%(paramstring)s) = %(expr)s}" % tmp_dict
            else:
                return "%(name)s(%(xname)s;%(hspc)s%(paramstring)s) "\
                       "= %(expr)s" % tmp_dict
        elif equation_type == 'short':
            if equation_format == 'latex' and ensuremath:
                return \
                    "\\ensuremath{%(name)s(%(xname)s;%(hspc)s"\
                    "%(paramstring)s)}" % tmp_dict
            else:
                return "%(name)s(%(xname)s;%(hspc)s%(paramstring)s)" % tmp_dict
        elif equation_type == 'name':
            if equation_format == 'latex' and ensuremath:
                return "\\ensuremath{%(name)s}" % tmp_dict
            else:
                return "%(name)s" % tmp_dict
        else:
            raise Exception("Unknown function equation type: "
                            "Expected `full`, `short` or `name`, got `%s`."
                            % (equation_type,))


def LaTeX(**kwargs):
    r"""
    Optional decorator for fit functions. This overrides a FitFunction's
    `latex_` attributes. The new values for the `latex_` attributes must be
    passed as keyword arguments to the decorator. Possible arguments:

    *name* : string
        :math:`LaTeX` representation of the function name.

    *parameter_names* : list of strings
        List of :math:`LaTeX` representations of the function's arguments.
        The length of this list must be equal to the function's argument
        number. The argument names should be in the same order as in the
        function definition.

    *x_name* : string
        :math:`LaTeX` representation of the independent variable's name.

    *expression* : string
        :math:`LaTeX`-formatted expression representing the
        function's formula.
    """

    # retrieve values from the dictionary
    name = kwargs.pop("name", None)
    parameter_names = kwargs.pop("parameter_names", None)
    x_name = kwargs.pop("x_name", None)
    expression = kwargs.pop("expression", None)

    if kwargs:
        logger.warn("Unknown keyword arguments for decorator LaTeX ignored: %r"
                    % (kwargs.keys(),))

    # override initial LaTeX-related parameters with the ones provided
    def override(fit_function):
        if name is not None:
            fit_function.latex_name = name
        if parameter_names is not None:
            # check if list has the right length
            if len(parameter_names) == fit_function.number_of_parameters:
                fit_function.latex_parameter_names = parameter_names
        if x_name is not None:
            fit_function.latex_x_name = x_name
        if expression is not None:
            fit_function.latex_expression = expression

        return fit_function

    return override


def ASCII(**kwargs):
    r"""
    Optional decorator for fit functions. This overrides a FitFunction's
    plain-text (ASCII) attributes. The new values for these attributes must be
    passed as keyword arguments to the decorator. Possible arguments:

    *name* : string
        Plain-text representation of the function name.

    *parameter_names* : list of strings
        List of plain-text representations of the function's arguments.
        The length of this list must be equal to the function's argument
        number. The argument names should be in the same order as in the
        function definition.

    *x_name* : string
        Plain-text representation of the independent variable's name.

    *expression* : string
        Plain-text-formatted expression representing the
        function's formula.
    """

    # retrieve values from the dictionary
    name = kwargs.pop("name", None)
    parameter_names = kwargs.pop("parameter_names", None)
    x_name = kwargs.pop("x_name", None)
    expression = kwargs.pop("expression", None)

    if kwargs:
        logger.warn("Unknown keyword arguments for decorator ASCII ignored: %r"
                    % (kwargs.keys(),))

    # override initial LaTeX-related parameters with the ones provided
    def override(fit_function):
        if name is not None:
            fit_function.name = name
        if parameter_names is not None:
            # check if list has the right length
            if len(parameter_names) == fit_function.number_of_parameters:
                fit_function.parameter_names = parameter_names
        if x_name is not None:
            fit_function.x_name = x_name
        if expression is not None:
            fit_function.expression = expression

        return fit_function

    return override
