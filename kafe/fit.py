# coding=utf-8
'''
.. module:: fit
    :platform: Unix
    :synopsis: This submodule defines a `Fit` object which performs the actual
        fitting given a `Dataset` and a fit function.
.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>
.. moduleauthor:: Joerg Schindler <joerg.schindler@student.kit.edu>
'''

# ----------------------------------------------------------------
# Changes:
#  06-Aug-14  G.Q.  default parameter errors set to 10% (not 0.1%)
#  07-Aug-14  G.Q.  covariance matrix returned by Minuit contains zero-values
#                   for lines/colums corresponding to fixed parameters; now
#                   calling a special version, MinuitCov_to_cor in
#                   print_fit_results
#             G.Q.  adjusted output formats in print_fit_xxx: `g` format
#                    gives enough digits, '.02e' is sufficient for errors
#             G.Q. output of print_fit_xxx a little more compact;
#                   took account of possibly fixed parameters
#             G.Q. adjusted tolerance on cov mat. changes an # of iterations
# 10-Aug-14   G.Q. initial fits without running HESSE algorithm in Minuit
#                   introduced steering flag "final_fit"
# 11-Aug-14   G.Q. took into accout fixed parameters in calculation of ndf
#                  added mechanism to constrain parameters within error:
#                     - provided function constrain_parameters()
#                     - add corresponding chi2-Term in chi2
# 05-Dec-14   G.Q. added multiple contours (plot_contour) for different
#                   DeltaChi^2 values resp. confidence levels
# 09-Dec-14   G.Q. added profile chi^2
# 12-Dec-14   G.Q. introudced data members of Fit class for final results
#                  replaced self.get_error_matrix() by self.par_cov_mat, the
#                   final covariance matrix
# 18-Dec-14   G.Q. addes "plot_correlations" to provide a graphical
#                  representation of the parameter covariance-matrix
#                  by showing all contours and profiles as an array of plots
# 15-Jan-16   G.Q. fixed color name "darmagenta" -> darkmagenta
# 29-Sep-16   J.S. changed/extended constrain_parameters():
#                  now takes a correlation matrix as a keyword to take
#                  correlations into account. New class GaussianConstraint added
#                  to store all constriants
# 07-Oct-16   G.Q. improved logic to suppress print-out if quiet is specified
#                  in do_fit()
#                  added variable final_fcn to Fit class to store chi2
# 08-Oct-16   G.Q. added function get_results()
# 16-Oct-16   D.S. supplying quiet=True to Fit() is now passed on to minimizer;
#                  no file is created for quiet=True.
# -------------------------------------------------------------------------

from __future__ import print_function
from .function_tools import FitFunction, outer_product
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from .numeric_tools import cov_to_cor, extract_statistical_errors, MinuitCov_to_cor, cor_to_cov

from .config import (FORMAT_ERROR_SIGNIFICANT_PLACES, F_SIGNIFICANCE_LEVEL,
                     M_MINIMIZER_TO_USE, log_file, null_file)
from math import floor, log

import os
from .stream import StreamDup

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')


# The default FCN
def chi2(xdata, ydata, cov_mat,
         fit_function, parameter_values,
         constrain=None):
    r'''
    The :math:`\chi^2` implementation. Calculates :math:`\chi^2` according
    to the formula:

    .. math::

        \chi^2 = \lambda^T C^{-1} \lambda


    Here, :math:`\lambda` is the residual vector :math:`\lambda = \vec{y} -
    \vec{f}(\vec{x})` and :math:`C` is the covariance matrix.

    If a constraint :math:`c_i\pm\sigma_i` is applied to a parameter :math:`p_i`,
    a `penalty term` is added for each constrained parameter:

    .. math::

        \chi^2_{\text{cons}} = \chi^2 + \sum_i{ \left( \frac{p_i - c_i}{\sigma_i} \right)^2 }

    Parameters
    ----------

    **xdata** : iterable
        The *x* measurement data

    **ydata** : iterable
        The *y* measurement data

    **cov_mat** : `numpy.matrix`
        The total covariance matrix

    **fit_function** : function
        The fit function :math:`f(x)`

    **parameter_values** : list/tuple
        The values of the parameters at which :math:`f(x)` should be evaluated.

    Keyword Arguments
    -----------------

    constrain : ``None`` or dictionary , optional
        The key of the dictionary holds the parameter ids,
        while the values are GaussianConstraint objects
        with values, errors and correlation of the parameters.

    '''

    # since the parameter_values are constants, the
    # fit function is a function of only one
    # variable: `x'. To apply it elementwise using
    # Python's `map' method, make a temporary
    # function where `x' is the only variable:
    def tmp_fit_function(x):
        return fit_function(x, *parameter_values)

    # calculate f(x) for all x in xdata
    fdata = np.asarray(list(map(tmp_fit_function, xdata)))
    # calculate residual vector
    residual = ydata - fdata

    chi2val = (residual.T.dot(cov_mat.I).dot(residual))[0, 0]  # return the chi^2

    # apply constraints, if any
    if constrain is not None:
        dchi2 = 0
        for val in constrain.values():
            dchi2 += val.calculate_chi2_penalty(parameter_values)
        chi2val += dchi2
    return chi2val


def round_to_significance(value, error, significance=FORMAT_ERROR_SIGNIFICANT_PLACES):
    '''
    Rounds the error to the established number of significant digits, then
    rounds the value to the same order of magnitude as the error.

    Parameters
    ----------

    **value** : float
        value to round to significance

    **error** : float
        uncertainty of the value

    Keyword Arguments
    -----------------

    significance : int, optional
        number of significant digits of the error to consider

    '''
    # round error to FORMAT_ERROR_SIGNIFICANT_PLACES significant digits
    if error:
        significant_digits = int(-floor(log(error)/log(10))) + significance - 1
        error = round(error, significant_digits)
        value = round(value, significant_digits)

    return (value, error)


class Fit(object):
    '''
    Object representing a fit. This object references the fitted `Dataset`,
    the fit function and the resulting fit parameters.

    Necessary arguments are a `Dataset` object and a fit function (which should
    be fitted to the `Dataset`). Optionally, an external function `FCN` (the
    minimum of which should be located to find the best fit) can be specified.
    If not given, the `FCN` function defaults to :math:`\chi^2`.

    Parameters
    ----------

    **dataset** : `Dataset`
        A `Dataset` object containing all information about the data

    **fit_function** : function
        A user-defined Python function to fit to the data. This
        function's first argument must be the independent variable `x`. All
        other arguments *must* be named and have default values given. These
        defaults are used as a starting point for the actual minimization. For
        example, a simple linear function would be defined like:

        >>> def linear_2par(x, slope=1., y_intercept=0.):
        ...     return slope * x + y_intercept

        Be aware that choosing sensible initial values for the parameters is
        often crucial for a succesful fit, particularly for functions of many
        parameters.

    Keyword Arguments
    -----------------

    external_fcn : function, optional
        An external `FCN` (function to minimize). This function must have the
        following call signature:

        >>> FCN(xdata, ydata, cov_mat, fit_function, parameter_values)

        It should return a float. If not specified, the default :math:`\chi^2`
        `FCN` is used. This should be sufficient for most fits.

    fit_name : string, optional
        An ASCII name for this fit. This is used as a label for the the
        matplotlib figure window and for naming the fit output file. If
        omitted, the fit will take over the name of the parent dataset.

    fit_label : :math:`LaTeX`-formatted string, optional
        A name/label/short description of the fit function. This appears in the
        legend describing the fitter curve. If omitted, this defaults to the
        fit function's :math:`LaTeX` expression.

    minimizer_to_use : 'ROOT' or 'minuit', optional
        Which minimizer to use. This defaults to whatever is set in the config
        file, but can be specifically overridden for some fits using this
        keyword argument
    '''

    def __init__(self, dataset, fit_function, external_fcn=chi2,
                 fit_name=None, fit_label=None,
                 minimizer_to_use=M_MINIMIZER_TO_USE,
                 quiet=False):
        '''
        Construct an instance of a ``Fit``
        '''

        # Initialize instance variables
        ################################

        self.dataset = dataset  #: this Fit instance's child `Dataset`

        # variables to store final results of this fit
        self.final_fcn = None
        """Final minimum of fcn (chi2)"""
        self.final_parameter_values = None
        """Final parameter values"""
        self.final_parameter_errors = None
        """Final parameter errors"""
        self.par_cov_mat = None
        """Parameter covariance matrix (`numpy.matrix`)"""
        self.parabolic_errors=True
        """``True`` if :math:`\chi^2` is approx. parabolic (boolean)"""
        self.minos_errors=None
        """MINOS Errors [err, err+, err-, gcor]"""
        self.contours=[]
        """Parameter Contours [id1, id2, dchi2, [xc], [yc]]"""
        self.profiles=[]
        """Parameter Profiles [id1, [xp], [dchi1(xp)]]"""

        if isinstance(fit_function, FitFunction):
            #: the fit function used for this `Fit`
            self.fit_function = fit_function
        else:
            # if not done alreasy, apply the FitFunction
            # decorator to fit_function ("silent cast")
            self.fit_function = FitFunction(fit_function)

            # and report
            logger.info("Custom fit function not decorated with "
                        "FitFunction decorator; doing a silent cast.")

        #: the (external) function to be minimized for this `Fit`
        self.external_fcn = external_fcn

        #: the total number of parameters
        self.number_of_parameters = self.fit_function.number_of_parameters

        self.set_parameters(self.fit_function.parameter_defaults,
                            None, no_warning=True)

        #: the names of the parameters
        self.parameter_names = self.fit_function.parameter_names
        #: :math:`LaTeX` parameter names
        self.latex_parameter_names = \
            self.fit_function.latex_parameter_names

        # store a dictionary to lookup whether a parameter is fixed
        self._fixed_parameters = np.zeros(self.number_of_parameters,
                                          dtype=bool)
        self.number_of_fixed_parameters = 0

        # Dictionary to store Gaussian_constrain object with the ids of constrained parameters as key
        self.constrain = {}
        self.number_of_constrained_parameters = 0

        # store the full function definition
        self.function_equation_full = \
            self.fit_function.get_function_equation('latex', 'full')

        # store a short version of the function's equation
        self.function_equation = \
            self.fit_function.get_function_equation('latex', 'short')

        self.fit_label = fit_label

        self.fit_name = fit_name

        # check if the dataset has any y errors at all
        if self.dataset.has_errors('y'):
            # set the y cov_mat as starting cov_mat for the fit
            # and report if singular matrix
            self.current_cov_mat = self.dataset.get_cov_mat(
                'y',
                fallback_on_singular='report'
            )
            '''the current covariance matrix used for the `Fit`'''
        else:
            # set the identity matrix as starting cov_mat for the fit
            self.current_cov_mat = np.asmatrix(np.eye(self.dataset.get_size()))
            logger.info("No `y`-errors provided for dataset. Assuming all "
                        "data points have the `y`-error 1.0")

        #: this `Fit`'s minimizer (`Minuit`)
        if type(minimizer_to_use) is str:
            # if specifying the minimizer type using a string
            if minimizer_to_use.lower() == "root":
                # raise error if ROOT is not found on the system
                try:
                    import ROOT
                except ImportError as e:
                    if hasattr(e, 'name') and e.name == "libPyROOT":
                        _msg = "Found PyROOT, but it is not compatible with this version of Python! (%s)" % (e.path,)
                        raise ImportError(_msg)
                    else:
                        raise ImportError("Minimizer 'root' requested, but could "
                                          "not find Python module 'ROOT'.")
                from .minuit import Minuit
                _minimizer_handle = Minuit
            elif minimizer_to_use.lower() == "iminuit":
                from .iminuit_wrapper import IMinuit
                _minimizer_handle = IMinuit
                #raise NotImplementedError, "'iminuit' minimizer not yet implemented"
            else:
                raise ValueError("Unknown minimizer '%s'" % (minimizer_to_use,))
        else:
            # assume class reference is given
            _minimizer_handle = minimizer_to_use

        self.minimizer = _minimizer_handle(self.number_of_parameters,
                                           self.call_external_fcn,
                                           self.parameter_names,
                                           self.current_parameter_values,
                                           None,
                                           # pass quiet flag to minimizer
                                           quiet=quiet)


        # set Minuit's initial parameters and parameter errors
        #            may be overwritten via ``set_parameters``
        self.minimizer.set_parameter_values(self.current_parameter_values)
        self.minimizer.set_parameter_errors(self.current_parameter_errors)  # default 10%, 0.1 if value==0.

        # store measurement data locally in Fit object
        #: the `x` coordinates of the data points used for this `Fit`
        self.xdata = self.dataset.get_data('x')
        #: the `y` coordinates of the data points used for this `Fit`
        self.ydata = self.dataset.get_data('y')

        # Define a stream for storing the output
        if self.dataset.basename is not None:
            _basename = self.dataset.basename
        else:
            _basename = 'untitled'
        if self.fit_name is not None:
            _basename += '_' + fit_name

        if not quiet:
            _basenamelog = log_file(_basename+'.log')
            # check for old logs
            if os.path.exists(_basenamelog):
                logger.info('Old log files found for fit `%s`. kafe will not '
                            'delete these files, but it is recommended to do '
                            'so, in order to reduce clutter.'
                            % (_basename,))

                # find first incremental name for which no file exists
                _id = 1
                while os.path.exists(log_file(_basename+'.'+str(_id)+'.log')):
                    _id += 1

                # move existing log to that location
                os.rename(_basenamelog, log_file('{bn}.{id}.log'.format(
                    bn=_basename, id=_id)))

            self.out_stream = StreamDup([log_file('fit.log'), _basenamelog])
        else:
            # write to NULL file
            # need to wrap in StreamDup due to timestamp function...
            self.out_stream = StreamDup([null_file()])

    def call_external_fcn(self, *parameter_values):
        '''
        Wrapper for the external `FCN`. Since the actual fit process depends on
        finding the right parameter values and keeping everything else constant
        we can use the `Dataset` object to pass known, fixed information to the
        external `FCN`, varying only the parameter values.

        Parameters
        ----------

        **parameter_values** : sequence of values
            the parameter values at which `FCN` is to be evaluated

        '''

        return self.external_fcn(self.xdata, self.ydata, self.current_cov_mat,
                                 self.fit_function, parameter_values,
                                 self.constrain)

    def get_function_error(self, x):
        r'''
        This method uses the parameter error matrix of the fit to calculate
        a symmetric (parabolic) error on the function value itself. Note that
        this method takes the entire parameter error matrix into account, so
        that it also accounts for correlations.

        The method is useful if, e.g., you want to draw a confidence band
        around the function in your plot routine.

        Parameters
        ----------

        **x** : `float` or sequence of `float`
            the values at which the function error is to be estimated

        Returns
        -------

        float or sequence of float
            the estimated error at the given point(s)
        '''

        try:
            iter(x)
        except:
            x = np.array([x])


        errors = np.zeros_like(x)
        # go through each data point and calculate the function error
        for i, fval in enumerate(x):
            # calculate the outer product of the gradient of f with itself
            # (with respect to the parameters)
            # use 1/100th of the smallest parameter error as spacing for df/dp
            derivative_spacing = 0.01 * np.sqrt(
                min(np.diag(self.par_cov_mat))
            )
            par_deriv_outer_prod = outer_product(
                self.fit_function.derive_by_parameters(
                    fval,
                    derivative_spacing,
                    self.current_parameter_values
                )
            )

            tmp_sum = np.sum(
                par_deriv_outer_prod * np.asarray(
                    self.par_cov_mat
                )
            )
            errors[i] = np.sqrt(tmp_sum)

        return errors

    def get_current_fit_function(self):
        '''
        This method returns a function object corresponding to the fit function
        for the current parameter values. The returned function is a function
        of a single variable.

        Returns
        -------

        function handle
            A function of a single variable corresponding to the fit function
            at the current parameter values.
        '''

        def current_fit_function(x):
            return self.fit_function(x, *self.current_parameter_values)

        return current_fit_function

    def get_error_matrix(self):
        '''
        This method returns the covariance matrix of the fit parameters which
        is obtained by querying the minimizer object for this `Fit`

        Returns
        -------

        *numpy.matrix*
            The covariance matrix of the parameters.
        '''
        return self.minimizer.get_error_matrix()

    def get_parameter_errors(self, rounding=False):
        '''
        Get the current parameter uncertainties from the minimizer.

        Keyword Arguments
        -----------------

        rounding : boolean, optional
            Whether or not to round the returned values to significance.

        Returns
        -------
        tuple
            A tuple of the parameter uncertainties
        '''
        output = []
        names = []
        for name, value, error in self.minimizer.get_parameter_info():
            names.append(name)
            if rounding:
                value, error = round_to_significance(value, error)
            output.append(error)

        # make sure parameters are in the defined order
        _ordering = list(map(self.parameter_names.index, names))
        _order = dict(zip(output, _ordering))
        output.sort(key=_order.get)

        return tuple(output)

    def get_parameter_values(self, rounding=False):
        '''
        Get the current parameter values from the minimizer.

        Keyword Arguments
        -----------------

        rounding : boolean, optional
            Whether or not to round the returned values to significance.

        Returns
        -------

        tuple
            A tuple of the parameter values
        '''

        output = []
        names = []
        for name, value, error in self.minimizer.get_parameter_info():
            names.append(name)
            if rounding:
                value, error = round_to_significance(value, error)
            output.append(value)

        # make sure parameters are in the defined order
        _ordering = list(map(self.parameter_names.index, names))
        _order = dict(zip(output, _ordering))
        output.sort(key=_order.get)

        return tuple(output)


    def set_parameters(self, *args, **kwargs):
        '''
        Sets the parameter values (and optionally errors) for this fit.
        This is usually called just before the fit is done, to establish
        the initial parameters. If a parameter error is omitted, it is
        set to 1/10th of the parameter values themselves. If the default
        value of the parameter is 0, it is set, by exception, to 0.1.

        This method accepts up to two positional arguments and several
        keyword arguments.

        Parameters
        ----------

        *args[0]* : tuple/list of floats, optional
            The first positional argument is expected to be
            a tuple/list containing the parameter values.

        *args[1]* : tuple/list of floats, optional
            The second positional argument is expected to be a
            tuple/list of parameter errors, which can also be set as an
            approximate estimate of the problem's uncertainty.

        Keyword Arguments
        -----------------

        no_warning : boolean, optional
            Whether to issue warnings (``False``) or not (``True``) when
            communicating with the minimizer fails. Defaults to ``False``.

        Valid keyword argument names are parameter names. The keyword arguments
        themselves may be floats (parameter values) or 2-tuples containing the
        parameter values and the parameter error in that order:

        *<parameter_name>* : float or 2-tuple of floats, optional
            Set the parameter with the name <'parameter_name'> to the value
            given. If a 2-tuple is given, the first element is understood
            to be the value and the second to be the parameter error.
        '''

        # Process arguments

        # get global keyword argument
        no_warning = kwargs.pop("no_warning", False)

        if args:  # if positional arguents provided
            if len(args) == 1:
                par_values, par_errors = args[0], None
            elif len(args) == 2:
                par_values, par_errors = args[0], args[1]
            else:
                raise Exception("Error setting parameters. The argument "
                                "pattern for method `set_parameters` could "
                                "not be parsed.")

            if len(par_values) == self.number_of_parameters:
                #: the current values of the parameters
                self.current_parameter_values = list(par_values)
            else:
                raise Exception("Cannot set parameters. Number of given "
                                "parameters (%d) doesn't match the Fit's "
                                "parameter number (%d)."
                                % (len(par_values), self.number_of_parameters))

            if par_errors is not None:
                if len(par_values) == self.number_of_parameters:
                    #: the current uncertainties of the parameters
                    self.current_parameter_errors = list(par_errors)
                else:
                    raise Exception("Cannot set parameter errors. Number of "
                                    "given parameter errors (%d) doesn't "
                                    "match the Fit's parameter number (%d)."
                                    % (len(par_errors),
                                       self.number_of_parameters))
            else:
                if not no_warning:
                    logger.warn("Parameter starting errors not given. Setting "
                                "to 1/10th of the parameter values.")
                #: the current uncertainties of the parameters
                self.current_parameter_errors = [
                    val/10. if val else 0.1  # handle the case val = 0
                    for val in self.current_parameter_values
                ]
        else:  # if no positional arguments, rely on keywords

            for param_name, param_spec in kwargs.items():
                par_id = self._find_parameter(param_name)
                if par_id is None:
                    raise ValueError("Cannot set parameter. `%s` not "
                                     "a valid ID or parameter name."
                                     % param_name)
                try:
                    # try to read both parameter value and error
                    param_val, param_err = param_spec
                except TypeError:
                    # if param_spec is not iterable, then only value
                    # was given
                    if not no_warning:
                        logger.warn("Parameter error not given for %s. "
                                    "Setting to 1/10th of the parameter "
                                    "value given." % (param_name,))
                    param_val, param_err = param_spec, param_spec * 0.1

                self.current_parameter_values[par_id] = param_val
                self.current_parameter_errors[par_id] = param_err

        # try to update the minimizer's parameters
        # (fails if minimizer not yet initialized)
        try:
            # set Minuit's start parameters and parameter errors
            self.minimizer.set_parameter_values(
                self.current_parameter_values)
            self.minimizer.set_parameter_errors(
                self.current_parameter_errors)
        except AttributeError:
            if not no_warning:
                logger.warn("Failed to set the minimizer's parameters. "
                            "Maybe minimizer not initialized for this Fit "
                            "yet?")

    def fix_parameters(self, *parameters_to_fix):
        '''
        Fix the given parameters so that the minimizer works without them
        when :py:meth:`~kafe.fit.Fit.do_fit` is called next. Parameters can be
        given by their names or by their IDs.
        '''
        for parameter in parameters_to_fix:
            # turn names into IDs, if needed
            par_id = self._find_parameter(parameter)
            if par_id is None:
                raise ValueError("Cannot fix parameter. `%s` not "
                                 "a valid ID or parameter name."
                                 % parameter)
            # found parameter, fix it
            self.minimizer.fix_parameter(par_id)
            self.number_of_fixed_parameters += 1
            self._fixed_parameters[par_id] = True
            logger.info("Fixed parameter %d (%s)"
                        % (par_id, self.parameter_names[par_id]))

    def release_parameters(self, *parameters_to_release):
        '''
        Release the given parameters so that the minimizer begins to work with
        them when :py:func:`do_fit` is called next. Parameters can be given by
        their
        names or by their IDs. If no arguments are provied, then release all
        parameters.
        '''
        if parameters_to_release:
            for parameter in parameters_to_release:
                # turn names into IDs, if needed
                par_id = self._find_parameter(parameter)
                if par_id is None:
                    raise ValueError("Cannot release parameter. `%s` not "
                                     "a valid ID or parameter name."
                                     % parameter )

                # Release found parameter
                self.minimizer.release_parameter(par_id)
                self.number_of_fixed_parameters -= 1
                self._fixed_parameters[par_id] = False
                logger.info("Released parameter %d (%s)"
                            % (par_id, self.parameter_names[par_id]))
        else:
            # go through all parameter IDs
            for par_id in range(self.number_of_parameters):
                # Release parameter
                self.minimizer.release_parameter(par_id)

            # Inform about release
            logger.info("Released all parameters")

    def constrain_parameters(self, parameters, parvals, parerrs, cor_mat=None):
        r'''
        Constrain the parameter with the given name to :math:`c\pm\sigma`.
        This is achieved by adding an appropriate `penalty term` to the
        :math:`\chi^2` function, see function :py:func:`~kafe.fit.chi2`.

        Parameters
        ----------

        **parameters**: list of int
            list of paramter id's or names to constrain

        **parvals**: list of float
            list of parameter values

        **parerrs**: list of float
            list of errors on parameters

        Keyword Arguments
        -----------------

        **cor_mat** : `numpy.matrix` optional
            correlation matrix of the parameters

        '''
        # Create temporary lists to hold our constrain data
        dummy = []
        parameter_constrain = [
            np.zeros(self.number_of_parameters, dtype=np.float32),
            np.zeros(self.number_of_parameters, dtype=np.float32)]

        # turn name(s) into Id(s), if needed
        for i, parameter in enumerate(parameters):
            par_id = self._find_parameter(parameter)
            if par_id is None:
                raise ValueError("Cannot constrain parameter. `%s` not "
                                 "a valid ID or parameter name."
                                 % parameter)
            elif [id for id in self.constrain.keys() if par_id in id]:
                raise ValueError("Cannot constrain parameter. '%s' is already "
                                 "a constrained parameter."
                                 % parameter)
            else:
                # Create dummy list with all ids
                dummy.append(par_id)
                # Create parameter_constrain, which holds the value and error
                # of the constrained parameters. The list is sorted, so that the
                # place of the parameter is the id
                parameter_constrain[0][par_id] = list(parvals)[i]
                parameter_constrain[1][par_id] = list(parerrs)[i]
                self.number_of_constrained_parameters += 1

            logger.info("Constrained parameter %d (%s)"
                        % (par_id, self.parameter_names[par_id]))
        cov_mat = None
        # Check if correlations are given
        if cor_mat is not None:
            # Convert correlation matrix to covariance matrix for easier computing later on
            cov_mat = cor_to_cov(cor_mat, parerrs)

        # Sort the tupel for better readability
        dummy.sort()
        # Create dictionary entry
        self.constrain.update({tuple(dummy): GaussianConstraint(parameter_constrain, cov_mat)})

    def parameter_is_fixed(self, parameter):
        '''
        Check whether a parameter is fixed. Accepts a parameter's name or ID
        and returns a boolean value.
        '''
        _idx = self._find_parameter(parameter)
        if _idx is not None:
            if self._fixed_parameters[_idx]:
                return True
            else:
                return False
        else:
            raise ValueError("Cannot check if parameter is fixed. `%s` not "
                             "a valid ID or parameter name." % parameter)

    # Private Methods
    ##################

    def _find_parameter(self, lookup_string):
        '''
        Accepts a parameter's name or ID and returns its ID. If not found,
        returns ``None``.
        '''
        # Try to find the parameter by its ID
        try:
            self.current_parameter_values[lookup_string]
        except TypeError:
            # ID invalid. Try to lookup by name.
                try:
                    found_id = self.parameter_names.index(lookup_string)
                except ValueError:
                    return None
        else:
            found_id = lookup_string

        return found_id

    def get_results(self):
        '''
        Return results from Fit
        '''
        return (self.final_parameter_values,
                self.final_parameter_errors,
                self.par_cov_mat,
                self.final_fcn)

    # Fit Workflow
    ###############

    def do_fit(self, quiet=False, verbose=False):
        '''
        Runs the fit algorithm for this `Fit` object.

        First, the :py:obj:`Dataset` is fitted considering only uncertainties
        in the
        `y` direction. If the `Dataset` has no uncertainties in the `y`
        direction, they are assumed to be equal to 1.0 for this preliminary
        fit, as there is no better information available.

        Next, the fit errors in the `x` direction (if they exist) are taken
        into account by projecting the covariance matrix for the `x` errors
        onto the `y` covariance matrix. This is done by taking the first
        derivative of the fit function in each point and "projecting" the `x`
        error onto the resulting tangent to the curve.

        This last step is repeated until the change in the error matrix caused
        by the projection becomes negligible.

        Keyword Arguments
        -----------------

        quiet : boolean, optional
            Set to ``True`` if no output should be printed.

        verbose : boolean, optional
            Set to ``True`` if more output should be printed.
        '''

        # insert timestamp
        self.out_stream.write_timestamp('Fit performed on')

        if not quiet:
            print("###########", file=self.out_stream,)
            print("# Dataset #", file=self.out_stream,)
            print("###########", file=self.out_stream,)
            print('', file=self.out_stream,)
            print(self.dataset.get_formatted(), file=self.out_stream,)

            print("################", file=self.out_stream,)
            print("# Fit function #", file=self.out_stream,)
            print("################", file=self.out_stream,)
            print("", file=self.out_stream,)
            print(self.fit_function.get_function_equation('ascii', 'full',),
                  file=self.out_stream,)
            print("", file=self.out_stream)

            if self.constrain:
                print("###############", file=self.out_stream,)
                print("# Constraints #", file=self.out_stream,)
                print("###############", file=self.out_stream,)
                print(''               , file=self.out_stream,)
                for i in self.constrain.values():
                    for j, err in enumerate(i.parameter_constrain[1]):
                        if(err):
                            print("%s: %g +\- %g" % (
                                self.parameter_names[j],
                                i.parameter_constrain[0][j],
                                err), file=self.out_stream,)
                    if i.cov_mat_inv is not None:
                        print("Correlation Matrix: ",              file=self.out_stream,)
                        print(format(cov_to_cor(i.cov_mat_inv.I)), file=self.out_stream,)
                print("", file=self.out_stream)

        max_x_iterations = 10

        logger.debug("Calling Minuit")
        if self.dataset.has_errors('x'):
            self.call_minimizer(final_fit=False, verbose=verbose, quiet=quiet)
        else:
            self.call_minimizer(final_fit=True, verbose=verbose, quiet=quiet)

        # if the dataset has x errors, project onto the current error matrix
        if self.dataset.has_errors('x'):
            logger.debug("Dataset has `x` errors. Iterating for `x` error.")
            iter_nr = 0
            while iter_nr < max_x_iterations:
                old_matrix = copy(self.current_cov_mat)
                self.project_x_covariance_matrix()

                logger.debug("`x` fit iteration %d" % (iter_nr,))
                if iter_nr==0:
                    self.call_minimizer(final_fit=False, verbose=verbose, quiet=quiet)
                else:
                    self.call_minimizer(final_fit=True, verbose=verbose, quiet=quiet)
                new_matrix = self.current_cov_mat

                # stop if the matrix has not changed within tolerance)
                # GQ: adjusted precision: rtol 1e-4 on cov-matrix is
                # clearly sufficient
                if np.allclose(old_matrix, new_matrix, atol=0, rtol=1e-4):
                    logger.debug("Matrix for `x` fit iteration has converged.")
                    break   # interrupt iteration
                iter_nr += 1

        # determine, retrieve and analyze errors from MINOS algorithm
        tol = 0.05
        if(quiet):
          log_level=-1
        else:
          log_level=1
        self.minos_errors = self.minimizer.minos_errors(log_level)
        # error analysis:
        for par_nr, par_val in enumerate(self.current_parameter_values):
            ep = self.minos_errors[par_nr][0]
            em = self.minos_errors[par_nr][1]
            if ep != 0 and em != 0:
              if (abs(ep + em)/(ep - em) > tol) or \
                 (abs(1. - 0.5*(ep - em)/self.minos_errors[par_nr][2])>tol):
                  self.parabolic_errors=False

        # store results ...
        self.final_fcn = self.minimizer.get_fit_info('fcn')
        self.final_parameter_values = self.current_parameter_values
        self.final_parameter_errors = self.current_parameter_errors
        self.par_cov_mat = self.get_error_matrix()
        # ... and print at end of fit
        if not quiet:
            self.print_fit_results()
            self.print_rounded_fit_parameters()
            self.print_fit_details()

    def print_raw_results(self):
        '''
        unformatted print-out of all fit results in
        '''
        print('\n')
        print('par values' + str(self.final_parameter_values))
        print('par errors' + str(self.final_parameter_errors))
        print('par. covariance matrix:')
        print(self.par_cov_mat)
        print('MINOS errors' + str( self.minos_errors))
        print('contours:')
        print(self.contours)
        print('profiles:')
        print(self.profiles)


    def call_minimizer(self, final_fit=True, verbose=False, quiet=False):
        '''
        Instructs the minimizer to do a minimization.
        '''

        verbosity = 0
        if(final_fit):
            verbosity = 2
        if (verbose):
            verbosity = 3
        if(quiet): verbosity = -1

        logger.debug("Calling minimizer")
        self.minimizer.minimize(
            final_fit=final_fit,log_print_level=verbosity)
        logger.debug("Retrieving data from minimizer")
        self.current_parameter_values = self.minimizer.get_parameter_values()
        self.current_parameter_errors = self.minimizer.get_parameter_errors()

    def project_x_covariance_matrix(self):
        r'''
        Project elements of the `x` covariance matrix onto the total
        matrix.

        This is done element-wise, according to the formula:

        .. math ::

            C_{\text{tot}, ij} = C_{y, ij} + C_{x, ij}
            \frac{\partial f}{\partial x_i}  \frac{\partial f}{\partial x_j}
        '''

        # Log projection (DEBUG)
        logger.debug("Projecting `x` covariance matrix.")

        # use 1/100th of the smallest error as spacing for df/dx
        precision_list = 0.01 * np.sqrt(np.diag(self.current_cov_mat) )

        if min(precision_list)==0:
            logger.warn('At least one input error is zero - set to 1e-7')
            for i, p in enumerate(precision_list):
                if not p:
                    precision_list[i] = 1.e-7

        outer_prod = outer_product(
            self.fit_function.derive_by_x(self.dataset.get_data('x'),
                                          precision_list,
                                          self.current_parameter_values)
        )

        proj_xcov_mat = np.asarray(self.dataset.get_cov_mat('x')) * outer_prod

        self.current_cov_mat = self.dataset.get_cov_mat('y') + \
            np.asmatrix(proj_xcov_mat)

    # Output functions
    ###################

    def print_rounded_fit_parameters(self):
        '''prints the fit parameters'''

        print("########################", file=self.out_stream,)
        print("# Final fit parameters #", file=self.out_stream,)
        print("########################", file=self.out_stream,)
        print(''                        , file=self.out_stream,)

        for name, value, error in self.minimizer.get_parameter_info():

            tmp_rounded = round_to_significance(value, error, FORMAT_ERROR_SIGNIFICANT_PLACES)
            if error:
                print("%s = %g +- %g" % (
                    name, tmp_rounded[0], tmp_rounded[1]), file=self.out_stream, )
            else:
                print("%s = %g    -fixed-" % (
                    name, tmp_rounded[0]), file=self.out_stream, )

        print("", file=self.out_stream,)

    def print_fit_details(self):
        '''prints some fit goodness details'''


        _ndf = (self.dataset.get_size() - self.number_of_parameters
               + self.number_of_fixed_parameters
               + self.number_of_constrained_parameters)


        chi2prob = self.minimizer.get_chi2_probability(_ndf)
        if chi2prob < F_SIGNIFICANCE_LEVEL:
            hypothesis_status = 'rejected (sig. %d%s)' \
                % (int(F_SIGNIFICANCE_LEVEL*100), '%')
        else:
            hypothesis_status = 'accepted (sig. %d%s)' \
                % (int(F_SIGNIFICANCE_LEVEL*100), '%')

        print('###############', file=self.out_stream, )
        print("# Fit details #", file=self.out_stream, )
        print("###############", file=self.out_stream, )
        print(''               , file=self.out_stream, )

        # Print a warning if NDF is zero
        if not _ndf:
            print( "# WARNING: Number of degrees of freedom is zero!", file=self.out_stream, )
            print("# Please review parameterization...", file=self.out_stream)
            print("", file=self.out_stream)
        elif _ndf < 0:
            print("# WARNING: Number of degrees of freedom is negative!", file=self.out_stream)
            print("# Please review parameterization...", file=self.out_stream)
            print("", file=self.out_stream)
        if(not self.parabolic_errors):
            print('Attention: use uncertainties from MINOS', file=self.out_stream)
            print('', file=self.out_stream)

        print('USING    %s' %(self.minimizer.name), file=self.out_stream)
        print('FCN/ndf  %.3g/%d = %.3g'
              % (self.minimizer.get_fit_info('fcn'), _ndf,
              self.minimizer.get_fit_info('fcn')/(_ndf)), file=self.out_stream)
        print('EdM      %g'
            %(self.minimizer.get_fit_info('edm')), file=self.out_stream)
        print('UP       %g'
            %(self.minimizer.get_fit_info('err_def')), file=self.out_stream)
        print('STA      ' + str(self.minimizer.get_fit_info('status_code')) , file=self.out_stream)
        print('', file=self.out_stream)
        print('chi2prob', round(chi2prob, 3), file=self.out_stream)
        print('HYPTEST  ' + str(hypothesis_status), file=self.out_stream)
        print('', file=self.out_stream)


    def print_fit_results(self):
        '''prints fit results'''

        print('##############', file=self.out_stream)
        print('# Fit result #', file=self.out_stream)
        print('##############', file=self.out_stream)
        print('', file=self.out_stream)

        par_err = extract_statistical_errors(self.par_cov_mat)
        par_cor_mat = MinuitCov_to_cor(self.par_cov_mat)

        print('# value        error   ', file=self.out_stream, end="")
        if self.number_of_parameters > 1:
            print('correlations', file=self.out_stream)
        else:
            print('', file=self.out_stream)
        for par_nr, par_val in enumerate(self.final_parameter_values):
            print('# '+self.parameter_names[par_nr], file=self.out_stream)
            print('{:.04e}  '.format(par_val), file=self.out_stream, end="")
            if par_err[par_nr]:
              print(format(par_err[par_nr], '.02e')+'  ', end="", file=self.out_stream)
            else:
              print('-fixed- ', end="", file=self.out_stream)
            if par_nr > 0 and par_err[par_nr]:
                for i in range(par_nr):
                    print('{:+.3f}  '.format(par_cor_mat[par_nr, i]), end="", file=self.out_stream)
            print('', file=self.out_stream)
      # print MINOS errors if needed
        if(not self.parabolic_errors):
            print( '!!! uncertainties from MINOS:', file=self.out_stream)
            for par_nr, par_val in enumerate(self.final_parameter_values):
                print( '# '+self.parameter_names[par_nr], file=self.out_stream)
                if par_err[par_nr]:
                    print('      {:.02e} + {:.02e}'.format(
                        self.minos_errors[par_nr][0], self.minos_errors[par_nr][1]), file=self.out_stream)
                else:
                    print('-fixed- ',end="", file=self.out_stream)
            print('', file=self.out_stream)
        print('', file=self.out_stream)

    def plot_contour(self, parameter1, parameter2, dchi2=2.3,
                     n_points=100, color='gray', alpha=.1, show=False,
                     axes=None):
        r'''
        Plots one or more two-dimensional contours for this fit into
        a separate figure and returns the figure object.

        Parameters
        ----------

        **parameter1** : int or string
            ID or name of the parameter to appear on the `x`-axis.

        **parameter2** : int or string
            ID or name of the parameter to appear on the `y`-axis.

        Keyword Arguments
        -----------------

        dchi2 : float or list of floats (otpional)
            delta-chi^2 value(s) used to evaluate contour(s)
            1. = 1 sigma
            2.3 = 68.0% (default)
            4.  = 2 sigma
            5.99 = 95.0%

        n_points : int, optional
            Number of plot points to use for the contour. Higher
            values yield smoother contours but take longer to
            render. Default is 100.

        color : string, optional
            A ``matplotlib`` color identifier specifying the fill color
            of the contour. Default is 'gray'.

        alpha : float, optional
            Transparency of the contour fill color ranging from 0. (fully
            transparent) to 1. (fully opaque). Default is 0.25

        show : boolean, optional
            Specify whether to show the figure before returning it. Defaults
            to ``False``.

        axes : `maplotlib.pyplot.axes`
            Sub-plot axes to add plot to

        Returns
        -------

        ``matplotlib`` figure object if no axes given
            A figure object containing the contour plot.
        '''

        # lookup parameter IDs
        par1 = self._find_parameter(parameter1)
        par2 = self._find_parameter(parameter2)
        _pvals = self.final_parameter_values
        _perrs = self.final_parameter_errors
        xval, yval = _pvals[par1], _pvals[par2]
        xer, yer = _perrs[par1], _perrs[par2]

        plt.tight_layout()
        if axes is None:
            #new (square) figure for contour(s)
            tmp_fig = plt.figure(figsize=(5., 5.))
            # get/create axes object for current figure
            tmp_ax = tmp_fig.gca()

        else:
            tmp_ax = axes
        # set axis labels
        tmp_ax.set_xlabel('$%s$' % (self.latex_parameter_names[par1],),
                          fontsize='xx-large')
        tmp_ax.set_ylabel('$%s$' % (self.latex_parameter_names[par2],),
                          fontsize='xx-large')
        # set size and number of tick labels
        tmp_ax.tick_params(axis='both', which='both',
                           labelsize='large')
        tmp_ax.ticklabel_format(axis='both', style='scientific',
                                scilimits=(-3, 4), useOffset=False)
        tmp_ax.locator_params(nbins=5)
        # plot central value and errors
        tmp_ax.errorbar(xval, yval, xerr=xer, yerr=yer, fmt='o')
        # tmp_ax.scatter(xval, yval, marker='+', label='parameter values')
        tmp_ax.set_prop_cycle("color",
                              ['black', 'darkblue', 'darkgreen', 'chocolate',
                                'darkmagenta', 'darkred', 'darkorange',
                                'darkgoldenrod'])
        # plot contours(s)
        dc2list = []
        try:
            iter_over_dchi2 = iter(dchi2)
        except:
            dc2list.append(dchi2)  # not iterable, append float
        else:
            dc2list.extend(dchi2)  # iterable, extend by list
        ncont = 0
        for dc2 in dc2list:
            ncont += ncont  # count contours in list
            self.minimizer.set_err(dc2)
            xs, ys = self.minimizer.get_contour(par1, par2, n_points)
            # store result
            self.contours.append([par1, par2, dc2, xs, ys])
            # plot contour lines
            cl=100*Chi22CL(dc2) # get corresponding confidence level
            print('Contour %.1f %%CL for parameters %d vs. %d with %d points'
                % (cl, par1, par2, len(xs)), file=self.out_stream)
            labelstr = "%.1f"%(cl) + r"\% CL"
            tmp_ax.fill(xs, ys, alpha=alpha, color=color)   # as filled area
            tmp_ax.plot(xs, ys, '--', linewidth=2, label=labelstr)  # as line
        print("", file=self.out_stream)
        self.minimizer.set_err(1.)  # set errdef back to default of 1.
        # plot a legend
        tmp_leg = tmp_ax.legend(loc='best', fontsize='small')
        # show the contour, if requested
        if axes is None:
            if show:
                tmp_fig.show()
            return tmp_fig

    def plot_profile(self, parid, n_points=21,
                     color='blue', alpha=.5, show=False, axes=None):
        r'''
        Plots a profile :math:`\\chi^2` for this fit into
        a separate figure and returns the figure object.

        Parameters
        ----------

        **parid** : int or string
            ID or name of parameter

        Keyword Arguments
        -----------------

        n_points : int, optional
           Number of plot points to use for the profile curve.

        color : string, optional
           A ``matplotlib`` color identifier specifying the line
           color. Default is 'blue'.

        alpha : float, optional
           Transparency of the contour fill color ranging from 0. (fully
           transparent) to 1. (fully opaque). Default is 0.25

        show : boolean, optional
           Specify whether to show the figure before returning it. Defaults
           to ``False``.

        axes : sub-plot axes to put plot

        Returns
        -------

        ``matplotlib`` figure object if axes is None
            A figure object containing the profile plot.
        '''

        from scipy import interpolate

        # lookup parameter ID
        id = self._find_parameter(parid)
        _pvals = self.final_parameter_values
        _perrs = self.final_parameter_errors
        val = _pvals[id]
        err = _perrs[id]

        print('Profile for parameter %d with %d points'
            % (id, n_points), file=self.out_stream)

        plt.tight_layout()
        if axes is None:
            # new (square) figure for contour(s)
            tmp_fig = plt.figure(figsize=(5., 5.))
            # get/create axes object for current figure
            tmp_ax = tmp_fig.gca()
        else:
            tmp_ax = axes

        # set axis labels
        tmp_ax.set_xlabel('$%s$' % (self.latex_parameter_names[id],),
                          fontsize='xx-large')
        tmp_ax.set_ylabel('$%s$' % ('\\Delta \\chi^2'),
                          fontsize='xx-large')
        # set size and number of tick labels
        tmp_ax.tick_params(axis='both', which='both',
                           labelsize='large')
        tmp_ax.ticklabel_format(axis='both', style='scientific',
                                scilimits=(-3, 4), useOffset=False)
        tmp_ax.locator_params(nbins=5)
        tmp_ax.set_ylim(0., 9.)
        tmp_ax.axvline(x=val, linestyle='--', linewidth=1, color='black')
        tmp_ax.axhline(y=1., linestyle='--', linewidth=1, color='darkred')
        tmp_ax.axhline(y=4., linestyle='-.', linewidth=1, color='darkred')
        # plot central value and errors
        tmp_ax.errorbar(val, 1., xerr=err, linewidth=3, fmt='o', color='black')
        # tmp_ax.scatter(xval, yval, marker='+', label='parameter values')
        # get profile
        xp, yp = self.minimizer.get_profile(id, n_points)
        self.profiles.append([id, xp, yp])  # store this result
        # plot (smoothed) profile
        yp = yp - np.min(yp)  # refer to minimum
        yspline = interpolate.UnivariateSpline(xp, yp, s=0)
        xnew = np.linspace(xp[0], xp[n_points - 1], 200)
        tmp_ax.plot(xnew, yspline(xnew), '-', linewidth=2, color=color,
                    label='profile $\\chi^2$')
        # plot parabolic expectation
        parabolicChi2 = (xnew - val) * (xnew - val) / (err * err)
        tmp_ax.plot(xnew, parabolicChi2, '-.', linewidth=1, color='green',
                    label='parabolic $\\chi^2$')

        tmp_leg = tmp_ax.legend(loc='best', fontsize='small')
        # show the plot, if requested
        if axes is None:
            if show:
                tmp_fig.show()
            return tmp_fig

    def plot_correlations(self):
        '''
        Plots two-dimensional contours for all pairs of parameters
        and profile for all parameters, arranges as a matrix.

        Returns
        -------

        ``matplotlib`` figure object
            A figure object containing the matrix of plots.
        '''
        npar = self.number_of_parameters - self.number_of_fixed_parameters
        cor_fig, axarr = plt.subplots(
            npar, npar, figsize=(5. * npar, 5. * npar))

        ip = -1
        for i in range(0, self.number_of_parameters):
            if not self._fixed_parameters[i]:
                ip += 1
            jp = -1
            for j in range(0, self.number_of_parameters):
                if not self._fixed_parameters[j]:
                    jp += 1
                # skip fixed parameters
                if not(self._fixed_parameters[i] or self._fixed_parameters[j]):
                    if ip > jp:
                     # empty space
                        axarr[jp, ip].axis('off')
                    elif ip == jp:
                        # plot profile
                        self.plot_profile(i, axes=axarr[ip, ip])
                    else:
                        # plot contour
                        self.plot_contour(
                            i, j, dchi2=[1., 2.3], axes=axarr[jp, ip])

        return cor_fig


class GaussianConstraint(object):
    '''
    Object used to constrain parameters. The object stores for each constrain
    the constrained parameters, the errors, the id of the parameter (the place
    at which each parameter is located in parameter_constrain) and the inverse
    covariance matrix of the constrained parameters.
    The class gives a tool to calculate the chi2 penalty term for the given
    constrained parameters, where the fitted parameter_values must be given.

    Parameters
    ----------

    constraint: list of two iterables
        The first iterable (:math:`{c_i}`) contains the constrained parameters'
        expected values and the second iterable (:math:`{\sigma_i}`) contains
        the constraint uncertainties. A parameter with constraint uncertainty
        set to 0 remains unconstrained.

    Keyword Arguments
    -----------------

    cov_mat: 'numpy matrix'
        Contains the covariance matrix of the constrains. The inverse covariance
        matrix will be saved to safe computing time.

    '''
    def __init__(self, constraint, cov_mat=None):

        # List of constrain values and errors (param_constrain[1]=errors, param_constrain[0]=values)
        self.parameter_constrain = constraint
        # Inverse covariance matrix
        if cov_mat is not None:
            self.cov_mat_inv = cov_mat.I
        else:
            self.cov_mat_inv = None

    def calculate_chi2_penalty(self, parameter_values):
        '''
        Calculates the :math:`\chi^2` penalty for the given constraint
        parameter. This function gets called in the :math:`\chi^2` function
        and returns a penalty term.

        Parameters
        ----------

        parameter_values: list/tuple
            The values of the parameters at which :math:`f(x)` should be evaluated.

        '''
        _vector = []
        dchi2 = 0
        if self.parameter_constrain is not None:
            if self.cov_mat_inv is not None:
                for i, err in enumerate(self.parameter_constrain[1]):
                    if err:  # there is a constraint, add to chi2
                        _vector.append(parameter_values[i] - self.parameter_constrain[0][i])
                _vector = np.asarray(_vector)
                dchi2 = (_vector.T.dot(self.cov_mat_inv).dot(_vector))[0, 0]

            else:
                for i, err in enumerate(self.parameter_constrain[1]):
                    if err:  # there is a constraint, add to chi2
                        dchi2 += ((parameter_values[i] - self.parameter_constrain[0][i]) / err) ** 2

        return dchi2






def CL2Chi2(CL):
    '''
    Helper function to calculate DeltaChi2 from confidence level CL
    '''
    return -2.*np.log(1.-CL)

def Chi22CL(dc2):
    '''
    Helper function to calculate confidence level CL from DeltaChi2
    '''
    return (1. - np.exp(-dc2 / 2.))

def build_fit(dataset, fit_function,
              fit_label='untitled', fit_name=None, initial_fit_parameters=None,
              constrained_parameters=None):
    '''
    This helper fuction creates a :py:class:`~kafe.fit.Fit` from a series of
    keyword arguments.

    Parameters
    ----------

    **dataset** : a *kafe* :py:class:`~kafe.dataset.Dataset`

    **fit_function** : a Python function, optionally with
        ``@FitFunction``, ``@LATEX`` and ``@FitFunction`` decorators

    Keyword Arguments
    -----------------

    fit_label : LaTeX label for this fit, optional
       Defaults to "untitled"

    fit_name : name for this fit, optional
       Defaults to the dataset name

    initial_fit_parameters : ``None`` or 2-tuple of list, sequence of floats
       specifying initial parameter values and errors

    constrained_parameters : ``None`` or 3-tuple of list, tuple/`np.array`
       of one string and 2 floats specifiying the names, values and
       uncertainties of constraints to apply to model parameters

    Returns
    -------

    ::py:class:`~kafe.fit.Fit` object

    '''

    # create a ``Fit`` object
    theFit = Fit(dataset, fit_function, fit_label=fit_label, fit_name=fit_name)
    # set initial parameter values and range
    if initial_fit_parameters is not None:
        theFit.set_parameters(initial_fit_parameters[0],      # values
                              initial_fit_parameters[1])      # ranges
    # set parameter constraints
    if constrained_parameters is not None:
        theFit.constrain_parameters(constrained_parameters[0],  # names
                                    constrained_parameters[1],  # val's
                                    constrained_parameters[2])  # uncert's

    return theFit
