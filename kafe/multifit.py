# coding=utf-8
'''
.. module:: fit
    :platform: Unix
    :synopsis: This submodule defines a `Multiit` object which performs a multifit
         given a number of models each consisting of a `Dataset` and a fit function.
.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>
.. moduleauthor:: Joerg Schindler <joerg.schindler@student.kit.edu>
'''

# ----------------------------------------------------------------
# Changes:
# 11-Oct-16   J.S.: Added class ParameterSpace.
# ----------------------------------------------------------------

from __future__ import print_function

import kafe
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from copy import copy

from .function_tools import outer_product
from .numeric_tools import extract_statistical_errors, MinuitCov_to_cor, cor_to_cov
from .fit import round_to_significance, Chi22CL
from .config import (FORMAT_ERROR_SIGNIFICANT_PLACES, F_SIGNIFICANCE_LEVEL,
                     M_MINIMIZER_TO_USE, log_file, null_file)
from .stream import StreamDup

logger = logging.getLogger('kafe')

def chi2( ydata, cov_mat,
         fdata):
    r'''
    The :math:`\chi^2` implementation. Calculates :math:`\chi^2` according
    to the formula:

    .. math::

        \chi^2 = \lambda^T C^{-1} \lambda


    Here, :math:`\lambda` is the residual vector :math:`\lambda = \vec{y} -
    \vec{f}(\vec{x})` and :math:`C` is the covariance matrix.


    Parameters
    ----------

    **ydata** : iterable
        The *y* measurement data

    **fdata** : iterable
        The function data

    **cov_mat** : `numpy.matrix`
        The total covariance matrix


    '''

    # calculate residual vector
    residual = ydata - fdata

    chi2val = (residual.T.dot(cov_mat.I).dot(residual))[0, 0]  # return the chi^2


    return chi2val

class Multifit(object):
    '''
    Object representing a Multifit. This object references the fitted `Dataset`,
    the fit function and the resulting fit parameters.

    Necessary arguments are a `Dataset` object and a fit function (which should
    be fitted to the `Dataset`).Multifit needs a list of tupels with the Dataset
    as a first entry and a Fitfunction as a second entry. Optionally, an external
    function `FCN` (the minimum of which should be located to find the best fit)
    can be specified. If not given, the `FCN` function defaults to :math:`\chi^2`.

    Parameters
    ----------
    **dataset_function** : list of tupels
        Each tupel has to contain a dataset as the first entry and the
        fit_function (for that dataset) as the second entry.

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
        keyword argument.
    '''
    def __init__(self, dataset_function,external_fcn=chi2,
                 fit_name=None, fit_label=None,
                 minimizer_to_use=M_MINIMIZER_TO_USE, quiet = False):

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

        self._minuit_lists_outdated = True
        self.fit_list = []
        for dataset, fit_function in dataset_function:
            self.fit_list.append(kafe.Fit(dataset, fit_function, quiet=True))

        # Create the Parameterspace for this multifit
        self.parameter_space = _ParameterSpace(self.fit_list)

        # Bool to store if datasets are the same
        self.corelate_datasets = False

        self.first_cov_mat_y = self._build_cov_mat_datapoints('y')
        self.current_cov_mat = self.first_cov_mat_y

        self.first_cov_mat_x = None
        if self.has_errors('x'):
            self.first_cov_mat_x = self._build_cov_mat_datapoints('x')


        # Total number of parameters
        self.total_number_of_parameters = 0

        # List for the minimizer. Sorted after the id system in the Parameterspace.
        self.current_parameter_values_minuit = None
        self.current_parameter_errors_minuit = None
        self.parameter_names_minuit = None
        self.latex_parameter_names_minuit = None

        self.minimizer_to_use = minimizer_to_use
        self.quiet_minuit = quiet
        # Init a object to hold the minimizer which will be initilized ind dofit()
        self.minimizer = None

        #: the (external) function to be minimized for this `MultiFit`
        self.external_fcn = external_fcn

        self.fit_label = fit_label

        self.fit_name = fit_name

        # store a dictionary to lookup whether a parameter is fixed
        self._fixed_parameters = None
        self.number_of_fixed_parameters = 0

        # Store all datasets/functions in the corresponding lists
        for fit in self.fit_list:
            self.total_number_of_parameters += len(fit.parameter_names)

        #: this `Fit`'s minimizer (`Minuit`)
        if type(self.minimizer_to_use) is str:
            # if specifying the minimizer type using a string
            if self.minimizer_to_use.lower() == "root":
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
                self._minimizer_handle = Minuit
            elif self.minimizer_to_use.lower() == "iminuit":
                from .iminuit_wrapper import IMinuit
                self._minimizer_handle = IMinuit
                # raise NotImplementedError, "'iminuit' minimizer not yet implemented"
            else:
                raise ValueError("Unknown minimizer '%s'" % (self.minimizer_to_use,))
        elif self.minimizer_to_use ==None:
            # No minimzer need. Testing some stuff
            self._minimizer_handle = None
        else:
            # assume class reference is given
            self._minimizer_handle = self.minimizer_to_use

        # Init the minimizer. If any links are given, a new minimizer will be created.

        self._init_minimizer()

        # Define a stream for storing the output
        if quiet== False:
            if self.fit_list[0].dataset.basename is not None:
                _basename = self.fit_list[0].dataset.basename
            else:
                _basename = 'untitled'

            if self.fit_name is not None:
                _basename += '_' + fit_name
            _basenamelog = log_file(_basename + '.log')
            # check for old logs
            if os.path.exists(_basenamelog):
                logger.info('Old log files found for fit `%s`. kafe will not '
                            'delete these files, but it is recommended to do '
                            'so, in order to reduce clutter.'

                            % (_basename,))

                # find first incremental name for which no file exists
                _id = 1
                while os.path.exists(log_file(_basename + '.' + str(_id) + '.log')):
                    _id += 1

                # move existing log to that location
                os.rename(_basenamelog, log_file(_basename + '.' + str(_id) + '.log'))
            self.out_stream = StreamDup([log_file('fit.log'), _basenamelog])
        else:
            self.out_stream= StreamDup([null_file()])

    def has_errors(self, axis):
        '''
        Checks if any dataset has errors on the given axis.
        '''
        for fit in self.fit_list:
            if fit.dataset.has_errors(axis):
                return True

        return False

    def autolink_datasets(self):
        '''
        Correlates all datasets, which are the same. This will have an effect in
        calculating :math:`\chi^2`.
        '''
        self.corelate_datasets = True
        self.first_cov_mat_y = self._build_cov_mat_datapoints('y')
        self.current_cov_mat = self.first_cov_mat_y

        if self.has_errors('x'):
            self.first_cov_mat_x = self._build_cov_mat_datapoints('x')

    def autolink_parameters(self):
        '''
        Autolinks all parameters with the same name.
        '''
        if self.number_of_fixed_parameters ==0:
            self._minuit_lists_outdated = True
            self.parameter_space.autolink_parameters()
        else:
            logger.warning("Cannot link parameter after a parameter was fixed. Release the parameter first.")

    def delink_parameters(self, param1, param2):
        '''
        Delinks two linked parameters.

        Parameters
        ----------
        **param1**: string
            Name of the first parameter. Will be translated to the intern parameter name
            (Function_name.parameter_name).
        **param2**: string
            Name of the second parameter.Will be translated to the intern parameter name
            (Function_name.parameter_name).
        '''
        if self.number_of_fixed_parameters ==0:
            self.parameter_space.delink(param1,param2)
            self._minuit_lists_outdated = True
        else:
            logger.warning("Cannot delink parameter after a parameter was fixed. Release the parameter first.")

    def link_parameters(self, param1, param2):
        '''
        Links two parameters together. Linked parameters will be considered
        the same parameter for the fit. link_parameters creates a dictionary
        entry in the parameterspace.alias. parameterspace.alias works as a translator.

        Parameters
        ----------
        **param1**: string
            Name of the first parameter. Will be translated to the intern parameter name
            (Function_name.parameter_name).
        **param2**: string
            Name of the second parameter.Will be translated to the intern parameter name
            (Function_name.parameter_name).

        '''
        if self.number_of_fixed_parameters ==0:
            if param1 != param2:
                self._minuit_lists_outdated = True
                self.parameter_space.link_parameters(param1,param2)
            else:
                logger.warning("Cannot link 2 parameters with the same name. Use autolink_parameters or use"
                               "the function.parameter_name notation")
        else:
            logger.warning("Cannot link parameter after a parameter was fixed. Release the parameter first.")

    def print_par_ids(self):
        '''
        Prints the parameters with their ids for the user
        '''
        print(self.parameter_space.parameter_to_id)

    def print_linked_parameters(self):
        '''
        Prints all linked parameters.
        '''
        print(self.parameter_space.alias)

    def set_parameter(self, *args, **kwargs):
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
        function = kwargs.pop('function', None)
        if args:  # if positional arguents provided
            if len(args) == 1:
                par_values, par_errors = args[0], None
            elif len(args) == 2:
                par_values, par_errors = args[0], args[1]
            else:
                raise Exception("Error setting parameters. The argument "
                                "pattern for method `set_parameters` could "
                                "not be parsed.")

            if len(par_values) == self.total_number_of_parameters:
                #: the current values of the parameters
                self.current_parameter_values_minuit = list(par_values)
            else:
                raise Exception("Cannot set parameters. Number of given "
                                "parameters (%d) doesn't match the Fit's "
                                "parameter number (%d)."
                                % (len(par_values), self.total_number_of_parameters))

            if par_errors is not None:
                if len(par_values) == self.total_number_of_parameters:
                    #: the current uncertainties of the parameters
                    self.current_parameter_errors_minuit = list(par_errors)
                else:
                    raise Exception("Cannot set parameter errors. Number of "
                                    "given parameter errors (%d) doesn't "
                                    "match the Fit's parameter number (%d)."
                                    % (len(par_errors),
                                       self.total_number_of_parameters))
            else:
                if not no_warning:
                    logger.warn("Parameter starting errors not given. Setting "
                                "to 1/10th of the parameter values.")
                #: the current uncertainties of the parameters
                self.current_parameter_errors_minuit = [
                    val/10. if val else 0.1  # handle the case val = 0
                    for val in self.current_parameter_values_minuit
                ]
        else:  # if no positional arguments, rely on keywords

            for param_name, param_spec in kwargs.items():
                if function:
                    param_name = function.name + str('.') + param_name
                par_id = self.parameter_space.get_parameter_ids([param_name])[0]
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
                self.current_parameter_values_minuit[par_id] = param_val
                self.current_parameter_errors_minuit[par_id] = param_err

        # try to update the minimizer's parameters
        # (fails if minimizer not yet initialized)
        if self._minimizer_handle:
            try:
                # set Minuit's start parameters and parameter errors
                self.minimizer.set_parameter_values(
                    self.current_parameter_values_minuit)
                self.minimizer.set_parameter_errors(
                    self.current_parameter_errors_minuit)
            except AttributeError:
                if not no_warning:
                    logger.warn("Failed to set the minimizer's parameters. "
                                "Maybe minimizer not initialized for this Fit "
                                "yet?")

    def fix_parameters(self, parameters_to_fix, parameters_to_fix_value= None):
        '''
        Fixes a parameter for this fit. Fixed parameters will not be minimized. All linking
        must be done before fixing is done.

        Parameters
        ----------
        **parameters_to_fix**: list of strings
            A list of strings with the parameternames as an entry
        '''
        par_id = []
        for parameter in parameters_to_fix:
            parameter_found = False
            for fit in self.fit_list:
                for param in fit.fit_function.parameter_names:
                    if param == parameter:
                        parameter = fit.fit_function.name+ str('.')+ param
                        parameter_found = True
            if parameter_found== False:
                logger.warning("Parameter '%s' not found. Parameter was not fixed " % (parameter,))
                break
            else:
                par_id.append(self.parameter_space.get_parameter_ids([parameter])[0])
                logger.info("Fixed parameter %d (%s)" % (par_id[-1], parameter))
        # found parameter, fix it
        if self._minuit_lists_outdated:
            self._init_minimizer()

        for i, id in enumerate(par_id):
            if parameters_to_fix_value:
                self.current_parameter_values_minuit[id]=parameters_to_fix_value[i]
            if self._minimizer_handle:
                self.minimizer.fix_parameter(id)
                self.minimizer.set_parameter_values(
                    self.current_parameter_values_minuit)
                self.minimizer.set_parameter_errors(
                    self.current_parameter_errors_minuit)
            self.current_parameter_errors_minuit[id] = 0


        self.number_of_fixed_parameters += len(par_id)
        #self._fixed_parameters[par_id] = True

    def release_parameters(self, *parameters_to_release):
        '''
        Release the given parameters so that the minimizer begins to work with
        them when :py:func:`do_fit` is called next. Parameters must be given by
        their
        names. If no arguments are provied, then release all
        parameters.
        '''
        if parameters_to_release:
            for parameter in parameters_to_release:
                parameter_found = False
                for fit in self.fit_list:
                    for param in fit.parameter_names:
                        if param == parameter:
                            parameter = fit.fit_function.name + str('.') + param
                            parameter_found = True

                if parameter_found == False:
                    logger.warning("Parameter not found. No parameter was released")
                else:
                    par_id = self.parameter_space.get_parameter_ids([parameter])
                    # Release found parameter
                    self.minimizer.release_parameter(par_id[0])
                    self.number_of_fixed_parameters -= 1
                    #self._fixed_parameters[par_id] = False
                    logger.info("Released parameter %d (%s)" % (par_id[-1], parameter))

        else:
            # go through all parameter IDs
            for par_id in range(self.total_number_of_parameters):
                # Release parameter
                self.minimizer.release_parameter(par_id)
            # Inform about release
            logger.info("Released all parameters")

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
        if self.minimizer:
            output = []
            for name, value, error in self.minimizer.get_parameter_info():

                if rounding:
                    value, error = round_to_significance(value, error)
                output.append(error)

            return tuple(output)
        else:
            logger.warning("Minimizer not yet called. Use do_fit first to call minimizer")

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
        if self.minimizer:
            output = []
            for name, value, error in self.minimizer.get_parameter_info():

                if rounding:
                    value, error = round_to_significance(value, error)
                output.append(value)

            return tuple(output)
        else:
            logger.warning("Minimizer not yet called. Use do_fit first to call minimizer")

    def get_error_matrix(self):
        '''
        This method returns the covariance matrix of the fit parameters which
        is obtained by querying the minimizer object for this `Fit`

        Returns
        -------

        *numpy.matrix*
            The covariance matrix of the parameters.
        '''
        if self.minimizer:
            return self.minimizer.get_error_matrix()
        else:
            logger.warning("Minimizer not yet called. Use do_fit first to call minimizer")

    # Private Methods
    ##################

    def _build_cov_mat_datapoints(self, axis):
        '''
        Builds the Cov_mat for the data points for the given axis. The cov_mat will take in account
        if 2 datasets are the same and correlate them, if self.corelate_datasets is True.


        '''
        dummy2 = []
        __querry_dummy2 = []
        for i,fit in enumerate(self.fit_list):
            # Create mask to store points where a non 0 matrix is needed.
            __querry = [False] * len(self.fit_list)
            # Diagonal entrys are never 0
            __querry[i] = True
            dummy2.append([0] * len(self.fit_list))
            # Check if datsets are correlated. If so change non diagonal elements.
            if self.corelate_datasets:
                if np.allclose(self.fit_list[i].dataset.get_data(axis), self.fit_list[i-1].dataset.get_data(axis),
                               atol=0, rtol=1e-4):
                    __querry[i-1] = True
            __querry_dummy2.append(__querry)

        for i,list in enumerate(dummy2):
            for j,entry in enumerate(list):
                if __querry_dummy2[i][j]:
                    dummy2[i][j] = self.fit_list[i].current_cov_mat
                else:
                    dummy2[i][j] = np.zeros((self.fit_list[i].dataset.get_size(), self.fit_list[j].dataset.get_size()))
        return np.bmat(dummy2)

    def _call_external_fcn(self, *parameter_values):
        '''
        Wrapper for the external `FCN`. Since the actual fit process depends on
        finding the right parameter values we can calculate the function datapoints
        for each given x data point. The external function now can build the residual.

        Parameters
        ----------

        **parameter_values** : sequence of values
            the parameter values at which `FCN` is to be evaluated

        '''

        _ydata = np.zeros(len(self.current_cov_mat))
        _fdata = np.zeros(len(self.current_cov_mat))
        i = 0
        for fit in self.fit_list:
            _parameter_values = self.parameter_space.get_current_parameter_values(parameter_values, fit.fit_function)

            def tmp_fit_function(x):
                return fit.fit_function(x, *_parameter_values)

            _dummy_ydata = fit.ydata
            _dummy_fdata = np.asarray(list(map(tmp_fit_function, fit.xdata)))
            for j, data in enumerate(_dummy_fdata):
                _ydata[i] = _dummy_ydata[j]
                _fdata[i] = _dummy_fdata[j]
                i += 1

        _chi2 = chi2(_ydata, self.current_cov_mat, _fdata)

        return _chi2

    # Fit Workflow
    ###############

    def do_fit(self, quiet=False, verbose=False):
        '''
        Runs the fit algorithm for this `MultiFit` object.

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

        All those steps are done for all Datasets and all Fitfunctions to create
        a multifit.

        Keyword Arguments
        -----------------

        quiet : boolean, optional
            Set to ``True`` if no output should be printed.

        verbose : boolean, optional
            Set to ``True`` if more output should be printed.
        '''

        # Check if lists are up to date. If not recalculate them
        if self._minuit_lists_outdated:
            self._init_minimizer()
        if self._minimizer_handle:
            max_x_iterations = 10
            logger.debug("Calling Minuit")
            self._call_minimizer(final_fit=True, verbose=verbose)
            # if the dataset has x errors, project onto the current error matrix
            if self.first_cov_mat_x is not None:
                logger.debug("Dataset has `x` errors. Iterating for `x` error.")
                iter_nr = 0
                while iter_nr < max_x_iterations:

                    old_matrix = copy(self.current_cov_mat)
                    self._project_x_covariance_matrix()
                    logger.debug("`x` fit iteration %d" % (iter_nr,))
                    if iter_nr == 0:
                        self._call_minimizer(final_fit=False, verbose=verbose)
                    else:
                        self._call_minimizer(final_fit=True, verbose=verbose)
                    new_matrix = self.current_cov_mat

                    # stop if the matrix has not changed within tolerance)
                    # GQ: adjusted precision: rtol 1e-4 on cov-matrix is
                    # clearly sufficient
                    if np.allclose(old_matrix, new_matrix, atol=0, rtol=1e-4):
                        logger.debug("Matrix for `x` fit iteration has converged.")
                        break  # interrupt iteration
                    iter_nr += 1

            self.par_cov_mat = self.get_error_matrix()


            # determine, retrieve and analyze errors from MINOS algorithm
            tol = 0.05
            self.minos_errors = self.minimizer.minos_errors()
            # error analysis:
            for par_nr, par_val in enumerate(self.current_parameter_values_minuit):
                ep = self.minos_errors[par_nr][0]
                em = self.minos_errors[par_nr][1]
                if ep != 0 and em != 0:
                    if (abs(ep + em) / (ep - em) > tol) or \
                            (abs(1. - 0.5 * (ep - em) / self.minos_errors[par_nr][2]) > tol):
                        self.parabolic_errors = False

        # store results ...
        self.final_parameter_values = self.current_parameter_values_minuit
        self.final_parameter_errors = self.current_parameter_errors_minuit
        if self._minimizer_handle:
            self.final_fcn = self.minimizer.get_fit_info('fcn')

        for fit in self.fit_list:
            fit.final_parameter_values = self.parameter_space.get_current_parameter_values(
                self.current_parameter_values_minuit, fit.fit_function)
            fit.final_parameter_errors = self.parameter_space.get_current_parameter_values(
                self.current_parameter_errors_minuit, fit.fit_function)
            fit.set_parameters(fit.final_parameter_values, fit.final_parameter_errors)
            self._construct_cov_mat(fit)
        # ... and print at end of fit
        if not quiet:
            self.print_fit_results()
            self.print_rounded_fit_parameters()
            self.print_fit_details()

    # Private Methods
    ##################

    def _calculate_minuit_lists(self):
        '''
        Recalculates or creates all lists which the minimizer needs for his workflow.
        If some parameter gets changed (fixed or linked) the lists will be recalculated and
        the minimizer will be initialized again.
        '''

        dic = self.parameter_space.build_current_parameter()
        self.parameter_names_minuit = dic['names']
        self.current_parameter_values_minuit = dic['values']
        self.current_parameter_errors_minuit = dic['errors']
        self.latex_parameter_names_minuit = dic['latex_names']
        self._minuit_lists_outdated = False
        self.total_number_of_parameters = len(self.parameter_names_minuit)

    def _init_minimizer(self):
        '''
        Starts the minimizer. Gets called after the minuit lists are calaculated
        '''
        # Init the minimizer
        self._calculate_minuit_lists()
        if self._minimizer_handle:
            self.minimizer = self._minimizer_handle(self.total_number_of_parameters,
                                                    self._call_external_fcn, self.parameter_names_minuit,
                                                    self.current_parameter_values_minuit,
                                                    self.current_parameter_errors_minuit, quiet=self.quiet_minuit)

            # set Minuit's initial parameters and parameter errors
            #            may be overwritten via ``set_parameters``
            self.minimizer.set_parameter_values(
                self.current_parameter_values_minuit)

            self.minimizer.set_parameter_errors(
                self.current_parameter_errors_minuit)

    def _call_minimizer(self, final_fit=True, verbose=False):
        '''
        Instructs the minimizer to do a minimization.
        '''

        verbosity = 0
        if(final_fit):
            verbosity = 2
        if (verbose):
            verbosity = 3

        logger.debug("Calling minimizer")
        self.minimizer.minimize(
            final_fit=final_fit,log_print_level=verbosity)
        logger.debug("Retrieving data from minimizer")

        self.current_parameter_values_minuit = list(self.minimizer.get_parameter_values())
        self.current_parameter_errors_minuit = list(self.minimizer.get_parameter_errors())

    def _project_x_covariance_matrix(self):
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
        precision_list = 0.01 * np.sqrt(np.diag(self.current_cov_mat))

        if min(precision_list) == 0:
            logger.warn('At least one input error is zero - set to 1e-7')
            for i, p in enumerate(precision_list):
                if not p:
                    precision_list[i] = 1.e-7
        _tmp = []
        for fit in self.fit_list:
            _tmp.append(fit.fit_function.derive_by_x(fit.dataset.get_data('x'),
                                                     precision_list,
                                                     self.parameter_space.get_current_parameter_values(self.current_parameter_values_minuit, fit.fit_function)))


        outer_prod = outer_product(np.concatenate(_tmp))

        proj_xcov_mat = np.asarray(self.first_cov_mat_x) * outer_prod

        self.current_cov_mat = self.first_cov_mat_y + \
                               np.asmatrix(proj_xcov_mat)

    # Output functions
    ###################

    def _construct_cov_mat(self, fit):
        '''
        Called after the fit to construct the cov_mat for each fit object for plotting

        '''
        Cor = MinuitCov_to_cor(self.par_cov_mat)
        number_of_parameters = len(fit.parameter_names)
        new_cor = np.zeros((number_of_parameters,number_of_parameters))
        dummy = self.parameter_space.fit_to_parameter_id(fit)

        # Create a new correlation Matrix
        for i, id1 in enumerate(dummy):
            for j, id2 in enumerate(dummy):
                new_cor[i][j] = Cor[id1][id2]

        # Calculate new covariance matrix
        new_cov = cor_to_cov(new_cor, fit.final_parameter_errors)
        fit.par_cov_mat = new_cov

    def get_final_parameter_values(self, function=None):
        '''
        Returns the final parameter values for the given function with the results from
        the minimizer. The list current_parameter_values holds all parameters from
        all fits.

        Parameters
        ----------

        **current_parameter_values**: List
            List with the current parameter values from the minimizer. List is sorted
            after the ids from self.parameter_to_id.

        Keyword Arguments
        -----------------

        **fit_function** : function
            A user-defined Python function to fit to the data. If no function is given,
            the values from the minimizer are given.

        Returns
        -------
        List
            List with the final parameter values.
        '''
        if function:
            return self.parameter_space.get_current_parameter_values(self.current_parameter_values_minuit,function)
        else:
            return self.current_parameter_values_minuit

    def get_final_parameter_errors(self, function=None):
        '''
        Returns the final parameter errors for the given function with the results from
        the minimizer. The list current_parameter_values holds all parameters from
        all fits.

        Parameters
        ----------

        **current_parameter_values**: List
            List with the current parameter values from the minimizer. List is sorted
            after the ids from self.parameter_to_id.

        Keyword Arguments
        -----------------

        **fit_function** : function
            A user-defined Python function to fit to the data. If no function is given,
            the errors from the minimizer are given.

        Returns
        -------
        List
            List with the final parameter errors.
        '''
        if function:
            return self.parameter_space.get_current_parameter_values(self.current_parameter_errors_minuit, function)
        else:
            return self.current_parameter_errors_minuit

    def get_final_parameter_names(self, function=None):
        '''
        Returns the parameter names for the given function with the results from
        the minimizer.

        Parameters
        ----------

        **current_parameter_values**: List
            List with the current parameter values from the minimizer. List is sorted
            after the ids from self.parameter_to_id.

        Keyword Arguments
        -----------------

        **fit_function** : function
            A user-defined Python function to fit to the data. If no function is given,
            the names from the minimizer are given.

        Returns
        -------
        List
            List with the final parameter names
        '''
        if function:
            return function.parameter_names
        else:
            return self.parameter_names_minuit

    def print_rounded_fit_parameters(self):
        '''prints the fit parameters'''

        print("########################", file=self.out_stream)
        print("# Final fit parameters #", file=self.out_stream)
        print("########################", file=self.out_stream)
        for fit in self.fit_list:
            print('', file= self.out_stream)
            if len(self.fit_list)>1:
                print("%s :" %(fit.fit_function.name), file=self.out_stream)
                print('', file= self.out_stream)
            id_list = self.parameter_space.fit_to_parameter_id(fit)
            for i,id in enumerate(id_list):
                name = fit.parameter_names[i]
                value = self.current_parameter_values_minuit[id]
                error = self.current_parameter_errors_minuit[id]
                tmp_rounded = round_to_significance(value, error, FORMAT_ERROR_SIGNIFICANT_PLACES)
                if error:
                    print("%s = %g +- %g" % (name, tmp_rounded[0], tmp_rounded[1]), file=self.out_stream)
                else:
                    print("%s = %g    -fixed-" % (name, tmp_rounded[0]), file=self.out_stream)
        if self.parameter_space.alias:
            print('', file= self.out_stream)
            print("#####################", file=self.out_stream)
            print("# Linked parameters #", file=self.out_stream)
            print("#####################", file=self.out_stream)
            print('', file= self.out_stream)
            for fit in self.fit_list:
                for name in fit.parameter_names:
                    name = fit.fit_function.name+ str('.')+name
                    if name in self.parameter_space.alias:
                        print("%s = %s "
                              % (name.split(".", 1)[1], self.parameter_space.alias[name].split(".", 1)[1]),
                              file=self.out_stream)
        print('', file=self.out_stream)

    def print_fit_results(self):
        '''prints fit results'''

        print('##############', file=self.out_stream)
        print('# Fit result #', file=self.out_stream)
        print('##############', file=self.out_stream)
        print('', file=self.out_stream)
        par_err = extract_statistical_errors(self.par_cov_mat)
        par_cor_mat = MinuitCov_to_cor(self.par_cov_mat)

        if self.total_number_of_parameters > 1:
            print('# value        error   correlations', file=self.out_stream)
        else:
            print('# value        error   ', file=self.out_stream)

        for par_nr, par_val in enumerate(self.final_parameter_values):
            print('# '+self.parameter_names_minuit[par_nr].split(".", 1)[1], file=self.out_stream)
            print(format(par_val, '.04e')+'  ', file=self.out_stream, end='')
            if par_err[par_nr]:
              print(format(par_err[par_nr], '.02e')+'  ', file=self.out_stream, end='')
            else:
              print('-fixed- ', file=self.out_stream, end='')
            if par_nr > 0 and par_err[par_nr]:
                for i in range(par_nr):
                    print(format(par_cor_mat[par_nr, i], '+.3f')+'  ', file=self.out_stream, end='')
            print('', file=self.out_stream)
        #print MINOS errors if needed
        if(not self.parabolic_errors):
            print('!!! uncertainties from MINOS:', file=self.out_stream)
            for par_nr, par_val in enumerate(self.final_parameter_values):
                print('# '+self.parameter_names_minuit[par_nr], file=self.out_stream)
                if par_err[par_nr]:
                    print('     '
                          '+'+format(self.minos_errors[par_nr][0],'.02e')+
                          ' '+format(self.minos_errors[par_nr][1],'.02e'),
                          file=self.out_stream)
                else:
                    print('-fixed- ', file=self.out_stream, end='')
            print('', file=self.out_stream)
        print('', file=self.out_stream)

    def print_fit_details(self):
        '''prints some fit goodness details'''

        _ndf = 0
        for fit in self.fit_list:
            _ndf += fit.dataset.get_size()
        _ndf -= self.total_number_of_parameters

        chi2prob = self.minimizer.get_chi2_probability(_ndf)
        if chi2prob < F_SIGNIFICANCE_LEVEL:
            hypothesis_status = 'rejected (sig. %d%s)' \
                                % (int(F_SIGNIFICANCE_LEVEL * 100), '%')
        else:
            hypothesis_status = 'accepted (sig. %d%s)' \
                                % (int(F_SIGNIFICANCE_LEVEL * 100), '%')

        print('###############', file=self.out_stream)
        print("# Fit details #", file=self.out_stream)
        print("###############", file=self.out_stream)
        print('', file=self.out_stream)

        # Print a warning if NDF is zero
        if not _ndf:
            print("# WARNING: Number of degrees of freedom is zero!", file=self.out_stream)
            print("# Please review parameterization...", file=self.out_stream)
            print('', file=self.out_stream)
        elif _ndf < 0:
            print("# WARNING: Number of degrees of freedom is negative!", file=self.out_stream)
            print("# Please review parameterization...", file=self.out_stream)
            print('', file=self.out_stream)
        if (not self.parabolic_errors):
            print('Attention: use uncertainties from MINOS', file=self.out_stream)
            print('', file=self.out_stream)

        print('USING    %s' % (self.minimizer.name), file=self.out_stream)
        print('FCN/ndf  %.3g/%d = %.3g' % (
                self.minimizer.get_fit_info('fcn'), _ndf,
                self.minimizer.get_fit_info('fcn') / (_ndf)), file=self.out_stream)
        print('EdM      %g' % (self.minimizer.get_fit_info('edm')), file=self.out_stream)
        print('UP       %g' % (self.minimizer.get_fit_info('err_def')), file=self.out_stream)
        print('STA     ', self.minimizer.get_fit_info('status_code'), file=self.out_stream)
        print('', file=self.out_stream)
        print('chi2prob', round(chi2prob, 3), file=self.out_stream)
        print('HYPTEST ', hypothesis_status, file=self.out_stream)
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
        par1_val = self.current_parameter_values_minuit[parameter1]
        par2_val = self.current_parameter_values_minuit[parameter2]
        par1_err = self.current_parameter_errors_minuit[parameter1]
        par2_err = self.current_parameter_errors_minuit[parameter2]
        xval, yval = par1_val, par2_val
        xer, yer = par1_err , par2_err

        plt.tight_layout()
        if axes is None:
            #new (square) figure for contour(s)
            tmp_fig = plt.figure(figsize=(5., 5.))
            # get/create axes object for current figure
            tmp_ax = tmp_fig.gca()

        else:
            tmp_ax = axes
        # set axis labels
        tmp_ax.set_xlabel('$%s$' %  (self.latex_parameter_names_minuit[parameter1],),
                          fontsize='xx-large')
        tmp_ax.set_ylabel('$%s$' %  (self.latex_parameter_names_minuit[parameter2],),
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
            xs, ys = self.minimizer.get_contour(parameter1, parameter2, n_points)
            # store result
            self.contours.append([parameter1, parameter2, dc2, xs, ys])
            # plot contour lines
            cl=100*Chi22CL(dc2) # get corresponding confidence level
            print('Contour %.1f %%CL for parameters %d vs. %d with %d points'
                  % (cl, parameter1, parameter2, len(xs)), file=self.out_stream)
            labelstr = "%.1f"%(cl) + r"\% CL"
            tmp_ax.fill(xs, ys, alpha=alpha, color=color)   # as filled area
            tmp_ax.plot(xs, ys, '--', linewidth=2, label=labelstr)  # as line
        print('', file=self.out_stream)
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
        #id = self._find_parameter(parid)
        id= self.parameter_names_minuit[parid]
        _pvals = self.current_parameter_values_minuit
        _perrs = self.current_parameter_errors_minuit
        val = _pvals[parid]
        err = _perrs[parid]

        print('Profile for parameter %d with %d points' % (parid, n_points), file=self.out_stream)

        plt.tight_layout()
        if axes is None:
            # new (square) figure for contour(s)
            tmp_fig = plt.figure(figsize=(5., 5.))
            # get/create axes object for current figure
            tmp_ax = tmp_fig.gca()
        else:
            tmp_ax = axes
        # set axis labels
        tmp_ax.set_xlabel('$%s$' % (self.latex_parameter_names_minuit[parid],),fontsize='xx-large')
        tmp_ax.set_ylabel('$%s$' % ('\\Delta \\chi^2'),fontsize='xx-large')
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
        xp, yp = self.minimizer.get_profile(parid, n_points)
        self.profiles.append([parid, xp, yp])  # store this result
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

    def plot_correlations(self, function=None):
        '''
        Plots two-dimensional contours for all pairs of parameters
        and profile for all parameters, arranges as a matrix.

        Keyword Arguments
        -----------------

        **function** : function
            Python fit function. If the keyword is given only the contours
            of the parameters from the function will be ploted.

        Returns
        -------

        ``matplotlib`` figure object
            A figure object containing the matrix of plots.
        '''
        if function is not None:
            npar = len(self.parameter_space.function_to_parameter[function])
            cor_fig, axarr = plt.subplots(
                npar, npar, figsize=(5. * npar, 5. * npar))
            self._fixed_parameters = np.zeros(self.total_number_of_parameters,
                                              dtype=bool)
            par = self.parameter_space.function_to_parameter[function]
            ids = self.parameter_space.get_parameter_ids(par)

            ip = -1
            for i in range(0, self.total_number_of_parameters):
                if not self._fixed_parameters[i]:
                    ip += 1
                jp = -1
                for j in range(0, self.total_number_of_parameters):
                    if not self._fixed_parameters[j]:
                        jp += 1
                    # skip fixed parameters
                    if not (self._fixed_parameters[i] or self._fixed_parameters[j]):
                        if ip > jp:
                            if ip in ids and jp in ids:
                                # empty space
                                axarr[jp, ip].axis('off')
                        elif ip == jp:
                            if i in ids:
                                # plot profile
                                self.plot_profile(i, axes=axarr[ip, ip])
                        else:
                            if i in ids and j in ids and i !=j:
                                # plot contour
                                self.plot_contour(
                                    i, j, dchi2=[1., 2.3], axes=axarr[jp, ip])

        else:
            npar = self.total_number_of_parameters - self.number_of_fixed_parameters
            cor_fig, axarr = plt.subplots(
                npar, npar, figsize=(5. * npar, 5. * npar))
            self._fixed_parameters = np.zeros(self.total_number_of_parameters,
                                              dtype=bool)
            ip = -1
            for i in range(0, self.total_number_of_parameters):
                if not self._fixed_parameters[i]:
                    ip += 1
                jp = -1
                for j in range(0, self.total_number_of_parameters):
                    if not self._fixed_parameters[j]:
                        jp += 1
                    # skip fixed parameters
                    if not (self._fixed_parameters[i] or self._fixed_parameters[j]):
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



class _ParameterSpace(object):
    '''
    Object used to manage the parameters from a multifit. Multifit tells the ParameterSpace
    which parameters get linked. Before the minimizer gets initilized Multifit creates the
    lists for the minimizer using the ParameterSpace.

    Parameters
    ----------
    **fit_list**: list containing fit.py objects
        A list containing all fit.py objects used for this multifit.

    '''
    def __init__(self, fit_list):
        logger = logging.getLogger('kafe')
        self.parameter_changed_bool = True
        # Total number of parameters
        self.total_number_of_parameters = 0

        # Dictionary which gives each parameter a individual id. If parameters get linked, the id
        # is changed so both have the same id
        self.parameter_to_id = {}

        # Dictionary which holds the function as a key and the parameters as value
        self.function_to_parameter = {}

        # Dictionary which takes a parameter name as a key and gives back the linked parameter
        self.alias = {}

        self.fit_list = fit_list

        for fit in self.fit_list:
            # Set the beginning values for each parameter
            _tmp = []
            for i in range(len(fit.fit_function.parameter_names)):
                _tmp.append(fit.fit_function.name + str(".") + fit.fit_function.parameter_names[i])
                self.total_number_of_parameters += 1
            # Create dictionary entry for each function with the given parameters as values
            self.function_to_parameter.update({fit.fit_function: _tmp})

        self.build_current_parameter()

    def autolink_parameters(self):
        '''
        Autolinks all parameters with the same name.
        '''
        # Get first parameter tupel
        for i, fit in enumerate( self.fit_list):
            # Get second parameter tupel. Note that to avoid doing the comparison a
            # second time only tupels ahead in the list will be taken.
            for j in range(i + 1, len( self.fit_list)):
                # Iterate over each tupel
                for param1 in fit.parameter_names:
                    for param2 in  self.fit_list[j].parameter_names:
                        # Check if the parameter name after the function name is the same
                        if param1 == param2:
                            # Link both parameters together
                            self.parameter_changed_bool = True
                            self.link_parameters(fit.fit_function.name + str('.') + param1,
                                                 self.fit_list[j].fit_function.name + str('.') + param2)

    def link_parameters(self, param1, param2):
        '''
        Links two parameters together. Linked parameters will be considered
        the same parameter for the fit. link_parameters creates a dictionary
        entry in self.alias. Self.alias works as a translator for the parameter
        space.

        Parameters
        ----------
        **param1**: string
            Name of the first parameter. Will be translated to the intern parameter name
            (Function_name.parameter_name).
        **param2**: string
            Name of the second parameter.Will be translated to the intern parameter name
            (Function_name.parameter_name).

        '''
        # Checks if parametername is given with function before, otherwise creates
        # parameter_name

        param1, param2 = self._convert_parameter_names(param1, param2)
        # A link between two parameters reduces the number of total parameters by one
        self.total_number_of_parameters -= 1
        self.parameter_changed_bool = True

        # Check if param2 is already a key.
        if param2 in self.alias:
            # If param2 is already a key try param1 as a key.
            self.alias.update({param1: param2})
            # Look if you can go through self.alias and end on param1. If so, a loop was
            # created. Delete the last entry in self.alias which caused the loop
            _param = param1
            while _param in self.alias:
                _param = self.alias[_param]

                if _param == param1:
                    # Give a warning for the user
                    logger.warn("Deleted already linked parameters")
                    del self.alias[param1]
                    self.total_number_of_parameters += 1
                    break
        else:
            self.alias.update({param2: param1})
            # Look if you can go through self.alias and end on param2. If so, a loop was
            # created. Delete the last entry in self.alias which caused the loop
            _param = param2
            while _param in self.alias:
                _param = self.alias[_param]
                if _param == param2:
                    # Give a warning for the user
                    logger.warn("Deleted already linked parameters")
                    del self.alias[param2]
                    self.total_number_of_parameters += 1
                    break

    def delink(self, param1, param2):
        '''
        Delinks two parameters. If the parameters can not be found raises a ValueError.

        Parameters
        ----------
        **param1**: string
            Name of the first parameter. Will be translated to the intern parameter name
            (Function_name.parameter_name).
        **param2**: string
            Name of the second parameter.Will be translated to the intern parameter name
            (Function_name.parameter_name).
        '''

        param1, param2 = self._convert_parameter_names(param1, param2)
        if param2 in self.alias:
            if param1 == self.alias[param2]:
                # Deleting a link between two parameters increases the number of total parameters by one
                self.total_number_of_parameters += 1
                self.parameter_changed_bool = True
                del self.alias[param2]
            else:
                raise logger.Warning('The link between %s and %s was not found. Maybe the parameters'
                                        'were linked indirectly? Use the direct link. ')
        elif param1 in self.alias:
            if param2 == self.alias[param1]:
                self.total_number_of_parameters += 1
                self.parameter_changed_bool = True
                del self.alias[param1]
            else:
                raise logger.Warning('The link between %s and %s was not found. Maybe the parameters'
                                     'were linked indirectly? Use the direct link. ' % (param1,param2))
        else:
            raise ValueError("None of the given parameters were found. No links were delted")

    def _convert_parameter_names(self,param1, param2=None):
        '''
        Checks if the parameters are given in the function_name.parameter_name style. If so returns those names
        If not creates those names.
        '''
        if param2:
            if "." not in param1 or "." not in param2:
                for fit in self.fit_list:
                    for param in fit.parameter_names:
                        if param == param1:
                            param1 = fit.fit_function.name + str('.') + param
                        elif param == param2:
                            param2 = fit.fit_function.name + str('.') + param
        else:
            if "." not in param1:
                for fit in self.fit_list:
                    for param in fit.parameter_names:
                        if param == param1:
                            param1 = fit.fit_function.name + str('.') + param
        if param2 :
            return param1, param2
        else:
            return param1

    def _update_parameter_to_id(self):
        '''
        Updates/Creates the parameter_to_id dictionary. Gets called if
        :py:meth:`~kafe.multifit.ParameterSpace.parameter_changed_bool` is True.
        '''
        id_counter = 0
        dic = {}
        # Creates the parameter to id dictionary which is used to generate the current
        # Parameter list for the minimizer. The dictionary takes a parameter and gives
        # a number back. This number determines at which place in the list the parameter
        # will be written
        for fit in self.fit_list:
            for parameter_name in fit.parameter_names:
                if fit.fit_function.name + str(".") + parameter_name not in self.alias:
                    dic.update({fit.fit_function.name + str(".") + parameter_name: id_counter})
                    id_counter += 1
        self.parameter_to_id = dic
        self.parameter_changed_bool = False

    def get_current_parameter_values(self, current_parameter_values, function):
        '''
        Builds the current parameter values for each function with the results from
        the minimizer. The list current_parameter_values holds all parameters from
        all fits. This function builds the lists for all functions, with all parameters
        at the right place so that :math:`\chi^2` can be calculated

        Parameters
        ----------

        **current_parameter_values**: List
            List with the current parameter values from the minimizer. List is sorted
            after the ids from self.parameter_to_id.

        **fit_function** : function
            A user-defined Python function to fit to the data.

        '''
        # Update parameter to id if needed
        if self.parameter_changed_bool:
            self._update_parameter_to_id()

        # Create list to store results

        _parameter_values = [0] * len(self.function_to_parameter[function])
        j = 0
        for param in self.function_to_parameter[function]:
            # Get the parameter ids
            if param in self.parameter_to_id:
                _parameter_values[j] = current_parameter_values[self.parameter_to_id[param]]
            else:
                # If id was not found the parameter is linked and has to be translated via self.alias
                # until the parameter id is found
                _param = self.alias[param]
                while _param not in self.parameter_to_id:
                    _param = self.alias[_param]
                _parameter_values[j] = current_parameter_values[self.parameter_to_id[_param]]
            j += 1

        return _parameter_values

    def build_current_parameter(self):
        '''
        Builds the lists ( :py:meth:`~kafe.multifit.Multifit.parameter_names_minuit`
        , :py:meth:`~kafe.multifit.Multifit.current_parameter_values_minuit`,
        :py:meth:`~kafe.multifit.Multifit.current_parameter_errors_minuit`,
        :py:meth:`~kafe.multifit.Multifit.latex_parameter_names_minuit`) for
        the minimizer. Parameter must be linked before this function to take effect.
        The list is sorted after the ids from :py:meth:`~kafe.multifit.ParameterSpace.parameter_to_id`.
        Ids start from 0 and end at self.total_number_of_parameters.
        No id in between is left out.
        Returns those 4 lists as a single list with each list as a entry. This is done to save
        computing time.
        '''
        parameter_names = [0] * self.total_number_of_parameters
        parameter_names_latex = [0] * self.total_number_of_parameters
        current_parameter = [0] * self.total_number_of_parameters
        current_parameter_error = [0] * self.total_number_of_parameters

        if self.parameter_changed_bool:
            self._update_parameter_to_id()
        # Creating the parameter_names/values/errors for the minimizer. Linked parameters
        # are taken into consideration.
        for fit in self.fit_list:
            for i, parameter_name in enumerate(fit.parameter_names):
                parameter_name = fit.fit_function.name+str(".")+parameter_name
                if parameter_name in self.parameter_to_id:
                    current_parameter[self.parameter_to_id[parameter_name]] = fit.current_parameter_values[i]
                    parameter_names[self.parameter_to_id[parameter_name]] = parameter_name
                    current_parameter_error[self.parameter_to_id[parameter_name]] = fit.current_parameter_errors[i]
                    parameter_names_latex[self.parameter_to_id[parameter_name]] = fit.latex_parameter_names[i]

        dic = {'names': parameter_names,
               'values': current_parameter,
               'errors': current_parameter_error,
               'latex_names': parameter_names_latex }
        return dic

    def fit_to_parameter_id(self, fit):
        '''
        Takes a fit as an argument an gives the ids of the parameter as a list back.
        '''
        dummy = []
        # Get parameter_ids from the fit
        for i, parameter in enumerate(fit.parameter_names):
            parameter = fit.fit_function.name + str('.') + parameter
            if parameter in self.parameter_to_id:
                id = self.parameter_to_id[parameter]
            else:
                _param = self.alias[parameter]
                while _param not in self.parameter_to_id:
                    _param = self.alias[_param]
                id = self.parameter_to_id[_param]

            dummy.append(id)
        return dummy

    def get_parameter_ids(self, parameter):
        '''
        Takes a list of parameters as an argument and gives the ids of those parameters as a list back.
        '''
        ids =[]

        for i in parameter:
            i = self._convert_parameter_names(i)
            if i in self.parameter_to_id:
                ids.append(self.parameter_to_id[i])
            else:
                _param = i
                while _param not in self.parameter_to_id:
                    _param = self.alias[_param]
                ids.append(self.parameter_to_id[_param])
        return ids
