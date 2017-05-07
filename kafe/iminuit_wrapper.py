'''
.. module:: iminuit
   :platform: Unix
   :synopsis: A submodule providing the `IMinuit` object, which uses
        the standalone function minimizer Python package *iminuit*.
        The `IMinuit` class has been adapted from the earlier `Minuit`
        wrapper for ROOT::TMinuit.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
'''

# ----------------------------------------------------------------
# Changes:
#  05-May-15    create module
#  08-Oct-16 GQ  printout level -1 if "quiet" specified
#                suppressed du2() if no printout requested
# ----------------------------------------------------------------

# import iminuit as python package
import iminuit

from .config import M_MAX_ITERATIONS, M_TOLERANCE, log_file, null_file
from .stream import redirect_stdout_to
from time import gmtime, strftime

import numpy as np
import scipy.stats as stats

import sys
import os

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')

# Constants
############

# And define some constants to pass to iminuit functions
P_DETAIL_LEVEL = 1
"""default level of detail for iminuit's output
(typical range: -1 to 3, default: 1)"""

# dictionary lookup for error codes
D_MATRIX_ERROR = {0: "Error matrix not calculated",
                  1: "Error matrix approximate!",
                  2: "Error matrix forced positive definite!",
                  3: "Error matrix accurate"}  #: Error matrix status codes

class IMinuit:
    '''
    A wrapper class for iminuit.
    '''

    # init signature exactly as for the 'Minuit' class
    def __init__(self, number_of_parameters, function_to_minimize,
                 parameter_names, start_parameters, parameter_errors,
                 quiet=True, verbose=False):
        '''
        Create an *iminuit* minimizer for a function `function_to_minimize`.
        Necessary arguments are the number of parameters and the function to be
        minimized `function_to_minimize`. The function `function_to_minimize`'s
        arguments must be numerical values. The same goes for its output.

        Another requirement is for every parameter of `function_to_minimize` to
        have a default value. These are then used to initialize Minuit.

        **number_of_parameters** : int
            The number of parameters of the function to minimize.

        **function_to_minimize** : function
            The function which `Minuit` should minimize. This must be a Python
            function with <``number_of_parameters``> arguments.

        **parameter_names** : tuple/list of strings
            The parameter names. These are used to keep track of the parameters
            in `Minuit`'s output.

        **start_parameters** : tuple/list of floats
            The start values of the parameters. It is important to have a good,
            if rough, estimate of the parameters at the minimum before starting
            the minimization. Wrong initial parameters can yield a local
            minimum instead of a global one.

        **parameter_errors** : tuple/list of floats
            An initial guess of the parameter errors. These errors are used to
            define the initial step size.

        *quiet* : boolean (optional, default: ``True``)
            If ``True``, suppresses all output from ``iminuit``.

        *verbose* : boolean (optional, default: ``False``)
            If ``True``, sets ``iminuit``'s print level to a high value, so
            that all output is logged.

        '''

        #: the name of this minimizer type
        self.name = "iminuit"

        #: the actual `FCN` called in ``FCN_wrapper``
        self.function_to_minimize = function_to_minimize

        #: number of parameters to minimize for
        self.number_of_parameters = number_of_parameters

        if not quiet:
            self.out_file = open(log_file("iminuit.log"), 'a')
        else:
            self.out_file = null_file()

        #: maximum number of iterations until ``iminuit`` gives up
        self.max_iterations = M_MAX_ITERATIONS

        #: ``iminuit`` errordef
        self.errordef = 1.0

        # set parameter names, initial values, errors (step size)
        self.set_parameter_names(parameter_names, update_iminuit=False)
        self.set_parameter_values(start_parameters, update_iminuit=False)
        self.set_parameter_errors(parameter_errors, update_iminuit=False)

        # need to construct signature of function_to_minimize
        _par_names = self.parameter_names
        _init_par_vals = self.current_parameters
        _init_par_errs = self.parameter_errors
        _init_par_dict = {}
        for _par, _val, _err in zip(_par_names, _init_par_vals, _init_par_errs):
            _init_par_dict[_par]= _val
            _init_par_dict["error_"+_par]= _err

        # initialize the minimizer
        self.__iminuit = iminuit.Minuit(self.function_to_minimize,
            forced_parameters=_par_names, errordef=self.errordef,
            **_init_par_dict)

        # set minimizer properties
        self.set_err()
        self.set_strategy()
        self.set_tolerance(M_TOLERANCE)

        # set print level according to flag
        if quiet:
            self.set_print_level(-1)      # suppress output
        elif verbose:
            self.set_print_level(3)      # detailed output
        else:
            self.set_print_level(1)      # frugal output


    def update_parameter_data(self, show_warnings=False):
        """
        (Re-)Sets the parameter names, values and step size in iminuit.
        """

        fitparam = self.__iminuit.fitarg.copy()   # copy minimizer arguments
        for parameter, value, err in zip(
                                    self.parameter_names,
                                    self.current_parameters,
                                    self.parameter_errors):
            fitparam[parameter] = value
            fitparam["error_"+parameter] = err
        # replace minimizer
        ##del self.__iminuit
        self.__iminuit = iminuit.Minuit(
            self.function_to_minimize,
            print_level=self.print_level,
            forced_parameters=self.parameter_names,
            errordef=self.errordef,
            **fitparam)

        return 0

    # Set methods
    ##############

    def set_print_level(self, print_level=P_DETAIL_LEVEL):
        '''Sets the print level for Minuit.

        *print_level* : int (optional, default: 1 (frugal output))
            Tells ``iminuit`` how much output to generate. The higher this
            value, the more output it generates.
        '''
        self.__iminuit.set_print_level(print_level)  # set Minuit print level
        self.print_level = print_level

    def set_strategy(self, strategy_id=1):
        '''Sets the strategy Minuit.

        *strategy_id* : int (optional, default: 1 (optimized))
            Tells ``iminuit`` to use a certain strategy. Refer to ``iminuit``'s
            documentation for available strategies.
        '''

        self.__iminuit.set_strategy(strategy_id)

    def set_err(self, up_value=1.0):
        '''Sets the ``UP`` value for Minuit.

        *up_value* : float (optional, default: 1.0)
            This is the value by which `FCN` is expected to change.
        '''
        # shadow errordef value
        self.errordef = up_value
        # Tell iminuit to use an up-value of 1.0
        self.__iminuit.set_errordef(up_value)

    def set_tolerance(self, tol):
        '''Sets the tolerance value for Minuit.

        **tol** : float
            The tolerance
        '''
        # shadow
        self.tolerance = tol
        # Tell iminuit tolerance
        self.__iminuit.tol = tol


    def set_parameter_values(self, parameter_values, update_iminuit=True):   # CAN_STAY
        '''
        Sets the fit parameters. If parameter_values=`None`, tries to infer
          defaults from the function_to_minimize.
        '''
        if len(parameter_values) == self.number_of_parameters:
            self.current_parameters = parameter_values
        else:
            raise Exception("Cannot get default parameter values from the \
            FCN. Not all parameters have default values given.")

        if update_iminuit:
            self.update_parameter_data()


    def set_parameter_names(self, parameter_names, update_iminuit=True):   # CAN_STAY
        '''Sets the fit parameter names.'''
        if len(parameter_names) == self.number_of_parameters:
            self.parameter_names = parameter_names
        else:
            raise Exception("Cannot set parameter names. "
                            "Tuple length mismatch.")

        if update_iminuit:
            self.update_parameter_data()


    def set_parameter_errors(self, parameter_errors=None, update_iminuit=True):   # CAN_STAY
        '''Sets the fit parameter errors. If parameter_values=`None`, sets the
        error to 10% of the parameter value.'''

        if parameter_errors is None:  # set to 0.1% of the parameter value
            if not self.current_parameters is None:
                self.parameter_errors = [max(0.1, 0.1 * par)
                                         for par in self.current_parameters]
            else:
                raise Exception("Cannot set parameter errors. No errors \
                                provided and no parameters initialized.")
        elif len(parameter_errors) != len(self.current_parameters):
            raise Exception("Cannot set parameter errors. \
                            Tuple length mismatch.")
        else:
            self.parameter_errors = parameter_errors

        if update_iminuit:
            self.update_parameter_data()



    # Get methods
    ##############

    def get_error_matrix(self, correlation=False):  # VIEWED TODO
        '''Retrieves the parameter error matrix from iminuit.

        correlation : boolean (optional, default ``False``)
            If ``True``, return correlation matrix, else return
            covariance matrix.

        return : `numpy.matrix`
        '''

        # get parameter covariance matrix from iminuit

        # FIX_UPSTREAM we need skip_fixed=False, but this is unsupported
        #_mat = self.__iminuit.matrix(correlation, skip_fixed=False)

        # ... so use skip_fixed=False instead and fill in the gaps
        _mat = self.__iminuit.matrix(correlation, skip_fixed=True)
        _mat = np.asmatrix(_mat)  # reshape into numpy matrix
        _mat = self._fill_in_zeroes_for_fixed(_mat)  # fill in fixed par 'gaps'

        return _mat

    def get_parameter_values(self):
        '''Retrieves the parameter values from iminuit.

        return : tuple
            Current `Minuit` parameter values
        '''
        if not self.__iminuit.is_clean_state():
            # if the fit has been performed at least once
            _param_struct = self.__iminuit.get_param_states()

            return tuple([p.value for p in _param_struct])
        else:
            # need to hack to get initial parameter values
            _v = self.__iminuit.values
            _pvals = [_v[pname] for pname in _v]
            return tuple(_pvals)

    def get_parameter_errors(self):
        '''Retrieves the parameter errors from iminuit.

        return : tuple
            Current `Minuit` parameter errors
        '''
        if not self.__iminuit.is_clean_state():
            # if the fit has been performed at least once
            _param_struct = self.__iminuit.get_param_states()

            return tuple([p.error for p in _param_struct])
        else:
            # need to hack to get initial parameter values
            _e = self.__iminuit.errors
            _perrs = [_e[pname] for pname in _e]
            return tuple(_perrs)


    def get_parameter_info(self):
        '''Retrieves parameter information from iminuit.

        return : list of tuples
            ``(parameter_name, parameter_val, parameter_error)``
        '''

        if not self.__iminuit.is_clean_state():
            # if the fit has been performed at least once
            _param_struct = self.__iminuit.get_param_states()
            return tuple([(p.name, p.value, p.error * (not p.is_fixed)) for p in _param_struct])
        else:
            # need to hack to get initial parameter info
            _v, _e = self.__iminuit.values, self.__iminuit.errors
            _pnames = [pname for pname in _v]
            _pvals = [_v[pname] for pname in _v]
            _perrs = [_e[pname] * (not self.__iminuit.is_fixed(pname)) for pname in _e]
            return tuple(zip(_pnames, _pvals, _perrs))


    def get_parameter_name(self, parameter_nr):
        '''Gets the name of parameter number ``parameter_nr``

        **parameter_nr** : int
            Number of the parameter whose name to get.
        '''

        return self.parameter_names[parameter_nr]

    def get_fit_info(self, info):
        '''Retrieves other info from `Minuit`.

        **info** : string
            Information about the fit to retrieve.
            This can be any of the following:

              - ``'fcn'``: `FCN` value at minimum,
              - ``'edm'``: estimated distance to minimum
              - ``'err_def'``: `Minuit` error matrix status code
              - ``'status_code'``: `Minuit` general status code

        '''

        _fmin = self.__iminuit.get_fmin()
        if info == 'fcn':
            return _fmin.fval

        elif info == 'edm':
            return _fmin.edm

        elif info == 'err_def':
            return _fmin.up

        elif info == 'status_code':
            if _fmin.has_covariance:
                if _fmin.has_made_posdef_covar:
                    return D_MATRIX_ERROR[2]
                elif _fmin.has_accurate_covar:
                    return D_MATRIX_ERROR[3]
                else:
                    return D_MATRIX_ERROR[1]
            else:
                return D_MATRIX_ERROR[0]

    def get_chi2_probability(self, n_deg_of_freedom):
        '''
        Returns the probability that an observed :math:`\chi^2` exceeds
        the calculated value of :math:`\chi^2` for this fit by chance,
        even for a correct model. In other words, returns the probability that
        a worse fit of the model to the data exists. If this is a small value
        (typically <5%), this means the fit is pretty bad. For values below
        this threshold, the model very probably does not fit the data.

        n_def_of_freedom : int
            The number of degrees of freedom. This is typically
            :math:`n_\text{datapoints} - n_\text{parameters}`.
        '''
        _fval = self.__iminuit.get_fmin().fval
        _ndf = n_deg_of_freedom

        # return value corresponds to ROOT.TMath.Prob(chi2, ndf)
        return 1. - stats.chi2.cdf(_fval, _ndf)

    def get_contour(self, parameter1, parameter2, n_points=21):
        '''
        Returns a list of points (2-tuples) representing a sampling of
        the :math:`1\\sigma` contour of the iminuit fit. The ``FCN`` has
        to be minimized before calling this.

        **parameter1** : int
            ID of the parameter to be displayed on the `x`-axis.

        **parameter2** : int
            ID of the parameter to be displayed on the `y`-axis.

        *n_points* : int (optional)
            number of points used to draw the contour. Default is 21.

        *returns* : 2-tuple of tuples
            a 2-tuple (x, y) containing ``n_points+1`` points sampled
            along the contour. The first point is repeated at the end
            of the list to generate a closed contour.
        '''

        self.out_file.write('\n')
        # entry in log-file
        self.out_file.write('\n')
        self.out_file.write('#'*(5+28))
        self.out_file.write('\n')
        self.out_file.write('# Contour for parameters %2d, %2d #\n'\
                            %(parameter1, parameter2) )
        self.out_file.write('#'*(5+28))
        self.out_file.write('\n\n')
        self.out_file.flush()

        # first, make sure we are at minimum
        self.minimize(final_fit=True, log_print_level=0)

        # get the parameter names
        if isinstance(parameter1, int):
            parameter1 = self.parameter_names[parameter1]
        if isinstance(parameter2, int):
            parameter2 = self.parameter_names[parameter2]

        with redirect_stdout_to(self.out_file):
            # get the contour (sigma=1) -> [[x1, y1], [x2, y2], ...]
            _, _, g = self.__iminuit.mncontour(parameter1, parameter2, n_points)

        g = np.asarray(g)

        x, y = g.T  # transpose to get x and y arrays
        #
        return (x, y)

    def get_profile(self, parameter, n_points=21):
        '''
        Returns a list of points (2-tuples) the profile
        the :math:`\\chi^2`  of the iminuit fit.


        **parid** : int
            ID of the parameter to be displayed on the `x`-axis.

        *n_points* : int (optional)
            number of points used for profile. Default is 21.

        *returns* : two arrays, par. values and corresp. :math:`\\chi^2`
            containing ``n_points`` sampled profile points.
        '''

        if isinstance(parameter, int):
            par_id = parameter
            parameter = self.parameter_names[parameter]
        else:
            try:
                par_id = self.parameter_names.index(parameter)
            except ValueError:
                raise ValueError("No parameter named '%s'" % (parameter,))

        self.out_file.write('\n')
        # entry in log-file
        self.out_file.write('\n')
        self.out_file.write('#'*(2+26))
        self.out_file.write('\n')
        self.out_file.write("# Profile for parameter %2d #\n" % (par_id))
        self.out_file.write('#'*(2+26))
        self.out_file.write('\n\n')
        self.out_file.flush()

        pv=[]
        chi2=[]
        _old_print_level = self.print_level
        self.__iminuit.set_print_level(0)   # suppress output

        # first, make sure we are at minimum, i.e. re-minimize
        self.minimize(final_fit=True, log_print_level=0)
        # get parameter name and id
        if isinstance(parameter, int):
            par_id = parameter
            parameter = self.parameter_names[parameter]
        else:
            try:
                par_id = self.parameter_names.index(parameter)
            except ValueError:
                raise ValueError("No parameter named '%s'" % (parameter,))

        with redirect_stdout_to(self.out_file):
            # Get profile using iminuit.Minuit.mnprofile
            binc, vals, _ = self.__iminuit.mnprofile(parameter, n_points, 3.)

        return binc, vals


    # Other methods
    ################

    def fix_parameter(self, parameter):
        '''
        Fix parameter <`parameter`>.

        **parameter** : string
            Name of the parameter to fix.
        '''

        # cannot do this directly in iminuit
        # must create new minimizer with fixed parameters

        if isinstance(parameter, int):
            par_id = parameter
            parameter = self.parameter_names[parameter]
        else:
            try:
                par_id = self.parameter_names.index(parameter)
            except ValueError:
                raise ValueError("No parameter named '%s'" % (parameter,))

        logger.info("Fixing parameter %d in Minuit" % (par_id,))

        fitparam = self.__iminuit.fitarg.copy()   # copy minimizer arguments
        #fitparam[parameter] = v
        fitparam['fix_%s'%parameter] = True     # set fix-flag for parameter
        # replace minimizer
        ##del self.__iminuit
        self.__iminuit = iminuit.Minuit(
            self.function_to_minimize,
            print_level=self.print_level,
            forced_parameters=self.parameter_names,
            errordef = self.errordef,
            **fitparam)


    def release_parameter(self, parameter):
        '''
        Release parameter <`parameter`>.

        **parameter** : string
            Name of the parameter to release.
        '''

        if isinstance(parameter, int):
            par_id = parameter
            parameter = self.parameter_names[parameter]
        else:
            try:
                par_id = self.parameter_names.index(parameter)
            except ValueError:
                raise ValueError("No parameter named '%s'" % (parameter,))

        logger.info("Releasing parameter %d in Minuit" % (par_id,))

        fitparam = self.__iminuit.fitarg.copy()   # copy minimizer arguments
        #fitparam[parameter] = v
        fitparam['fix_%s'%parameter] = False     # set fix-flag for parameter
        # replace minimizer
        ##del self.__iminuit
        self.__iminuit = iminuit.Minuit(
            self.function_to_minimize,
            print_level=self.print_level,
            forced_parameters=self.parameter_names,
            **fitparam)



    def reset(self):
        '''Resets iminuit by re-creating the minimizer.'''
        fitparam = self.__iminuit.fitarg.copy()   # copy minimizer arguments
        # replace minimizer
        ##del self.__iminuit
        self.__iminuit = iminuit.Minuit(
            self.function_to_minimize,
            print_level=self.print_level,
            forced_parameters=self.parameter_names,
            **fitparam)

    def FCN_wrapper(self, **kw_parameters):
        '''
        This wrapper converts from the "keyword argument" way of calling the
        function to a "positional argument" way, taking into account the order
        of the parameters as they appear in `self.parameter_names`.

        This mapping is done for each call, so it's quite resource intensive,
        but this is unavoidable, since external FCNs to minimize expect
        positional arguments.

        **kw_parameters** : dict
            Map of parameter name to parameter value.
        '''

        # translate keyword arguments to positional arguments
        parameter_list = [kw_parameters[name] for name in self.parameter_names]

        # call the positional FCN.
        return self.function_to_minimize(*parameter_list)

    def minimize(self, final_fit=True, log_print_level=2):
        '''Do the minimization. This calls `Minuit`'s algorithms ``MIGRAD``
        for minimization and, if `final_fit` is `True`, also ``HESSE``
        for computing/checking the parameter error matrix.'''

        # Run minimization algorithm (MIGRAD + HESSE)
        error_code = 0

        prefix = "Minuit run on"  # set the timestamp prefix

        # insert timestamp
        self.out_file.write('\n')
        self.out_file.write('#'*(len(prefix)+4+20))
        self.out_file.write('\n')
        self.out_file.write("# %s " % (prefix,) +
                            strftime("%Y-%m-%d %H:%M:%S #\n", gmtime()))
        self.out_file.write('#'*(len(prefix)+4+20))
        self.out_file.write('\n\n')
        self.out_file.flush()

        # redirect stdout stream
        _redirection_target = None
        _redirection_target = self.out_file

        with redirect_stdout_to(_redirection_target):
            self.__iminuit.set_print_level(log_print_level)  # set iminuit print level
            logger.debug("Running MIGRAD")

            self.__iminuit.migrad(ncall=self.max_iterations) ##FUTURE: precision=?self.tolerance?)

            if final_fit:
                logger.debug("Running HESSE")
                self.__iminuit.hesse()
            # return to normal print level
            self.__iminuit.set_print_level(self.print_level)


    def minos_errors(self, log_print_level=1):
        '''
           Get (asymmetric) parameter uncertainties from MINOS
           algorithm. This calls `Minuit`'s algorithms ``MINOS``,
           which determines parameter uncertainties using profiling
           of the chi2 function.

           returns : tuple
             A tuple of (err+, err-, parabolic error, global correlation)
        '''

        # redirect stdout stream
        _redirection_target = self.out_file

        with redirect_stdout_to(_redirection_target):
            self.__iminuit.set_print_level(log_print_level)
            logger.debug("Running MINOS")
            _results = self.__iminuit.minos(maxcall=self.max_iterations)

            # return to normal print level
            self.__iminuit.set_print_level(self.print_level)

        output = []

        for par_id, parameter in enumerate(self.parameter_names):
            if parameter in _results.keys():
                _minstruct = _results[parameter]
                # positive, negative parameter error
                errpos, errneg = _minstruct.upper, _minstruct.lower
                # parabolic error, global correlation coefficient
                err, gcor = self.__iminuit.errors[parameter], self.__iminuit.gcc[parameter]
            else:
                # fixed parameters don't show up -> return zero errors
                errpos, errneg = 0, 0
                err, gcor = 0, 0
            output.append([float(errpos),float(errneg),float(err),float(gcor)])

        return output

    def _fill_in_zeroes_for_fixed(self, submatrix):
        '''
        Takes the partial error matrix (submatrix) and adds
        rows and columns with 0.0 where the fixed
        parameters should go.
        '''
        _mat = submatrix

        _fparams = self.__iminuit.list_of_fixed_param()
        _fparam_ids = [self.parameter_names.index(k) for k in _fparams]
        for _id in _fparam_ids:
            _mat = np.insert(np.insert(_mat, _id, 0., axis=0), _id, 0., axis=1)

        return _mat
