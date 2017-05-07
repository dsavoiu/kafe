'''
.. module:: minuit
   :platform: Unix
   :synopsis: A submodule providing the `Minuit` object, which communicates
        with CERN *ROOT*'s function minimizer *Minuit*.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>
'''

# ----------------------------------------------------------------
# Changes:
#  06-Aug-14  G.Q.  default parameter errors set to 10% (not 0.1%)
#  10-Aug-14  G.Q.  minimize(): allowed for initial fits w.o. HESSE
#  08-Dec-14  G.Q.  added execution of MINOS for final fit
#  09-Dec-14  G.Q.  added chi2 profiling (function get_profile)
#  08-Oct-16  GQ  printout level -1 if "quiet" specified
#                 suppressed du2() if no printout requested
# ----------------------------------------------------------------

# ROOT's data types needed to use TMinuit:
from ROOT import TMinuit, Double, Long
from ROOT import TMath  # for uning ROOT's chi2prob function
from array import array as arr  # array needed for TMinuit arguments

from .config import M_MAX_ITERATIONS, M_TOLERANCE, log_file, null_file
from .stream import redirect_stdout_to
from time import gmtime, strftime

import numpy as np

import sys
import os

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')

# Constants
############

# And define some constants to pass to TMinuit functions
P_DETAIL_LEVEL = 1
"""default level of detail for TMinuit's output
(typical range: -1 to 3, default: 1)"""

# dictionary lookup for error codes
D_MATRIX_ERROR = {0: "Error matrix not calculated",
                  1: "Error matrix approximate!",
                  2: "Error matrix forced positive definite!",
                  3: "Error matrix accurate"}  #: Error matrix status codes


class Minuit:
    '''
    A class for communicating with ROOT's function minimizer tool Minuit.
    '''

    def __init__(self, number_of_parameters, function_to_minimize,
                 parameter_names, start_parameters, parameter_errors,
                 quiet=True, verbose=False):
        '''
        Create a Minuit minimizer for a function `function_to_minimize`.
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
            If ``True``, suppresses all output from ``TMinuit``.

        *verbose* : boolean (optional, default: ``False``)
            If ``True``, sets ``TMinuit``'s print level to a high value, so
            that all output is logged.

        '''
        #: the name of this minimizer type
        self.name = "ROOT::TMinuit"

        #: the actual `FCN` called in ``FCN_wrapper``
        self.function_to_minimize = function_to_minimize

        #: number of parameters to minimize for
        self.number_of_parameters = number_of_parameters

        if not quiet:
            self.out_file = open(log_file("minuit.log"), 'a')
        else:
            self.out_file = null_file()

        # create a TMinuit instance for that number of parameters
        self.__gMinuit = TMinuit(self.number_of_parameters)

        # instruct Minuit to use this class's FCN_wrapper method as a FCN
        self.__gMinuit.SetFCN(self.FCN_wrapper)

        # set print level according to flag
        if quiet:
            self.set_print_level(-1000)  # suppress output
        elif verbose:
            self.set_print_level(10)     # detailed output
        else:
            self.set_print_level(0)      # frugal output

        # initialize minimizer
        self.set_err()
        self.set_strategy()
        self.set_parameter_values(start_parameters)
        self.set_parameter_errors(parameter_errors)
        self.set_parameter_names(parameter_names)

        #: maximum number of iterations until ``TMinuit`` gives up
        self.max_iterations = M_MAX_ITERATIONS

        #: ``TMinuit`` tolerance
        self.tolerance = M_TOLERANCE

    def update_parameter_data(self, show_warnings=False):
        """
        (Re-)Sets the parameter names, values and step size on the
        C++ side of Minuit.
        """
        error_code = Long(0)
        try:
            # Set up the starting fit parameters in TMinuit
            for i in range(0, self.number_of_parameters):
                self.__gMinuit.mnparm(i, self.parameter_names[i],
                                      self.current_parameters[i],
                                      0.1 * self.parameter_errors[i],
                                      0, 0, error_code)
                # use 10% of the par. 1-sigma errors as the initial step size
        except AttributeError as e:
            if show_warnings:
                logger.warn("Cannot update Minuit data on the C++ side. "
                            "AttributeError: %s" % (e, ))
        return error_code

    # Set methods
    ##############

    def set_print_level(self, print_level=P_DETAIL_LEVEL):
        '''Sets the print level for Minuit.

        *print_level* : int (optional, default: 1 (frugal output))
            Tells ``TMinuit`` how much output to generate. The higher this
            value, the more output it generates.
        '''
        self.__gMinuit.SetPrintLevel(print_level)  # set Minuit print level
        self.print_level = print_level

    def set_strategy(self, strategy_id=1):
        '''Sets the strategy Minuit.

        *strategy_id* : int (optional, default: 1 (optimized))
            Tells ``TMinuit`` to use a certain strategy. Refer to ``TMinuit``'s
            documentation for available strategies.
        '''
        error_code = Long(0)
        # execute SET STRATEGY command
        self.__gMinuit.mnexcm("SET STRATEGY",
                              arr('d', [strategy_id]), 1, error_code)

    def set_err(self, up_value=1.0):
        '''Sets the ``UP`` value for Minuit.

        *up_value* : float (optional, default: 1.0)
            This is the value by which `FCN` is expected to change.
        '''
        # Tell TMinuit to use an up-value of 1.0
        error_code = Long(0)
        # execute SET ERR command
        self.__gMinuit.mnexcm("SET ERR", arr('d', [up_value]), 1, error_code)

    def set_parameter_values(self, parameter_values):
        '''
        Sets the fit parameters. If parameter_values=`None`, tries to infer
          defaults from the function_to_minimize.
        '''
        if len(parameter_values) == self.number_of_parameters:
            self.current_parameters = parameter_values
        else:
            raise Exception("Cannot get default parameter values from the \
            FCN. Not all parameters have default values given.")

        self.update_parameter_data()

    def set_parameter_names(self, parameter_names):
        '''Sets the fit parameters. If parameter_values=`None`, tries to infer
        defaults from the function_to_minimize.'''
        if len(parameter_names) == self.number_of_parameters:
            self.parameter_names = parameter_names
        else:
            raise Exception("Cannot set param names. Tuple length mismatch.")

        self.update_parameter_data()

    def set_parameter_errors(self, parameter_errors=None):
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

        self.update_parameter_data()

    # Get methods
    ##############

    def get_error_matrix(self):
        '''Retrieves the parameter error matrix from TMinuit.

        return : `numpy.matrix`
        '''

        # set up an array of type `double' to pass to TMinuit
        tmp_array = arr('d', [0.0]*(self.number_of_parameters**2))
        # get parameter covariance matrix from TMinuit
        self.__gMinuit.mnemat(tmp_array, self.number_of_parameters)
        # reshape into 2D array
        return np.asmatrix(
            np.reshape(
                tmp_array,
                (self.number_of_parameters, self.number_of_parameters)
            )
        )

    def get_parameter_values(self):
        '''Retrieves the parameter values from TMinuit.

        return : tuple
            Current `Minuit` parameter values
        '''

        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)

        for i in range(0, self.number_of_parameters):
            self.__gMinuit.GetParameter(i, p, pe)  # retrieve fitresult

            result.append(float(p))

        return tuple(result)

    def get_parameter_errors(self):
        '''Retrieves the parameter errors from TMinuit.

        return : tuple
            Current `Minuit` parameter errors
        '''

        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)

        for i in range(0, self.number_of_parameters):
            self.__gMinuit.GetParameter(i, p, pe)  # retrieve fitresult

            result.append(float(pe))

        return tuple(result)

    def get_parameter_info(self):
        '''Retrieves parameter information from TMinuit.

        return : list of tuples
            ``(parameter_name, parameter_val, parameter_error)``
        '''

        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)

        for i in range(0, self.number_of_parameters):
            self.__gMinuit.GetParameter(i, p, pe)  # retrieve fitresult
            result.append((self.get_parameter_name(i), float(p), float(pe)))

        return result

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

        # declare vars in which to retrieve other info
        fcn_at_min = Double(0)
        edm = Double(0)
        err_def = Double(0)
        n_var_param = Long(0)
        n_tot_param = Long(0)
        status_code = Long(0)

        # Tell TMinuit to update the variables declared above
        self.__gMinuit.mnstat(fcn_at_min,
                              edm,
                              err_def,
                              n_var_param,
                              n_tot_param,
                              status_code)

        if info == 'fcn':
            return fcn_at_min

        elif info == 'edm':
            return edm

        elif info == 'err_def':
            return err_def

        elif info == 'status_code':
            try:
                return D_MATRIX_ERROR[status_code]
            except:
                return status_code

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
        chi2 = Double(self.get_fit_info('fcn'))
        ndf = Long(n_deg_of_freedom)
        return TMath.Prob(chi2, ndf)

    def get_contour(self, parameter1, parameter2, n_points=21):
        '''
        Returns a list of points (2-tuples) representing a sampling of
        the :math:`1\\sigma` contour of the TMinuit fit. The ``FCN`` has
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
#
# first, make sure we are at minimum
        self.minimize(final_fit=True, log_print_level=0)

        # get the TGraph object from ROOT
        g = self.__gMinuit.Contour(n_points, parameter1, parameter2)

        # extract point data into buffers
        xbuf, ybuf = g.GetX(), g.GetY()
        N = g.GetN()

        # generate tuples from buffers
        x = np.frombuffer(xbuf, dtype=float, count=N)
        y = np.frombuffer(ybuf, dtype=float, count=N)

        #
        return (x, y)

    def get_profile(self, parid, n_points=21):
        '''
        Returns a list of points (2-tuples) the profile
        the :math:`\\chi^2`  of the TMinuit fit.


        **parid** : int
            ID of the parameter to be displayed on the `x`-axis.

        *n_points* : int (optional)
            number of points used for profile. Default is 21.

        *returns* : two arrays, par. values and corresp. :math:`\\chi^2`
            containing ``n_points`` sampled profile points.
        '''

        self.out_file.write('\n')
        # entry in log-file
        self.out_file.write('\n')
        self.out_file.write('#'*(2+26))
        self.out_file.write('\n')
        self.out_file.write("# Profile for parameter %2d #\n" % (parid))
        self.out_file.write('#'*(2+26))
        self.out_file.write('\n\n')
        self.out_file.flush()

        # redirect stdout stream
        _redirection_target = None
        ## -- disable redirection completely, for now
        ##if log_print_level >= 0:
        ##    _redirection_target = self.out_file

        with redirect_stdout_to(_redirection_target):
            pv = []
            chi2 = []
            error_code = Long(0)
            self.__gMinuit.mnexcm("SET PRINT",
                     arr('d', [0.0]), 1, error_code)  # no printout

            # first, make sure we are at minimum, i.e. re-minimize
            self.minimize(final_fit=True, log_print_level=0)
            minuit_id = Double(parid + 1) # Minuit parameter numbers start with 1

            # retrieve information about parameter with id=parid
            pmin = Double(0)
            perr = Double(0)
            self.__gMinuit.GetParameter(parid, pmin, perr)  # retrieve fitresult

            # fix parameter parid ...
            self.__gMinuit.mnexcm("FIX",
                                    arr('d', [minuit_id]),
                                    1, error_code)
            # ... and scan parameter values, minimizing at each point
            for v in np.linspace(pmin - 3.*perr, pmin + 3.*perr, n_points):
                pv.append(v)
                self.__gMinuit.mnexcm("SET PAR",
                     arr('d', [minuit_id, Double(v)]),
                                   2, error_code)
                self.__gMinuit.mnexcm("MIGRAD",
                     arr('d', [self.max_iterations, self.tolerance]),
                                   2, error_code)
                chi2.append(self.get_fit_info('fcn'))

            # release parameter to back to initial value and release
            self.__gMinuit.mnexcm("SET PAR",
                                  arr('d', [minuit_id, Double(pmin)]),
                                   2, error_code)
            self.__gMinuit.mnexcm("RELEASE",
                                    arr('d', [minuit_id]),
                                    1, error_code)

        return pv, chi2


    # Other methods
    ################

    def fix_parameter(self, parameter_number):
        '''
        Fix parameter number <`parameter_number`>.

        **parameter_number** : int
            Number of the parameter to fix.
        '''
        error_code = Long(0)
        logger.info("Fixing parameter %d in Minuit" % (parameter_number,))
        # execute FIX command
        self.__gMinuit.mnexcm("FIX",
                              arr('d', [parameter_number+1]), 1, error_code)

    def release_parameter(self, parameter_number):
        '''
        Release parameter number <`parameter_number`>.

        **parameter_number** : int
            Number of the parameter to release.
        '''
        error_code = Long(0)
        logger.info("Releasing parameter %d in Minuit" % (parameter_number,))
        # execute RELEASE command
        self.__gMinuit.mnexcm("RELEASE",
                              arr('d', [parameter_number+1]), 1, error_code)

    def reset(self):
        '''Execute TMinuit's `mnrset` method.'''
        self.__gMinuit.mnrset(0)  # reset TMinuit

    def FCN_wrapper(self, number_of_parameters, derivatives,
                    f, parameters, internal_flag):
        '''
        This is actually a function called in *ROOT* and acting as a C wrapper
        for our `FCN`, which is implemented in Python.

        This function is called by `Minuit` several times during a fit. It
        doesn't return anything but modifies one of its arguments (*f*).
        This is *ugly*, but it's how *ROOT*'s ``TMinuit`` works. Its argument
        structure is fixed and determined by `Minuit`:

        **number_of_parameters** : int
            The number of parameters of the current fit

        **derivatives** : C array
            If the user chooses to calculate the first derivative of the
            function inside the `FCN`, this value should be written here. This
            interface to `Minuit` ignores this derivative, however, so
            calculating this inside the `FCN` has no effect (yet).

        **f** : C array
            The desired function value is in f[0] after execution.

        **parameters** : C array
            A C array of parameters. Is cast to a Python list

        **internal_flag** : int
            A flag allowing for different behaviour of the function.
            Can be any integer from 1 (initial run) to 4(normal run). See
            `Minuit`'s specification.
        '''

        # Retrieve the parameters from the C side of ROOT and
        # store them in a Python list -- resource-intensive
        # for many calls, but can't be improved (yet?)
        parameter_list = np.frombuffer(parameters, dtype=float,
                                       count=self.number_of_parameters)

        # call the Python implementation of FCN.
        f[0] = self.function_to_minimize(*parameter_list)

    def minimize(self, final_fit=True, log_print_level=2):
        '''Do the minimization. This calls `Minuit`'s algorithms ``MIGRAD``
        for minimization and, if `final_fit` is `True`, also ``HESSE``
        for computing/checking the parameter error matrix.'''

        # Set the FCN again. This HAS to be done EVERY
        # time the minimize method is called because of
        # the implementation of SetFCN, which is not
        # object-oriented but sets a global pointer!!!
        logger.debug("Updating current FCN")
        self.__gMinuit.SetFCN(self.FCN_wrapper)

        # Run minimization algorithm (MIGRAD + HESSE)
        error_code = Long(0)

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
        if log_print_level >= 0:
            _redirection_target = self.out_file

        with redirect_stdout_to(_redirection_target):
            self.__gMinuit.SetPrintLevel(log_print_level)  # set Minuit print level
            logger.debug("Running MIGRAD")
            self.__gMinuit.mnexcm("MIGRAD",
                                  arr('d', [self.max_iterations, self.tolerance]),
                                  2, error_code)
            if(final_fit):
                logger.debug("Running HESSE")
                self.__gMinuit.mnexcm("HESSE", arr('d', [self.max_iterations]), 1, error_code)
            # return to normal print level
            self.__gMinuit.SetPrintLevel(self.print_level)


    def minos_errors(self, log_print_level=1):
        '''
           Get (asymmetric) parameter uncertainties from MINOS
           algorithm. This calls `Minuit`'s algorithms ``MINOS``,
           which determines parameter uncertainties using profiling
           of the chi2 function.

           returns : tuple
             A tuple of [err+, err-, parabolic error, global correlation]
        '''

        # Set the FCN again. This HAS to be done EVERY
        # time the minimize method is called because of
        # the implementation of SetFCN, which is not
        # object-oriented but sets a global pointer!!!
        logger.debug("Updating current FCN")
        self.__gMinuit.SetFCN(self.FCN_wrapper)

        # redirect stdout stream
        _redirection_target = None
        if log_print_level >= 0:
            _redirection_target = self.out_file

        with redirect_stdout_to(_redirection_target):
            self.__gMinuit.SetPrintLevel(log_print_level)
            logger.debug("Running MINOS")
            error_code = Long(0)
            self.__gMinuit.mnexcm("MINOS", arr('d', [self.max_iterations]), 1, error_code)

            # return to normal print level
            self.__gMinuit.SetPrintLevel(self.print_level)


        output = []
        errpos=Double(0) # positive parameter error
        errneg=Double(0) # negative parameter error
        err=Double(0)    # parabolic error
        gcor=Double(0)   # global correlation coefficient

        for i in range(0, self.number_of_parameters):
            self.__gMinuit.mnerrs(i, errpos, errneg, err, gcor)
            output.append([float(errpos),float(errneg),float(err),float(gcor)])

        return output
