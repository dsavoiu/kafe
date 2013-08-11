'''
.. module:: minuit
   :platform: Unix
   :synopsis: A submodule providing the `Minuit` object, which communicates with CERN *ROOT*'s function minimizer *Minuit*.
   
.. moduleauthor:: Daniel Savoiu <daniel.savoiu@ekp.kit.edu>
'''

from ROOT import gROOT, TMinuit, Double, Long # ROOT's data types needed to use TMinuit
from ROOT import TMath  # for uning ROOT's chi2prob function
from array import array as arr # array needed for TMinuit arguments
from constants import *

import numpy as np

# Constants
############

# And define some constants to pass to TMinuit functions
P_DETAIL_LEVEL = 1          # default level of detail for TMinuit's output (typical range: -1 to 3, default: 1)


# dictionary lookup for error codes
D_MATRIX_ERROR = {0 : "Error matrix not calculated",
                  1 : "Error matrix approximate!",
                  2 : "Error matrix forced positive definite!",
                  3 : "Error matrix accurate" } #: Error matrix status codes

class Minuit:
    '''
    A class for communicating with ROOT's function minimizer tool Minuit.
    '''

    def __init__(self, number_of_parameters, function_to_minimize, par_names, start_params, param_errors, quiet=True, verbose=False): # number_of_parameters, 
        '''
        Create a Minuit minimizer for a function `function_to_minimize`. Necessary arguments are [the 
        number of parameters and] the function to be minimized `function_to_minimize`. The function
        `function_to_minimize`'s arguments must be numerical values. The same goes for its output.
        
        Another requirement is for every parameter of `function_to_minimize` to have a default value.
        These are then used to initialize Minuit. 
        
        **number_of_parameters** : int 
            The number of parameters of the function to minimize.
        
        **function_to_minimize** : function
            The function which `Minuit` should minimize. This must be a Python function with <``number_of_parameters``> arguments.
        
        **par_names** : tuple/list of strings
            The parameter names. These are used to keep track of the parameters in `Minuit`'s output.
            
        **start_params** : tuple/list of floats
            The start values of the parameters. It is important to have a good, if rough, estimate of
            the parameters at the minimum before starting the minimization. Wrong initial parameters can
            yield a local minimum instead of a global one.
            
        **param_errors** : tuple/list of floats
            An initial guess of the parameter errors. These errors are used to define the initial step size.
            
        *quiet* : boolean (optional, default: ``True``)
            If ``True``, suppresses all output from ``TMinuit``.
            
        *verbose* : boolean (optional, default: ``False``)
            If ``True``, sets ``TMinuit``'s print level to a high value, so that all output is logged.
            
        '''
        
        self.function_to_minimize = function_to_minimize      #: the actual `FCN` called in ``FCN_wrapper``
        
        self.number_of_parameters = number_of_parameters      #: number of parameters to minimize for
        
        self.__gMinuit = TMinuit(self.number_of_parameters)   # create a TMinuit instance for that number of parameters
        self.__gMinuit.SetFCN(self.FCN_wrapper)               # instruct Minuit to use this class's FCN_wrapper method as a FCN
        
        # set print level according to flag
        if quiet:
            self.set_print_level(-1000) # suppress output
        elif verbose:
            self.set_print_level(10)    # detailed output
        else:
            self.set_print_level(0)     # frugal output
            
        self.set_err()
        self.set_strategy()
        self.set_parameter_values(start_params)                                                             # set the current parameter values
        self.set_parameter_errors(param_errors)                                                             # set the current parameter errors
        #self.parameter_names = self.function_to_minimize.func_code.co_varnames[:function_to_minimize.func_code.co_argcount]      # extract the parameter names      from the FCN
        self.set_parameter_names(par_names)
        
        error_code = Long(0)
        # Set up the starting fit parameters in TMinuit
        for i in range(0, self.number_of_parameters):
            self.__gMinuit.mnparm(i, self.parameter_names[i], self.current_parameters[i], 0.1 * self.parameter_errors[i], 0, 0, error_code)
            # use 10% of the parameter 1-sigma errors as the initial step size
        
        self.max_iterations = M_MAX_ITERATIONS  #: maximum number of iterations until ``TMinuit`` gives up
        self.tolerance = M_TOLERANCE        #: ``TMinuit`` tolerance

    # Set methods
    ##############
    
    def set_print_level(self, print_level=P_DETAIL_LEVEL):
        '''Sets the print level for Minuit.
        
        *print_level* : int (optional, default: 1 (frugal output))
            Tells ``TMinuit`` how much output to generate. The higher this value, the
            more output it generates.
        '''
        self.__gMinuit.SetPrintLevel(print_level) # set Minuit print level
        
    def set_strategy(self, strategy_id=1):
        '''Sets the strategy Minuit.
        
        *strategy_id* : int (optional, default: 1 (optimized))
            Tells ``TMinuit`` to use a certain strategy. Refer to ``TMinuit``'s
            documentation for available strategies.
        '''
        error_code = Long(0)
        self.__gMinuit.mnexcm("SET STRATEGY", arr('d', [strategy_id]), 1, error_code) # execute SET STRATEGY command
    
    def set_err(self, up_value=1.0):
        '''Sets the ``UP`` value for Minuit.
        
        *up_value* : float (optional, default: 1.0)
            This is the value by which `FCN` is expected to change.
        '''
        # Tell TMinuit to use an up-value of 1.0
        error_code = Long(0)
        self.__gMinuit.mnexcm("SET ERR", arr('d', [up_value]), 1, error_code) # execute SET ERR command
    
        
    def set_parameter_values(self, param_values): #=None):
        '''Sets the fit parameters. If param_values=`None`, tries to infer defaults from the function_to_minimize.'''
        if len(param_values) == self.number_of_parameters:
            self.current_parameters = param_values
        else:
            raise Exception, "Cannot get default parameter values from the FCN. Not all parameters have default values given."
        
    def set_parameter_names(self, param_names):
        '''Sets the fit parameters. If param_values=`None`, tries to infer defaults from the function_to_minimize.'''
        if len(param_names) == self.number_of_parameters:
            self.parameter_names = param_names
        else:
            raise Exception, "Cannot set param names. Tuple length mismatch."    
    
    def set_parameter_errors(self, param_errors=None):
        '''Sets the fit parameter errors. If param_values=`None`, sets the error to 1% of the parameter value.'''
    
        if param_errors is None: # set to 1% of the parameter value
            if not self.current_parameters is None:
                self.parameter_errors = [max(0.01, 0.01 * par) for par in self.current_parameters]
            else:
                raise Exception, "Cannot set parameter errors. No errors provided and no parameters initialized."
        elif len(param_errors) != len(self.current_parameters):
            raise Exception, "Cannot set parameter errors. Tuple length mismatch."
        else:
            self.parameter_errors = param_errors
    
    
    # Get methods
    ##############
    
    def get_error_matrix(self):
        '''Retrieves the parameter error matrix from TMinuit.
        
        return : `numpy.matrix`
        '''
        
        tmp_array = arr('d', [0.0]*(self.number_of_parameters**2)) # setup an array of type `double' to pass to TMinuit
        self.__gMinuit.mnemat(tmp_array, self.number_of_parameters) # get parameter covariance matrix from TMinuit
        return np.asmatrix( np.reshape(tmp_array,(self.number_of_parameters, self.number_of_parameters)) ) # reshape into 2D array
    
    
    def get_parameter_values(self):
        '''Retrieves the parameter values from TMinuit.
        
        return : tuple
            Current `Minuit` parameter values
        '''
        
        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)
        
        for i in range(0, self.number_of_parameters):
            #print "minuit: %r " % self
            #print "gMinuit: %r " % self.__gMinuit
            #print "GetParameter: %r " % self.__gMinuit.GetParameter
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
            ``(param_name, param_val, param_error)``
        '''
        
        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)
        
        for i in range(0, self.number_of_parameters):
            self.__gMinuit.GetParameter(i, p, pe)  # retrieve fitresult
            
            result.append( (self.get_parameter_name(i), float(p), float(pe)) )
        
        return result

    def get_parameter_name(self, param_nr):
        '''Gets the name of parameter number ``param_nr``
        
        **param_nr** : int
            Number of the parameter whose name to get.
        '''
        #if param_values is None: # infer parameters from FCN
        #    if len(self.function_to_minimize.func_defaults) == self.number_of_parameters:    # only if all defaults given
        #        self.current_parameters = self.function_to_minimize.func_defaults   # get defaults from FCN
        #    else:
        #        raise Exception, "Cannot get default parameter values from the FCN. Not all parameters have default values given."
        return self.parameter_names[param_nr]    

    def get_fit_info(self, info):
        '''Retrieves other info from `Minuit`. 
        
        **info** : string
            Information about the fit to retrieve. This can be any of the following:
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
        
        if info=='fcn':
            return fcn_at_min
        
        elif info=='edm':
            return edm
        
        elif info=='err_def':
            return err_def
        
        elif info=='status_code':
            try:
                return D_MATRIX_ERROR[status_code]
            except:
                return status_code
        
    def get_chi2_probability(self, n_deg_of_freedom):
        '''
        Returns the probability that an observed :math:`\chi^2` exceeds
        the calculated value of :math:`\chi^2` for this fit by chance, even for a correct model.
        In other words, returns the probability that a worse fit of the model to the data exists.
        If this is a small value (typically <5%), this means the fit is pretty bad. For
        values below this threshold, the model very probably does not fit the data.
        
        n_def_of_freedom : int
            The number of degrees of freedom. This is typically :math:`n_\text{datapoints} - n_\text{parameters}`.
        '''
        chi2 = Double(self.get_fit_info('fcn'))
        ndf = Long(n_deg_of_freedom)
        return TMath.Prob(chi2, ndf)
        
    def get_contour(self, parameter1, parameter2, n_points=20):
        '''
        Returns a list of points (2-tuples) representing a sampling of
        the :math:`1\\sigma` contour of the TMinuit fit. The ``FCN`` has to be
        minimized before calling this.
        
        **parameter1** : int
            ID of the parameter to be displayed on the `x`-axis.
            
        **parameter2** : int
            ID of the parameter to be displayed on the `y`-axis.
            
        *n_points* : int (optional)
            number of points used to draw the contour. Default is 20.
        
        *returns* : 2-tuple of tuples
            a 2-tuple (x, y) containing ``n_points+1`` points sampled
            along the contour. The first point is repeated at the end
            of the list to generate a closed contour.
        '''
        
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
    
    # Other methods
    ################
    
    def reset(self):
        self.__gMinuit.mnrset(0) # reset TMinuit
        
    def FCN_wrapper(self, number_of_parameters, derivatives, f, parameters, internal_flag):
        '''
        This is actually a function called in *ROOT* and acting as a C wrapper 
        for our `FCN`, which is implemented in Python.
        
        This function is called by `Minuit` several times during a fit. It doesn't return
        anything but modifies one of its arguments (*f*). This is *ugly*, but it's how *ROOT*'s
        ``TMinuit`` works. Its argument structure is fixed and determined by `Minuit`:
        
        **number_of_parameters** : int
            The number of parameters of the current fit
        
        
        **derivatives** : ?? 
            Computed gradient (??)
        
        **f** : C array
            The desired function value is in f[0] after execution.
        
        **parameters** : C array
            A C array of parameters. Is cast to a Python list
        
        **internal_flag** : int
            A flag allowing for different behaviour of the function.
            Can be any integer from 1 (initial run) to 4(normal run). See `Minuit`'s specification.
        '''
        
        if internal_flag==1:    # the internal flag is 1 for the initial run (NOT TRUE -> CHECK)
            pass                # do something before the FCN is called for the first time
        
        # Retrieve the parameters from the C side of ROOT and
        # store them in a Python list -- VERY resource-intensive
        # for many calls, but can't be improved (yet?)
#         parameter_list = []
#         for j in range(self.number_of_parameters):
#             parameter_list.append(parameters[j])
        
        parameter_list = np.frombuffer(parameters, dtype=float, count=self.number_of_parameters)
        
        ##print "Calling FCN%r = %s" % (parameter_list, self.function_to_minimize(*parameter_list)) 
        f[0] = self.function_to_minimize(*parameter_list) # call the Python implementation of FCN.

    def minimize(self):
        '''Do the minimization. This calls `Minuit`'s algorithms ``MIGRAD`` for minimization
        and ``HESSE`` for computing/checking the parameter error matrix.'''
        
        # Set the FCN again. This HAS to be done EVERY
        # time the minimize method is called because of
        # the implementation of SetFCN, which is not
        # object-oriented but sets a global pointer!!!
        self.__gMinuit.SetFCN(self.FCN_wrapper)               # instruct Minuit to use this class's FCN_wrapper method as a FCN
        
        # Run minimization algorithm (MIGRAD + HESSE)
        error_code = Long(0)
        
        self.__gMinuit.mnexcm("MIGRAD", arr('d', [self.max_iterations, self.tolerance]), 2, error_code)
        self.__gMinuit.mnexcm("HESSE", arr('d', [6000]), 1, error_code) # Call HESSE with max. 6000 calls


if __name__ == "__main__":
    
    def quadratic(x=3e-24, y=4e-23):
        #return (args[0] - 3.14) ** 2 + (args[1] + 2.71) ** 2 + 1.0    # should be minimal (=1.0) at arg[0] = 3.14 and arg[1] = -2.71
        return (x - 3.14e-24) ** 2 + (y + 2.71e-23) ** 2 + 1.0    # should be minimal (=1.0) at arg[0] = 3.14 and arg[1] = -2.71
       
    def quartic(x=5, y=10):
        return (x - 2.9174) ** 4 + (y + 8.12947) ** 2 + 1.0    # should be minimal (=1.0) at arg[0] = 3.14 and arg[1] = -2.71
       
    
    myMinimizer = Minuit(2, quadratic, ('x','y'), (3e-24, 4e-23), (3e-25, 4e-25))
    myMinimizer2 = Minuit(2, quartic, ('x','y'), (5, 10), (.05, .10))
    
    myMinimizer.minimize()
    myMinimizer2.minimize()
    
    print myMinimizer.get_parameter_values()
    print myMinimizer.get_parameter_errors()
    print myMinimizer.get_error_matrix()
    
    print myMinimizer2.get_parameter_values()
    print myMinimizer2.get_parameter_errors()
    print myMinimizer2.get_error_matrix()
    
    