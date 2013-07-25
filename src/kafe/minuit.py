'''
.. module:: minuit
   :platform: Unix
   :synopsis: A submodule providing the `Minuit` object, which communicates with CERN *ROOT*'s function minimizer *Minuit*.
   
.. moduleauthor:: Daniel Savoiu <daniel.savoiu@ekp.kit.edu>
'''

from ROOT import TMinuit, Double, Long # ROOT's data types needed to use TMinuit
from ROOT import TMath
from array import array as arr # array needed for TMinuit arguments
# from copy import copy # needed to make shallow copies of objects instead of references
# from math import floor, log
# 
# 
import numpy as np

# Constants

# Set up class constants imposing general restrictions on TMinuit:
M_TOLERANCE = 0.1           # default tolerance
M_MAX_ITERATIONS = 6000     # Maximum number of TMinuit iterations until aborting the process
M_X_FIT_ITERATIONS = 2      # Number of additional iterations for M{x} fit (0 disregards M{x} errors), default = 2


# And define some constants to pass to TMinuit functions
P_DETAIL_LEVEL = 3          # default level of detail for TMinuit's output (-1 to 3)


# dictionary lookup for error codes
D_MATRIX_ERROR = {0 : "Error matrix not calculated", 1 : "Error matrix approximate!", 2 : "Error matrix forces positive definite!", 3 : "Error matrix accurate" } # Error matrix staus codes

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
        
        @param function_to_minimize: Function to minimize. Its arguments are numerical values (parameters).
        @type function_to_minimize: function
        '''
        
        self.function_to_minimize = function_to_minimize                                                         # set the actual FCN called in FCN_wrapper
        #self.number_of_parameters = self.function_to_minimize.func_code.co_argcount                              # extract the number of parameters from the FCN
        self.number_of_parameters = number_of_parameters
        
        self.__gMinuit = TMinuit(self.number_of_parameters)   # create a TMinuit instance for that number of parameters
        self.__gMinuit.SetFCN(self.FCN_wrapper)                    # instruct Minuit to use this class's FCN_wrapper method as a FCN
        
        # set print level according to flag
        if quiet:
            self.set_print_level(-1000)
        elif verbose:
            self.set_print_level(3)
        else:
            self.set_print_level(0)
            
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
        
        self.max_iterations = 6000
        self.tolerance = 0.1

    # Set methods
    ##############
    
    def set_print_level(self, print_level=3):
        '''Sets the print level for Minuit. Default: 0 (suppress all output)'''
        self.__gMinuit.SetPrintLevel(print_level) # set Minuit print level
        
    def set_strategy(self, strategy_id=1):
        '''Sets the strategy Minuit. Default: 1 (optimized)'''
        error_code = Long(0)
        self.__gMinuit.mnexcm("SET STRATEGY", arr('d', [strategy_id]), 1, error_code) # execute SET STRATEGY command
    
    def set_err(self, up_value=1.0):
        '''Sets the UP value for Minuit. Default: 1.0 (good for chi2)'''
        # Tell TMinuit to use an up-value of 1.0
        error_code = Long(0)
        self.__gMinuit.mnexcm("SET ERR", arr('d', [up_value]), 1, error_code) # execute SET ERR command
    
        
    def set_parameter_values(self, param_values): #=None):
        '''Sets the fit parameters. If param_values=`None`, tries to infer defaults from the function_to_minimize.'''
        #if param_values is None: # infer parameters from FCN
        #    if len(self.function_to_minimize.func_defaults) == self.number_of_parameters:    # only if all defaults given
        #        self.current_parameters = self.function_to_minimize.func_defaults   # get defaults from FCN
        #    else:
        #        raise Exception, "Cannot get default parameter values from the FCN. Not all parameters have default values given."
        if len(param_values) == self.number_of_parameters:
            self.current_parameters = param_values
        else:
            raise Exception, "Cannot get default parameter values from the FCN. Not all parameters have default values given."
        
    def set_parameter_names(self, param_names): #=None):
        '''Sets the fit parameters. If param_values=`None`, tries to infer defaults from the function_to_minimize.'''
        #if param_values is None: # infer parameters from FCN
        #    if len(self.function_to_minimize.func_defaults) == self.number_of_parameters:    # only if all defaults given
        #        self.current_parameters = self.function_to_minimize.func_defaults   # get defaults from FCN
        #    else:
        #        raise Exception, "Cannot get default parameter values from the FCN. Not all parameters have default values given."
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
        '''Retrieves the parameter error matrix from TMinuit'''
        
        tmp_array = arr('d', [0.0]*(self.number_of_parameters**2)) # setup an array of type `double' to pass to TMinuit
        self.__gMinuit.mnemat(tmp_array, self.number_of_parameters) # get parameter covariance matrix from TMinuit
        return np.asmatrix( np.reshape(tmp_array,(self.number_of_parameters, self.number_of_parameters)) ) # reshape into 2D array
    
    
    def get_parameter_values(self):
        '''Retrieves the parameter values from TMinuit. Returns a tuple.'''
        
        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)
        
        for i in range(0, self.number_of_parameters):
            self.__gMinuit.GetParameter(i, p, pe)  # retrieve fitresult
            
            
            result.append(float(p))
        
        return tuple(result)
    
    def get_parameter_errors(self):
        '''Retrieves the parameter errors from TMinuit. Returns a tuple.'''
        
        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)
        
        for i in range(0, self.number_of_parameters):
            self.__gMinuit.GetParameter(i, p, pe)  # retrieve fitresult
            
            
            result.append(float(pe))
        
        return tuple(result)
    
    def get_parameter_info(self):
        '''Retrieves parameter information from TMinuit. Returns a list of tuples (param_name, param_val, param_error)'''
        
        result = []
        # retrieve fit parameters
        p, pe = Double(0), Double(0)
        
        for i in range(0, self.number_of_parameters):
            self.__gMinuit.GetParameter(i, p, pe)  # retrieve fitresult
            
            
            result.append( (self.get_parameter_name(i), float(p), float(pe)) )
        
        return result

    def get_parameter_name(self, param_nr): #=None):
        '''Sets the fit parameters. If param_values=`None`, tries to infer defaults from the function_to_minimize.'''
        #if param_values is None: # infer parameters from FCN
        #    if len(self.function_to_minimize.func_defaults) == self.number_of_parameters:    # only if all defaults given
        #        self.current_parameters = self.function_to_minimize.func_defaults   # get defaults from FCN
        #    else:
        #        raise Exception, "Cannot get default parameter values from the FCN. Not all parameters have default values given."
        return self.parameter_names[param_nr]    

    def get_fit_info(self, info):
        '''Retrieves other info from Minuit. The argument `info' can be any of the following:
            - `fcn`: `FCN` value at minimum,
            - `edm`: estimated distance to minimum
            - `err_def`: `Minuit` error matrix status code
            - `status_code`: `Minuit` general status code
            
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
        Returns the probability that an observed S{chi}-square exceeds
        the calculated value of S{chi}-square for this fit by chance, even for a correct model.
        In other words, returns the probability that a worse fit of the model to the data exists.
        If this is a small value (typically M{< 5%}), this means the fit is pretty bad. For
        values below this threshold, the model very probably does not fit the data.
        '''
        chi2 = Double(self.get_fit_info('fcn'))
        ndf = Long(n_deg_of_freedom)
        return TMath.Prob(chi2, ndf)
        
    
    # Other methods
    ################
    
    def reset(self):
        self.__gMinuit.mnrset(0) # reset TMinuit
        
    def FCN_wrapper(self, number_of_parameters, derivatives, f, parameters, internal_flag):
        '''
        This is actually a function called in ROOT and acting as a C wrapper 
        for our B{FCN}, which is implemented in Python.
        
        This function is called by Minuit several times during a fit. It doesn't return
        anything but sets modifies one of its arguments (*f*). This is ugly. Its argument
        structure is fixed and determined by Minuit:
        
        @param number_of_parameters: The number of parameters of the current fit
        @type number_of_parameters: int
        
        @param derivatives: computed gradient ??
        @type derivatives: ??
        
        @param f: A (C-compatible) array. The desired function value is in f[0] after execution.
        @type f: array
        
        @param parameters: A C array of parameters. Is cast to a Python list
        @type parameters: array
        
        @param internal_flag: A flag allowing for different behaviour of the function
        @type internal_flag: any int from M{1} (initial run) to M{4} (normal run)
        '''
        
        if internal_flag==1:    # the internal flag is 1 for the initial run
            pass                # do something before the FCN is called for the first time
        
        # Retrieve the parameters from the C side of ROOT and
        # store them in a Python list -- VERY resource-intensive
        # for many calls, but can't be improved (yet?)
        parameter_list = []
        for j in range(self.number_of_parameters):
            parameter_list.append(parameters[j])
         
        f[0] = self.function_to_minimize(*parameter_list) # call the Python implementation of FCN.
        
#         f[0] = self.function_to_minimize(parameter_list) # call the Python FCN.

    def minimize(self):
        '''Do the minimization'''
        # Run minimization algorithm (MIGRAD + HESSE)
        error_code = Long(0)
        
        self.__gMinuit.mnexcm("MIGRAD", arr('d', [self.max_iterations, self.tolerance]), 2, error_code)
        self.__gMinuit.mnexcm("HESSE", arr('d', [6000]), 1, error_code) # Call HESSE with max. 6000 calls


if __name__ == "__main__":
    
    def quadratic(x=3e-24, y=4e-23):
        #return (args[0] - 3.14) ** 2 + (args[1] + 2.71) ** 2 + 1.0    # should be minimal (=1.0) at arg[0] = 3.14 and arg[1] = -2.71
        return (x - 3.14e-24) ** 2 + (y + 2.71e-23) ** 2 + 1.0    # should be minimal (=1.0) at arg[0] = 3.14 and arg[1] = -2.71
       
    
    myMinimizer = Minuit(quadratic)
    
    myMinimizer.minimize()
    print myMinimizer.get_error_matrix()
    #myMinimizer.minimize()
    #print myMinimizer.get_error_matrix()
    #myMinimizer.minimize()
    #print myMinimizer.get_error_matrix()
    #myMinimizer.minimize()
    #print myMinimizer.get_error_matrix()
    
    print myMinimizer.get_chi2_probability(1)
    
    