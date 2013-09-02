'''
.. module:: function_tools
   :platform: Unix
   :synopsis: This submodule contains several useful tools for getting information about a function, including the number, names and default values of its parameters and its derivatives with respect to the independent variable or the parameters.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

'''

import numpy as np                                  # import NumPy
from scipy.misc import derivative as scipy_der      # get a numerical derivative calculating function from SciPy
from copy import copy                               # needed to make shallow copies of objects instead of references

from constants import *

def derivative(func, derive_by_index, variables_tuple, derivative_spacing):
    r'''
    Gives :math:`\frac{\partial f}{\partial x_k}` for :math:`f = f(x_0, x_1, \ldots)`. `func` is :math:`f`, `variables_tuple` is :math:`\{x_i\}` and `derive_by_index` is :math:`k`.
    '''
    
    # define a dummy function, so that the variable by which f is to be derived is the only variable
    def tmp_func(derive_by_var):
        argument_list = []
        for arg_nr, arg_val in enumerate(variables_tuple):
            if arg_nr==derive_by_index:
                argument_list.append(derive_by_var)
            else:
                argument_list.append(arg_val)
        return func(*argument_list)    
    
    # return the derivative of that function
    return scipy_der(tmp_func, variables_tuple[derive_by_index], dx=derivative_spacing)

def get_function_property(func, prop):
    '''
    Returns a specific property of the function. This assumes that the function is defined as
    
        >>> def func(x, par1=1.0, par2=3.14, par3=2.71, ...): ...
    
    **func** : function
        A function object from which to extract the property.
    
    **prop** : any of ``'name'``, ``'parameter names'``, ``'parameter defaults'``, ``'number of parameters'``
        A string representing a property.
    '''
    
    if prop == 'name':
        return func.__name__                    # get the function name from the Python function
    elif prop == 'number of parameters':
        return func.func_code.co_argcount-1     # number of parameters is the argument number - 1
    elif prop == 'parameter names':
        return func.func_code.co_varnames[1:func.func_code.co_argcount]   # get list of parameter names
    elif prop == 'parameter defaults':
        return func.func_defaults               # get list of parameter defaults
    else:
        raise Exception, "Error: Unknown function property `%s'." % (prop,)

def derive_by_x(func, x_0, param_list, derivative_spacing):
    r'''
    If `x_0` is iterable, gives the array of derivatives of a function :math:`f(x, par_1, par_2, \ldots)`
    around :math:`x = x_i` at every :math:`x_i` in :math:`\vec{x}`.
    If `x_0` is not iterable, gives the derivative of a function :math:`f(x, par_1, par_2, \ldots)` around :math:`x = \verb!x_0!`.
    '''
    try:
        iterator_over_x_0 = iter(x_0) # try to get an iterator object
    except TypeError, te:
        # object is not iterable, return the derivative in x_0 (float)
        return scipy_der(func, x_0, args=param_list, dx=derivative_spacing)
    else:
        # object is iterable, go through it and derive at each x_0 in it
        output_list = []
        for x in iterator_over_x_0:
            output_list.append(derive_by_x(func, x, param_list, derivative_spacing)) # call recursively
            
        return np.asarray(output_list)

def derive_by_parameters(func, x_0, param_list, derivative_spacing):
    r'''
    Returns the gradient of `func` with respect to its parameters, i.e. with respect to every variable
    of `func` except the first one.
    '''
    output_list = []
    variables_tuple = tuple([x_0] + list(param_list))   # compile all function arguments into a variables tuple
    
    for derive_by_index in xrange(1, func.func_code.co_argcount): # go through all arguments except the first one
        output_list.append(derivative(func, derive_by_index, variables_tuple, derivative_spacing))
    
    return np.asarray(output_list)
    

def outer_product(input_array):
    r'''
    Takes a `NumPy` array and returns the outer (dyadic, Kronecker) product with itself.
    If `input_array` is a vector :math:`\mathbf{x}`, this returns :math:`\mathbf{x}\mathbf{x}^T`.
    ''' 
    la = len(input_array)                                       # get vector size
    return np.kron(input_array, input_array).reshape(la, la)    # return outer product as numpy array


if __name__ == '__main__':
    def test_function(x, par0=3.14, par1=1.11, par2=2.71):
        return (x - par0) ** 2 - par1 * x + par2
    
    def test_function_grad(x, par0=3.14, par1=1.11, par2=2.71):
        return (
                 2 * (x - par0) - par1, # derivative by x
                -2 * (x - par0),        # derivative by par0
                -x,                     # derivative by par1
                1)                      # derivative by par2

        
    x_array = [0., 1., 2., 3.]
    
    print '    df/dx|x=0 : ', derive_by_x(test_function, 0, (1,-1,0))
    print 'df/dx_vec|x=0..3 : ', derive_by_x(test_function, x_array, (1,-1,0))
    
    #print outer_product(derive_by_x(test_function, x_array, (1,-1,0)))
    
    print outer_product(derive_by_parameters(test_function, 0, (1,-1,0)))



# class Function:
#     '''A class representing a 1D function on the reals.'''
# 
#     def __init__(self, function_pointer):
#         '''
#             Construct the Function object. Arguments:
#             
#                 function_pointer:    the Python function to be used as a 1D function on the reals.
#             
#             It is assumed that the first argument of this function is the indenpendent variable.
#             All other arguments are parameters. These MUST have a default value.
#         '''
#         
#         self.function_pointer = function_pointer    # get the Python function
#         
#         # magic: use the Python function object to extract some info about the function
#         self.name = self.function_pointer.__name__                                  # get the function name from the Python function
#         self.number_of_parameters = self.function_pointer.func_code.co_argcount-1   # number of parameters is the argument number - 1
#         self.parameters = self.function_pointer.func_code.co_varnames[1:self.number_of_parameters+1] # get list of parameter names
#         self.parameter_defaults = self.function_pointer.func_defaults               # get list of parameter defaults
#         
#         # verify that each parameter has a default value and fail if this is not the case
#         if len(self.parameters) != len(self.parameter_defaults):
#             raise SyntaxError, 'One ore more default parameter values not specified for function %s' % (self.name,)
#         
#     def __call__(self, *args, **kwargs):
#         '''Call the function object. This just forwards the call to the original Python function.'''
#         return self.function_pointer(*args, **kwargs)
#     
#     def elementwise(self, x_array, *args, **kwargs):
#         '''Call the function for every element of x_array and return an array of results.'''
#         output_list = []
#         for x in x_array:
#             output_list.append(self.__call__(x, *args, **kwargs))
#         return np.array(output_list)
#     
#     def derivatives(self, x_array, *args):
#         '''Get the derivative of the function at every point in x_array and return an array of results.'''
#         output_list = []
#         for x in x_array:
#             tmp_arg_list = [x] + list(args) # add the x as "zeroth variable"
#             output_list.append(derivative(self.function_pointer, 0, tmp_arg_list)) # get the derivative in this zeroth (independent) variable
#         return np.array(output_list)
# 
#     def derivatives_by_parameters(self, x_0, *args):
#         '''Return an array of derivatives df/dp_i (p_i runs through parameters) at a certain point x_0'''
#         output_list = []
#         tmp_arg_list = [x_0] + list(args) # add the x as "zeroth variable"
#         for arg_index in xrange(1, len(args) + 1):
#             output_list.append(derivative(self.function_pointer, arg_index, tmp_arg_list)) # get the derivatives in the arguments
#         return np.array(output_list)
#         
#     def derivatives_outer_product(self, x_array, *args):
#         '''Returns a matrix A for which A[i][j] = f'(x[i])*f'(x[j])'''
#         fp = self.derivatives(x_array, *args)   # calculate the derivative vector
#         la = len(x_array)                       # get vector size
#         return np.kron(fp,fp).reshape(la, la)   # return outer product as numpy array
#     
#     def derivatives_by_parameters_outer_product(self, x_0, *args):
#         '''Returns a matrix A for which A[i][j] = df/dp_i * df/dp_j at a point x_0'''
#         fp = self.derivatives_by_parameters(x_0, *args) # calculate the derivative vector
#         la = len(args) # get vector size
#         return np.kron(fp,fp).reshape(la, la)           # return outer product as numpy array


if __name__ == "__main__":
    def myf(x, y=1, z=3):
        latex = 'x^2 + y^3 - y z + x z^2'
        return x*x + y**2 - y*z + x*z**2
    
        # df/dx =  2x + z^2
        # df/dy =  2y - z
        # df/dz = 2zx - y
 
    def function_info(y):
  
        y_code = \
        y.func_code.co_nlocals, \
        y.func_code.co_stacksize, \
        y.func_code.co_flags, \
        y.func_code.co_code, \
        y.func_code.co_consts, \
        y.func_code.co_names, \
        y.func_code.co_varnames, \
        y.func_code.co_filename, \
        y.func_code.co_firstlineno, \
        y.func_code.co_lnotab
 
        print y_code
         
    #print get_function_property(myf, 'latex')
    function_info(myf)
    
    
