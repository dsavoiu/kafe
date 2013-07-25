'''
.. module:: dataset
   :platform: Unix
   :synopsis: This submodule defines a `Dataset` object, a container class for storing measurement data and error data.
   
.. moduleauthor:: Daniel Savoiu <daniel.savoiu@ekp.kit.edu>
'''

from string import join, split

from fit import Fit
from numeric_tools import *     # automatically includes numpy as np

DEBUG_MODE = 1

def debug_print(message):
    if DEBUG_MODE:
        print message        

def build_dataset(xdata, ydata, **kwargs):
        '''
        This helper function creates a `Dataset` from a series of keyword arguments.
        
        Valid keyword arguments are: 
        
        **xdata** and **ydata**
            These keyword arguments are mandatory and should be iterables containing the measurement data.
        
        *error specification keywords*
            A valid keyword is composed of an axis (*x* or *y*), an error relativity specification (*abs* or *rel*)
            and error correlation type (*stat* or *syst*). The errors are then set as follows:
            
                1. For statistical errors:
                    - if keyword argument is a `NumPy` array, the error list is set to that
                    - if keyword argument is a number, an error list with identical entries is generated 
                2. For systematic errors:
                    - keyword argument *must* be a single number. The global correlated error for the acis is then set to that.
            
            So, for example:
            
            >>> myDataset = build_dataset(..., yabsstat=0.3, yrelsyst=0.1)
            
            creates a dataset where the statistical error of each `y` coordinate is set to 0.3 and the overall systematic
            error of `y` is set to 0.1. 
            
        '''
        
        # cast data to array
        data = (np.asarray(xdata), np.asarray(ydata))
        size = len(xdata)
        
        # check that x and y data have the same length
        if size != len(ydata):
            raise Exception, "xdata and ydata must have matching lengths (%d != %d)" % (size, len(ydata))
        
        # initialize cov_mats with zero matrices
        cov_mats = [np.asmatrix(np.zeros((size, size))), np.asmatrix(np.zeros((size, size)))]
        
        kwargs_to_transmit = {}
        
        for key, val in kwargs.iteritems():   # go through the keyword arguments
        
            if key in ('title'):
                kwargs_to_transmit.update({key: val})
            else:
                err_spec = key
                err_val = val
        
            if len(err_spec) != 8:       # check that the error specification has required length
                raise SyntaxError, "Cannot interpret error specification `%s'." % (err_spec,)
            
            # interpret the error specification
            axis = err_spec[0]          # extract the axis from the error specification
            relativity = err_spec[1:4]  # extract the relativity from the error spec.
            correlation = err_spec[4:]  # extract the correlation from the error spec.
            
            # check error specification for integrity
            if axis not in ('x', 'y'):
                raise SyntaxError, "Unknown axis `%s'." % (axis, )
            if relativity not in ('abs', 'rel'):
                raise SyntaxError, "Unknown relativity specification `%s'. Expected `abs' or `rel'." % (relativity, )
            if correlation not in ('stat', 'syst'):
                raise SyntaxError, "Unknown correlation specification `%s'. Expected `stat' or `syst'." % (correlation, )
            
            # get axis is from axis name
            axis = ('x', 'y').index(axis)
            
            # make sure errors are floats. Cast to float if necessary...
            if isinstance(err_val, np.ndarray) or isinstance(err_val, int):
                # cast err_val to a float
                err_val = 1.0 * err_val
            
            if correlation == 'syst':
                # systematic errors should be floats
                if not isinstance(err_val, float):
                    # if not, raise error
                    raise SyntaxError, "Error setting systematic error `%s', expected number." % (err_spec,)
                
                # otherwise, calculate covariance matrix
                if relativity == 'rel':             
                    err_val *= data[axis]  # relative errors need to be weighted with the actual data

                    # systematic error matrix given by outer product of err_var vector with itself
                    cov_mats[axis] += np.asmatrix( np.outer(err_val, err_val) )
                else:
                    # systematic error matrix is proportional to np.ones
                    cov_mats[axis] += np.asmatrix( np.ones((size,size)) * err_val**2 )
                    
            elif correlation == 'stat':
                # statistical errors should be error lists
                if isinstance(err_val, float): # if a float value is given
                    # turn float value into array of identical values
                    err_val = np.ones(size) * err_val
                
                # check if err_val is iterable    
                try:
                    iter(err_val)
                except:
                    raise SyntaxError, "Error setting statistical error `%s', expected number or NumPy array." % (err_spec,)
                else:
                    err_val = np.asarray(err_val)   # cast to numpy array
                
                if relativity == 'rel':             
                    err_val *= data[axis]  # relative errors need to be weighted with the actual data
                
                cov_mats[axis] += np.asmatrix( np.diag(err_val)**2 )
                    
        return Dataset(data=data, cov_mats=cov_mats, **kwargs_to_transmit)


class Dataset: #(object):
    '''
    The `Dataset` object is a data structure for storing measurement and error data. In this implementation,
    the `Dataset` has the compulsory field `data`, which is used for storing the measurement data,
    and another field `cov_mats`, used for storing the covariance matrix for each axis.

    There are several ways a `Dataset` can be constructed. The most straightforward way is to specify an
    input file containing a plain-text representation of the dataset:
    
    >>> my_dataset = Dataset(input_file='/path/to/file')
    
    or
    
    >>> my_dataset = Dataset(input_file=my_file_object)
    
    If an `input_file` keyword is provided, all other input is ignored. The `Dataset` plain-text representation
    format is as follows::
    
        # x data
        x_1  sigma_x_1  
        x_2  sigma_x_2  cor_x_12
        ...  ...        ...       ...
        x_N  sigma_x_N  cor_x_1N  ...  cor_x_NN
        
        # y data
        y_1  sigma_y_1  
        y_2  sigma_y_2  cor_y_12
        ...  ...        ...       ...
        y_N  sigma_y_N  cor_y_1N  ...  cor_y_NN
    
    Here, the `sigma_...` represents the statistical error of the data point and `cor_..._ij` is the
    correlation coefficient between the *i*-th and *j*-th data point. 
    
    Alternatively, field data can be set by passing iterables as keyword arguments. Available keywords
    for this purpose are:
    
    **data**
    
        a tuple/list of measurement data. Each element of the tuple/list must be iterable and 
        be of the same length. The first element of the **data** tuple/list is assumed to be
        the `x` data, and the second to be the `y` data:
        
        >>> my_dataset = Dataset(data=([0., 1., 2., 3., 4.], [1.23, 3.45, 5.62, 7.88, 9.64]))
        
        Alternatively, x-y value pairs can also be passed as **data**. The following is equivalent to the above:
        
        >>> my_dataset = Dataset(data=([0.0, 1.23], [1.0, 3.45], [2.0, 5.62], [3.0, 7.88], [4.0, 9.64]))
        
        In case the `Dataset` contains two data points, the ordering is ambiguous. In this case, the
        first ordering (`x` data first, then `y` data) is assumed.
        
    **cov_mats**
    
        a tuple/list of two-dimensional iterables containing the covariance matrices for `x` and `y`, in that
        order. Covariance matrices can be any sort of two-dimensional NxN iterables, assuming N is the number
        of data points.
        
        >>> my_dataset = Dataset(data=([0., 1., 2.], [1.23, 3.45, 5.62]), cov_mats=(my_cov_mat_x, my_cov_mat_y))
        
        This keyword argument can be omitted, in which case covariance matrices of zero are assumed.
        To specify a covariance matrix for a single axis, replace the other with ``None``.
    
        >>> my_dataset = Dataset(data=([0., 1., 2.], [1.23, 3.45, 5.62]), cov_mats=(None, my_cov_mat_y))
    
    **title**
    
        the name of the `Dataset`. If omitted, the `Dataset` will be given the generic name 'Untitled Dataset'.
            
    '''

    def __init__(self, **kwargs):
        '''Construct the Dataset'''
        
        # Definitions
        ##############
        
        
        self.n_axes = 2         #: dimensionality of the `Dataset`. Currently, only 2D `Datasets` are supported
        self.n_datapoints = 0   #: number of data points in the `Dataset`   
        self.data = [None, None]          #: list containing measurement data (axis-ordering) 
        self.cov_mats = [None, None]      #: list of covariance matrices
        
        # Metadata
        self.axis_labels = ['x', 'y']       #: axis labels
        self.axis_units = ['', '']          #: units to assume for axis
        
        self.__axis_alias = {0: 0, 1: 1, 'x': 0, 'y': 1, '0': 0, '1': 1}    #: dictionary to get axis id from an alias 
        
        # Some boolean fields for simple yes/no queries
        self.__query_cov_mats_regular = [False, False]    #: a list of booleans indicating whether covariance matrices are regular (``True``) or singular (``False``)
        self.__query_has_errors = [False, False]          #: a list of booleans indicating whether statistical errors are provided for an axis
        self.__query_has_correlations = [False, False]    #: a list of booleans indicating whether error correlations are provided for an axis
        
        # Process keyword arguments
        #############################
        
        # name the Dataset
        if kwargs.has_key('title'):
            self.data_label = kwargs['title']
        else:
            self.data_label = 'Untitled Dataset'

        # check for an input file
        if kwargs.has_key('input_file'):
            self.read_from_file(kwargs['input_file'])
            return   # exit constructor after loading input file
            
        # Load data
        ############
        
        # preliminary checks
        if not kwargs.has_key('data'):
            raise Exception, "No data provided for Dataset."
        else:
            if len(kwargs['data']) != self.n_axes:
                raise Exception, "Unsupported number of axes: %d" % (len(kwargs['data']),)
            
        
        for axis in range(self.n_axes):         # go through the axes
            self.set_data(axis, kwargs['data'][axis])        # load data for axis            
            self.set_cov_mat(axis, kwargs['cov_mats'][axis]) # load cov mat for axis
    
    # Set methods
    ##############
    
    def set_data(self, axis, data):
        '''
        Set the measurement data for an axis.
        
        **axis**
            Axis for which to set the measurement data. Can be ``'x'`` or ``'y'``. Type: string
            
        **data**
            Measurement data for axis. Type: any iterable
        '''
        
        # get axis id from an alias
        axis = self.get_axis(axis)
        
        try:
            # check if the object is iterable (list or array)
            # by trying to get its iterator.
            iterator = iter(data)
        except TypeError:
            # if this fails, then this object is not iterable
            raise TypeError, "Error loading data for axis `%s`. Expected iterable, got %s." % (axis, type(data))
        else:
            # if that succeeds, then this object is iterable
            self.data[axis] = np.asarray(data) # cast the iterable to a numpy array and store data
            if axis==0:
                self.n_datapoints = len(self.data[0]) # set the dataset's size
            
    
    def set_cov_mat(self, axis, mat):
        '''
        Set the error matrix for an axis.
        
        **axis**
            Axis for which to load the error matrix. Can be ``'x'`` or ``'y'``.
             
        **mat**
            Error matrix for the axis. Passing `None` unsets the error matrix. Type: `numpy.matrix` or `None`
        '''
        
        # get axis id from an alias
        axis = self.get_axis(axis)
        
        if mat is not None:
            try:
                mat = np.asmatrix(mat) # try to cast argument to a matrix
            except:
                raise TypeError, "Cast to matrix failed. Object was of type `%s'" % (type(mat),)
            
        # check if the covariance matrix is singular and set/unset a flag accordingly
        try:
            mat.I    # try to invert it
        except:
            self.__query_cov_mats_regular[axis] = True    # if that fails, mat is singular
        else:
            self.__query_cov_mats_regular[axis] = False   # else, mat is regular
        
        # check if the matrix is zero or None and set/unset a flag accordingly
        if mat is None or (mat==0).all():               # check if matrix in None or zero
            self.__query_has_errors[axis] = False
            self.__query_has_correlations[axis] = False
        elif (np.diag(np.diag(mat))==mat).all():        # check if matrix is diagonal
            self.__query_has_errors[axis] = True
            self.__query_has_correlations[axis] = False
        else:
            self.__query_has_errors[axis] = True
            self.__query_has_correlations[axis] = True
        
        # set the matrix
        if mat is None:
            self.cov_mats[axis] = np.asmatrix(np.zeros((self.get_size(),self.get_size())))
        else:
            self.cov_mats[axis] = mat

    # Get methods
    ##############

    def get_axis(self, axis_alias):
        '''
        Get axis id from an alias.
        
        **axis_alias**
            Alias of the axis whose id should be returned. This is for example either ``'0'`` or ``'x'`` for the `x`-axis (id 0).
        '''
        
        try:
            axis = self.__axis_alias[axis_alias]
        except:
            raise SyntaxError, "Unknown axis %s." % (axis_alias, )
        
        return axis

    def get_size(self):
        '''
        Get the size of the `Dataset`. This is equivalent to the length of the `x`-axis data.
        '''
        
        if self.data[0] is None:
            return 0
        else:
            return len(self.data[0])

    def get_data_span(self, axis, include_error_bars=False):
        '''
        Get the data span for an axis. The data span is a tuple (`min`, `max`) containing
        the smallest and highest coordinates for an axis.
        
        **axis**
            Axis for which to get the data span. Can be ``'x'`` or ``'y'``. Type: string
        
        *include_error_bars* : bool
            ``True`` if the returned span should be enlarged to
            contain the error bars of the smallest and largest datapoints (default: ``False``) Type: boolean
        '''
        
        # get axis id from an alias
        axis = self.get_axis(axis)
        
        max_error_bar_size = 0. 
        min_error_bar_size = 0. 
        
        max_idx = tuple(self.get_data(axis)).index(max(self.get_data(axis)))   # get the index of the max datapoint
        min_idx = tuple(self.get_data(axis)).index(min(self.get_data(axis)))   # get the index of the min datapoint
        
        if include_error_bars:
            #try:
            max_error_bar_size = np.sqrt(self.get_cov_mat(axis)[max_idx, max_idx])  # get the error of the maximum datapoint
            min_error_bar_size = np.sqrt(self.get_cov_mat(axis)[min_idx, min_idx])  # get the error of the minimum datapoint
            #except:
            #    pass
        
        return [self.get_data(axis)[min_idx] - min_error_bar_size, self.get_data(axis)[max_idx] + max_error_bar_size]
        
    def get_data(self, axis):
        '''
        Get the measurement data for an axis.
        
        **axis**
            Axis for which to get the measurement data. Can be ``'x'`` or ``'y'``. Type: string
        '''
        
        # get axis id from an alias
        axis = self.get_axis(axis)
        
        return self.data[axis]
    
    def get_cov_mat(self, axis, fallback_on_singular=None):
        '''
        Get the error matrix for an axis.
        
        **axis** string or int
            Axis for which to load the error matrix. Can be ``'x'`` or ``'y'``. Type: string
            
        *fallback_on_singular* : `numpy.matrix` or string
            What to return if the matrix is singular. If this is ``None`` (default), the matrix is returned anyway.
            If this is a `numpy.matrix` object or similar, that is returned istead. Alternatively, the shortcuts
            ``'identity'`` or ``1`` and ``'zero'`` or ``0`` can be used to return the identity and zero matrix
            respectively. 
        '''
        
        # get axis id from an alias
        axis = self.get_axis(axis)
        if fallback_on_singular is None:
            return self.cov_mats[axis]
        else:
            if not self.__query_cov_mats_regular[axis]: # if matrix is singular
                try:
                    fallback_matrix=np.asmatrix(fallback_on_singular)   # try to cast to matrix
                except:
                    if fallback_on_singular=='identity' or fallback_on_singular==1:
                        fallback_matrix=np.eye(self.get_size())
                    elif fallback_on_singular=='zero' or fallback_on_singular==0:
                        fallback_matrix=np.zeros((self.get_size(), self.get_size()))
                    elif fallback_on_singular=='report':
                        print "Warning: Covariance matrix for axis %s is singular!" % (axis,)
                        return self.cov_mats[axis]  # if not, return the (regular) matrix itself
                    else:
                        raise SyntaxError, "Cannot interpret fallback matrix specification `%s`" % (fallback_matrix,)
                    
                return fallback_matrix  # return the fallback matrix
            else:
                return self.cov_mats[axis]  # if not, return the (regular) matrix itself
    
    # Other methods
    ################
    
    def cov_mat_is_regular(self, axis):
        '''
        Returns `True` if the covariance matrix for an axis is regular and ``False`` if it is
        singular.
        
        **axis** : string or int
            Axis for which to check for regularity of the covariance matrix. Can be ``'x'`` or ``'y'``.
        
        '''
        
        # get axis id from alias
        axis = self.get_axis(axis)
        
        return self.__query_cov_mats_regular[axis]
    
    def has_correlations(self, axis):
        '''
        Returns `True` if the covariance matrix for an axis is regular and ``False`` if it is
        singular.
        
        **axis** string or int
            Axis for which to check for regularity of the covariance matrix. Can be ``'x'`` or ``'y'``.
        
        '''
        
        # get axis id from alias
        axis = self.get_axis(axis)
        return self.__query_has_correlations[axis]
    
    def has_errors(self, axis):
        '''
        Returns `True` if the covariance matrix for an axis is regular and ``False`` if it is
        singular.
        
        **axis** string or int
            Axis for which to check for regularity of the covariance matrix. Can be ``'x'`` or ``'y'``.
        
        '''
        
        # get axis id from alias
        axis = self.get_axis(axis)
        return self.__query_has_errors[axis]
    
    def get_formatted(self, format_string=".06e", delimiter='\t'):
        '''
        Returns the dataset in a plain-text format which is human-readable and
        can later be used as an input file for the creation of a new `Dataset`.
       
        .. _get_formatted: 
       
        The format is as follows::
        
            # x data
            x_1  sigma_x_1  
            x_2  sigma_x_2  cor_x_12
            ...  ...        ...       ...
            x_N  sigma_x_N  cor_x_1N  ...  cor_x_NN
            
            # y data
            y_1  sigma_y_1  
            y_2  sigma_y_2  cor_y_12
            ...  ...        ...       ...
            y_N  sigma_y_N  cor_y_1N  ...  cor_y_NN
        
        Here, the ``x_i`` and ``y_i`` represent the measurement data, the ``sigma_?_i`` are the
        statistical uncertainties of each data point, and the ``cor_?_ij`` are the correlation
        coefficients between the *i*-th and *j*-th data point.
        
        If the ``x`` or ``y`` errors are not correlated, then the entire correlation coefficient matrix
        can be omitted. If there are no statistical uncertainties for an axis, the second
        column can also be omitted. A blank line is required at the end of each data block!
        
        *format_string* : string (optional)
            A format string with which each entry will be rendered. Default is ``'.06e'``, which means
            the numbers are represented in scientific notation with six significant digits.
            
        *delimiter* : string (optional)
            A delimiter used to separate columns in the output.
        
        '''
        
        output_list = []
        
        # go through the axes
        for axis in range(self.n_axes):
            helper_list = []                                                # define a helper list which we will fill out
            stat_errs = extract_statistical_errors(self.get_cov_mat(axis))  # get the statistical errors of the data
            data = self.get_data(axis)
            cor_mat = cov_to_cor(self.get_cov_mat(axis))
            
            helper_list.append(['# Axis %d: %s' % (axis, self.axis_labels[axis])]) # add section title as a comment
            helper_list.append(['# datapoints'])                                   # add a row for headings
            
            if self.__query_has_errors[axis]:                             # if the dataset has stat errors
                helper_list[-1].append('stat. err.')                    # add a heading for second column
                if self.__query_has_correlations[axis]:                          # if there are also correlations (syst errors)
                    helper_list[-1].append('correlation coefficients')  # add a heading for the correlation matrix
            
            for idx in range(len(data)):
                helper_list.append([])                                              # append a new "row" to the helper list
                helper_list[-1].append( format(data[idx], format_string) )          # append the coordinate of the data point
                if self.__query_has_errors[axis]:                             # if the dataset has stat errors
                    helper_list[-1].append( format(stat_errs[idx], format_string) ) # append the stat error of the data point
                    if self.__query_has_correlations[axis]:                          # if there are also correlations (syst errors)
                        for col in range(idx):                                                  # go through the columns of the correlation matrix
                            helper_list[-1].append( format(cor_mat[idx, col], format_string))   # append the correlation coefficients to the helper list
            
            helper_list.append([])  # append an empty list -> blank line
            output_list.append(helper_list)
            
            
        # turn the list into a string
        tmp_string = '' #
        for row in output_list:
            for entry in row:
                tmp_string += join(entry, delimiter) + '\n'
                
        return tmp_string
                    
    def write_formatted(self, file_path, format_string=".06e", delimiter='\t'):
        '''
        Writes the dataset to a plain-text file. For details on the format, see `get_formatted`_.
        
        
        **file_path** : string
            Path of the file object to write. **WARNING**: *overwrites existing files*!
            
        *format_string* : string (optional)
            A format string with which each entry will be rendered. Default is ``'.06e'``, which means
            the numbers are represented in scientific notation with six significant digits.
            
        *delimiter* : string (optional)
            A delimiter used to separate columns in the output.
        
        '''
        
        # write the output of self.get_formatted to the file
        with open(file_path, 'w') as my_file:
            my_file.write(self.get_formatted(format_string))
            my_file.close()
                       
    def read_from_file(self, input_file):
        '''
        Reads the `Dataset` object from a file.
        
        returns : boolean
            ``True`` if the read succeeded, ``False`` if not.
            
        '''
        
        try:
            tmp_lines = input_file.readlines()        # try to read the lines of the file
        except AttributeError:                      # this will fail if a file path string was passed, so alternatively:
            tmp_file = open(input_file, 'r')          # open the file pointed to by the path
            tmp_lines = tmp_file.readlines()        # and then read the lines of the file
            
        # Parse the file
        #################
            
        tmp_data = []       # append data progressively to this list 
        tmp_errors = []     # append error data progressively to this list
        tmp_cormat = []     # temporary storage of the correlation matrix
        
        tmp_rownumber = 0   # keep track of the current (error matrix) row
        tmp_axis = 0        # keep track of the current axis
        tmp_linenumber = 0  # keep track of the current line being read
        
        tmp_has_stat_errors = True # assume statistical errors
        tmp_has_syst_errors = True # assume correlations
        
        tmp_reading_data_block = False # don't assume that the file begins with a data block 
        
        for line in tmp_lines:                      # go through the lines of the file
            
            tmp_linenumber += 1                     # update the line number
            
            if '#' in line:
                line = split(line, '#')[0]          # ignore anything after a comment sign (#)
                
            if (not line) or (line.isspace()):      # if empty line encountered
                if tmp_reading_data_block:          # if currenty reading a data block, end reading it and commit data
                    
                    # Commit the parsed data to the object
                    #######################################
                    
                    self.set_data(tmp_axis, tmp_data)                                     # commit measurement data
                    
                    if tmp_has_syst_errors: # if there is a correlation matrix
                        # Turn the lists into a lower triangle matrix
                        tmp_cormat = zero_pad_lower_triangle(tmp_cormat)
                    
                        # Symmetrize: copy the lower triangle to the upper half
                        tmp_cormat = make_symmetric_lower(tmp_cormat)
                    
                        self.set_cov_mat(tmp_axis, cor_to_cov(tmp_cormat, tmp_errors))    # commit covariance matrix

                    elif tmp_has_stat_errors: # if there are just statistical errors
                        self.set_cov_mat(tmp_axis, np.asmatrix(np.diag(tmp_errors)**2))   # commit covariance matrix
                    else: # if there are no errors
                        self.set_cov_mat(tmp_axis, None)                                  # unset cov mat
                        
                    # Reset temporary variables
                    #############################
                    
                    tmp_data = []       # append data progressively to this list 
                    tmp_errors = []     # append error data progressively to 
                    tmp_cormat = []     # temporary storage of the correlation matrix
                    tmp_has_stat_errors = True # assume statistical errors
                    tmp_has_syst_errors = True # assume correlations
                    tmp_rownumber = 0   # reset row number
                    tmp_axis += 1       # go to next axis
                    tmp_reading_data_block = False # end data block
                    
            else:  # else, if line is not empty, it must contain data
                
                if not tmp_reading_data_block:
                    tmp_reading_data_block = True       # attest that we are currently reading a data block  
                
                tmp_fields = split(line)                # get the entries on the line as a list (whitespace-delimited)
                
                if len(tmp_fields) == 1:        # if there is only one entry, we know it's just the measurement data
                    if tmp_has_stat_errors:
                        tmp_has_stat_errors = False # no stat errs for axis
                    if tmp_has_syst_errors:
                        tmp_has_syst_errors = False # no syst errs for axis
            
                tmp_data.append(float(tmp_fields[0]))   # first field is the coordinate of the data point
                
                if tmp_has_stat_errors:
                    tmp_errors.append(float(tmp_fields[1])) # second field is the error in that coordinate
                
                if tmp_has_syst_errors:                 # if there are correlations
                    tmp_cormat.append(map(float,tmp_fields[2:]) + [1.0])    # other fields are correlation coefficients (diagonal 1.0 added)
            
                if len(tmp_fields) != tmp_rownumber+2:  # if there are not enough entries for a valid correlation matrix
                    tmp_has_syst_errors = False         # attest that there is not a valid correlation matrix
                
                tmp_rownumber += 1                  # update row number
        
        return True
        
        
        
if __name__ == "__main__":

#     def linear_2par(x, slope=1, x_offset=0):
#         
#         return slope * (x - x_offset)
#     
#     myXData = np.asarray([1, 2, 3, 4, 5])
#     myYData = np.asarray([2, 3.1, 4.2, 5.5, 7.23])
#     
#     tstFloat = 0.1
#     tstArray = np.asarray([.1, .1, .2, .5, .8])
#     
#     
#     
#     myDataset = Dataset(xdata=myXData, 
#                          ydata=myYData,
#                          xabsstat=0.1,
#                          xabssyst=0.1)
# 
#     print myDataset.get_formatted()
#     
#     myDataset.write_formatted('/home/daniel/tmp/23465.inp')
#     myDataset = Dataset(input_file='/home/daniel/tmp/23465.inp')
#     
#     print myDataset.get_formatted()
    
    myDataset = build_dataset(xdata = [1,2,3], ydata=[2,4,8], yabssyst=0.1, yrelstat=.1)
    
    