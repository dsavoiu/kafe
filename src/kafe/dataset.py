'''
.. module:: dataset
   :platform: Unix
   :synopsis: This submodule defines a `Dataset` object, a container class for
        storing measurement data and error data.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''

from string import join, split

import numpy as np

from numeric_tools import cov_to_cor, cor_to_cov, extract_statistical_errors, \
    zero_pad_lower_triangle, make_symmetric_lower

NUMBER_OF_AXES = 2

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')


def build_dataset(xdata, ydata, cov_mats=None, **kwargs):
    '''
    This helper function creates a `Dataset` from a series of keyword
    arguments.

    Valid keyword arguments are:

    **xdata** and **ydata** : list/tuple/`np.array` of floats
        These keyword arguments are mandatory and should be iterables
        containing the measurement data.

    *cov_mats* : ``None`` or 2-tuple (optional)
        This argument defaults to ``None``, which means no covariance matrices
        are used. If covariance matrices are needed, a tuple with two entries
        (the first for `x` covariance matrices, the second for `y`) must be
        passed.

        Each element of this tuple may be either ``None`` or a NumPy matrix
        object containing a covariance matrix for the respective axis.

    *error specification keywords* : iterable or numeric (see below)
        In addition to covariance matrices, errors can be specified for each
        axis (*x* or *y*) according to a simplified error model.

        In this respect, a valid keyword is composed of an axis, an error
        relativity specification (*abs* or *rel*) and error correlation type
        (*err* or *cor*). The errors are then set as follows:

            1. For totally uncorrelated errors (*err*):
                - if keyword argument is iterable, the error list is set to \
that
                - if keyword argument is a number, an error list with \
identical entries is generated
            2. For fully correlated errors (*cor*):
                - keyword argument *must* be a single number. The global \
correlated error for the axis is then set to that.

        So, for example:

        >>> myDataset = build_dataset(..., yabserr=0.3, yrelcor=0.1)

        creates a Dataset with an uncorrelated error of 0.3 for each `y`
        coordinate and a fully correlated (systematic) error of `y` of 0.1.



    '''

    # cast data to array
    data = (np.asarray(xdata), np.asarray(ydata))
    size = len(xdata)

    # check that x and y data have the same length
    if size != len(ydata):
        raise Exception(
            "xdata and ydata must have matching lengths (%d != %d)"
            % (size, len(ydata))
        )

    # if no cov_mats specifies
    if cov_mats is None:
        # initialize cov_mats with zero matrices
        cov_mats = [
            np.asmatrix(np.zeros((size, size))),
            np.asmatrix(np.zeros((size, size)))
        ]
    else:
        # go through cov_mat specification
        for mat_id, mat in enumerate(cov_mats):
            # If `None` is specified, substitute zero matrix
            if mat is None:
                cov_mats[mat_id] = np.asmatrix(np.zeros((size, size)))

    kwargs_to_transmit = {}

    for key, val in kwargs.iteritems():   # go through the keyword arguments

        if key in ('title'):
            kwargs_to_transmit.update({key: val})
            continue
        else:
            err_spec = key
            err_val = val

        # check that the error specification has required length
        # TODO:   If more error types allowed, this check is no good,
        # TODO:   so implement dictionary lookup here if that should change.
        if len(err_spec) != 7:
            raise SyntaxError(
                "Cannot interpret error specification `%s'."
                % (err_spec,)
            )

        # interpret the error specification
        axis = err_spec[0]  # extract the axis from the error specification
        relativity = err_spec[1:4]  # extract the relativity from the err spec.
        correlation = err_spec[4:]  # extract the correl. from the error spec.

        # check error specification for integrity
        if axis not in ('x', 'y'):
            raise SyntaxError("Unknown axis `%s'." % (axis, ))
        if relativity not in ('abs', 'rel'):
            raise SyntaxError(
                "Unknown relativity specification `%s'. "
                "Expected `abs' or `rel'."
                % (relativity, )
            )
        if correlation not in ('err', 'cor'):
            raise SyntaxError(
                "Unknown correlation specification `%s'. "
                "Expected `err' or `cor'."
                % (correlation, )
            )

        # get axis is from axis name
        axis = ('x', 'y').index(axis)

        # make sure errors are floats. Cast to float if necessary...
        if isinstance(err_val, np.ndarray) or isinstance(err_val, int):
            # cast err_val to a float
            err_val = 1.0 * err_val

        if correlation == 'cor':
            # systematic errors should be floats
            if not isinstance(err_val, float):
                # if not, raise error
                raise SyntaxError(
                    "Error setting correlated error `%s', "
                    "expected number." % (err_spec,)
                )

            # otherwise, calculate covariance matrix
            if relativity == 'rel':
                err_val *= data[axis]  # relative errors need to be weighted
                                       # with the actual data

                # systematic error matrix given by outer product of err_var
                # vector with itself
                cov_mats[axis] += np.asmatrix(np.outer(err_val, err_val))
            else:
                # systematic error matrix is proportional to np.ones
                cov_mats[axis] += np.asmatrix(
                    np.ones((size, size)) * err_val**2
                )

        elif correlation == 'err':
            # statistical errors should be error lists
            if isinstance(err_val, float):  # if a float value is given
                # turn float value into array of identical values
                err_val = np.ones(size) * err_val

            # check if err_val is iterable
            try:
                iter(err_val)
            except:
                raise SyntaxError(
                    "Error setting statistical error `%s', "
                    "expected number or NumPy array."
                    % (err_spec,)
                )
            else:
                err_val = np.asarray(err_val)   # cast to numpy array

            if relativity == 'rel':
                err_val *= data[axis]  # relative errors need to be weighted
                                       # with the actual data

            cov_mats[axis] += np.asmatrix(np.diag(err_val)**2)

    return Dataset(data=data, cov_mats=cov_mats, **kwargs_to_transmit)


class Dataset(object):
    '''
    The `Dataset` object is a data structure for storing measurement and error
    data. In this implementation, the `Dataset` has the compulsory field
    `data`, which is used for storing the measurement data, and another field
    `cov_mats`, used for storing the covariance matrix for each axis.

    There are several ways a `Dataset` can be constructed. The most
    straightforward way is to specify an input file containing a plain-text
    representation of the dataset:

    >>> my_dataset = Dataset(input_file='/path/to/file')

    or

    >>> my_dataset = Dataset(input_file=my_file_object)

    If an `input_file` keyword is provided, all other input is ignored. The
    `Dataset` plain-text representation format is as follows::

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

    Here, the `sigma_...` represents the fully uncorrelated error of the data
    point and `cor_..._ij` is the correlation coefficient between the *i*-th
    and *j*-th data point.

    Alternatively, field data can be set by passing iterables as keyword
    arguments. Available keywords for this purpose are:

    **data** : tuple/list of tuples/lists/arrays of floats

        a tuple/list of measurement data. Each element of the tuple/list must
        be iterable and be of the same length. The first element of the
        **data** tuple/list is assumed to be the `x` data, and the second to be
        the `y` data:

        >>> my_dataset = Dataset(data=([0., 1., 2.], [1.23, 3.45, 5.62]))

        Alternatively, x-y value pairs can also be passed as **data**. The
        following is equivalent to the above:

        >>> my_dataset = Dataset(data=([0.0, 1.23], [1.0, 3.45], [2.0, 5.62]))

        In case the `Dataset` contains two data points, the ordering is
        ambiguous. In this case, the first ordering (`x` data first, then `y`
        data) is assumed.

    *cov_mats* : tuple/list of `numpy.matrix` (optional)

        a tuple/list of two-dimensional iterables containing the covariance
        matrices for `x` and `y`, in that order. Covariance matrices can be any
        sort of two-dimensional NxN iterables, assuming N is the number of data
        points.

        >>> my_dataset = Dataset(data=([0., 1., 2.], [1.23, 3.45, 5.62]), \
            cov_mats=(my_cov_mat_x, my_cov_mat_y))

        This keyword argument can be omitted, in which case covariance matrices
        of zero are assumed. To specify a covariance matrix for a single axis,
        replace the other with ``None``.

        >>> my_dataset = Dataset(data=([0., 1., 2.], [1.23, 3.45, 5.62]), \
            cov_mats=(None, my_cov_mat_y))

    *title* : string (optional)

        the name of the `Dataset`. If omitted, the `Dataset` will be given the
        generic name 'Untitled Dataset'.

    *axis_labels* : list of strings (optional)

        labels for the `x` and `y` axes. If omitted, these will be set to ``'x'``
        and ``'y'``, respectively.

    *axis_units* : list of strings (optional)

        units for the `x` and `y` axes. If omitted, these will be assumed to be
        dimensionless, i.e. the unit will be an empty string.
    '''

    def __init__(self, **kwargs):
        '''Construct the Dataset'''

        # Definitions
        ##############

        self.n_axes = 2
        """dimensionality of the `Dataset`. Currently, only 2D `Datasets` are
        supported"""
        self.n_datapoints = 0   #: number of data points in the `Dataset`
        self.data = [None, None]
        #: list containing measurement data (axis-ordering)
        self.cov_mats = [None, None]      #: list of covariance matrices

        # Metadata
        #: axis labels
        self.axis_labels = list(kwargs.get('axis_labels', ['x', 'y']))
        self.axis_units = list(kwargs.get('axis_units', ['', '']))

        #: dictionary to get axis id from an alias
        self.__axis_alias = {0: 0, 1: 1, 'x': 0, 'y': 1, '0': 0, '1': 1}

        # Some boolean fields for simple yes/no queries
        self.__query_cov_mats_regular = [False, False]
        """a list of booleans indicating whether covariance matrices are
            regular (``True``) or singular (``False``)"""
        self.__query_has_errors = [False, False]
        """a list of booleans indicating whether statistical errors are
            provided for an axis"""
        self.__query_has_correlations = [False, False]
        """a list of booleans indicating whether error correlations are
            provided for an axis"""

        # Process keyword arguments
        #############################

        # name the Dataset
        self.data_label = kwargs.get('title', "Untitled Dataset")

        # check for an input file
        if 'input_file' in kwargs:
            self.read_from_file(kwargs['input_file'])
            return   # exit constructor after loading input file

        # Load data
        ############

        # preliminary checks
        if not 'data' in kwargs:
            raise Exception("No data provided for Dataset.")
        else:
            if len(kwargs['data']) != NUMBER_OF_AXES:
                # in case of xy value pairs, transpose
                data = np.asarray(kwargs['data']).T.tolist()
                if len(data) != NUMBER_OF_AXES:
                    raise Exception(
                        "Unsupported number of axes: %d"
                        % (len(kwargs['data']),)
                    )
                else:
                    # set the transposed data as the read data
                    kwargs['data'] = data

        for axis in xrange(self.n_axes):  # go through the axes
            self.set_data(axis, kwargs['data'][axis])  # load data for axis
            if 'cov_mats' in kwargs:
                # load cov mat for axis
                self.set_cov_mat(axis, kwargs['cov_mats'][axis])
            else:
                # don't load cov mat for axis
                self.set_cov_mat(axis, None)

    # Set methods
    ##############

    def set_data(self, axis, data):
        '''
        Set the measurement data for an axis.

        **axis** : ``'x'`` or ``'y'``
            Axis for which to set the measurement data.

        **data** : iterable
            Measurement data for axis.
        '''

        # get axis id from an alias
        axis = self.get_axis(axis)

        try:
            # check if the object is iterable (list or array)
            # by trying to get its iterator.
            iter(data)
        except TypeError:
            # if this fails, then this object is not iterable
            raise TypeError(
                "Error loading data for axis `%s`. "
                "Expected iterable, got %s."
                % (axis, type(data))
            )
        else:
            # if that succeeds, then this object is iterable:
            # cast the iterable to a numpy array and store data
            self.data[axis] = np.asarray(data)
            if axis == 0:
                self.n_datapoints = len(self.data[0])  # set the dataset's size

    def set_cov_mat(self, axis, mat):
        '''
        Set the error matrix for an axis.

        **axis** : ``'x'`` or ``'y'``
            Axis for which to load the error matrix.

        **mat** : `numpy.matrix` or ``None``
            Error matrix for the axis. Passing ``None`` unsets the error
            matrix.
        '''

        # get axis id from an alias
        axis = self.get_axis(axis)

        if mat is not None:
            try:
                mat = np.asmatrix(mat)  # try to cast argument to a matrix
            except:
                raise TypeError(
                    "Cast to matrix failed. "
                    "Object was of type `%s'"
                    % (type(mat),)
                )

        # check if the covariance matrix is singular
        # and set/unset a flag accordingly
        try:
            mat.I  # try to invert it
        except:
            # if that fails, mat is singular
            self.__query_cov_mats_regular[axis] = True
        else:
            # else, mat is regular
            self.__query_cov_mats_regular[axis] = False

        # check if the matrix is zero or None and set/unset a flag accordingly
        if mat is None or (mat == 0).all():
            self.__query_has_errors[axis] = False
            self.__query_has_correlations[axis] = False
        # check if matrix is diagonal
        elif (np.diag(np.diag(mat)) == mat).all():
            self.__query_has_errors[axis] = True
            self.__query_has_correlations[axis] = False
        else:
            self.__query_has_errors[axis] = True
            self.__query_has_correlations[axis] = True

        # set the matrix
        if mat is None:
            self.cov_mats[axis] = np.asmatrix(
                np.zeros((self.get_size(), self.get_size()))
            )
        else:
            self.cov_mats[axis] = mat

    # Get methods
    ##############

    def get_axis(self, axis_alias):
        '''
        Get axis id from an alias.

        **axis_alias** : string or int
            Alias of the axis whose id should be returned. This is for example
            either ``'0'`` or ``'x'`` for the `x`-axis (id 0).
        '''

        try:
            axis = self.__axis_alias[axis_alias]
        except:
            raise SyntaxError("Unknown axis %s." % (axis_alias, ))

        return axis

    def get_size(self):
        '''
        Get the size of the `Dataset`. This is equivalent to the length of the
        `x`-axis data.
        '''

        if self.data[0] is None:
            return 0
        else:
            return len(self.data[0])

    def get_data_span(self, axis, include_error_bars=False):
        '''
        Get the data span for an axis. The data span is a tuple (`min`, `max`)
        containing the smallest and highest coordinates for an axis.

        **axis** : ``'x'`` or ``'y'``
            Axis for which to get the data span.

        *include_error_bars* : boolean (optional)
            ``True`` if the returned span should be enlarged to
            contain the error bars of the smallest and largest datapoints
            (default: ``False``)
        '''

        # get axis id from an alias
        axis = self.get_axis(axis)

        max_error_bar_size = 0.
        min_error_bar_size = 0.

        # get the index of the min and max datapoints
        max_idx = tuple(self.get_data(axis)).index(max(self.get_data(axis)))
        min_idx = tuple(self.get_data(axis)).index(min(self.get_data(axis)))

        if include_error_bars:
            # get the error of the min and max datapoints
            max_error_bar_size = np.sqrt(
                self.get_cov_mat(axis)[max_idx, max_idx]
            )
            min_error_bar_size = np.sqrt(
                self.get_cov_mat(axis)[min_idx, min_idx]
            )

        return [
            self.get_data(axis)[min_idx] - min_error_bar_size,
            self.get_data(axis)[max_idx] + max_error_bar_size
        ]

    def get_data(self, axis):
        '''
        Get the measurement data for an axis.

        **axis** : string
            Axis for which to get the measurement data. Can be ``'x'`` or
            ``'y'``.
        '''

        # get axis id from an alias
        axis = self.get_axis(axis)

        return self.data[axis]

    def get_cov_mat(self, axis, fallback_on_singular=None):
        '''
        Get the error matrix for an axis.

        **axis** :  ``'x'`` or ``'y'``
            Axis for which to load the error matrix.

        *fallback_on_singular* : `numpy.matrix` or string (optional)
            What to return if the matrix is singular. If this is ``None``
            (default), the matrix is returned anyway. If this is a
            `numpy.matrix` object or similar, that is returned istead.
            Alternatively, the shortcuts ``'identity'`` or ``1`` and ``'zero'``
            or ``0`` can be used to return the identity and zero matrix
            respectively.
        '''

        # get axis id from an alias
        axis = self.get_axis(axis)
        if fallback_on_singular is None:
            return self.cov_mats[axis]
        else:
            # if matrix is singular
            if not self.__query_cov_mats_regular[axis]:
                try:  # try to cast to matrix
                    fallback_matrix = np.asmatrix(fallback_on_singular)
                except:
                    if (
                        fallback_on_singular == 'identity' or
                        fallback_on_singular == 1
                    ):
                        fallback_matrix = np.eye(self.get_size())
                    elif (
                        fallback_on_singular == 'zero' or
                        fallback_on_singular == 0
                    ):
                        fallback_matrix = np.zeros(
                            (self.get_size(), self.get_size())
                        )
                    elif fallback_on_singular == 'report':
                        logger.warning(
                            "Warning: Covariance matrix for axis %s is "
                            "singular!" % (axis,)
                        )
                        # if not, return the (regular) matrix itself
                        return self.cov_mats[axis]
                    else:
                        raise SyntaxError(
                            "Cannot interpret fallback "
                            "matrix specification `%s`"
                            % (fallback_matrix,)
                        )

                return fallback_matrix  # return the fallback matrix
            else:
                # if not, return the (regular) matrix itself
                return self.cov_mats[axis]

    # Other methods
    ################

    def cov_mat_is_regular(self, axis):
        '''
        Returns `True` if the covariance matrix for an axis is regular and
        ``False`` if it is singular.

        **axis** : ``'x'`` or ``'y'``
            Axis for which to check for regularity of the covariance matrix.

        '''

        # get axis id from alias
        axis = self.get_axis(axis)

        return self.__query_cov_mats_regular[axis]

    def has_correlations(self, axis):
        '''
        Returns `True` if the specified axis has correlation data, ``False`` if
        not.

        **axis** :  ``'x'`` or ``'y'``
            Axis for which to check for correlations.
        '''

        # get axis id from alias
        axis = self.get_axis(axis)
        return self.__query_has_correlations[axis]

    def has_errors(self, axis):
        '''
        Returns `True` if the specified axis has statistical error data.

        **axis** :  ``'x'`` or ``'y'``
            Axis for which to check for error data.

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

        Here, the ``x_i`` and ``y_i`` represent the measurement data, the
        ``sigma_?_i`` are the statistical uncertainties of each data point, and
        the ``cor_?_ij`` are the correlation coefficients between the *i*-th
        and *j*-th data point.

        If the ``x`` or ``y`` errors are not correlated, then the entire
        correlation coefficient matrix can be omitted. If there are no
        statistical uncertainties for an axis, the second column can also be
        omitted. A blank line is required at the end of each data block!

        *format_string* : string (optional)
            A format string with which each entry will be rendered. Default is
            ``'.06e'``, which means the numbers are represented in scientific
            notation with six significant digits.

        *delimiter* : string (optional)
            A delimiter used to separate columns in the output.

        '''

        output_list = []

        # go through the axes
        for axis in xrange(self.n_axes):
            # define a helper list which we will fill out
            helper_list = []
            # get the statistical errors of the data
            stat_errs = extract_statistical_errors(self.get_cov_mat(axis))
            data = self.get_data(axis)
            # try to get a correlation matrix
            try:
                cor_mat = cov_to_cor(self.get_cov_mat(axis))
            except ZeroDivisionError:
                # if it fails, this means there are no
                # errors for the axis, so return
                # a zero matrix
                sz = self.get_size()
                cor_mat = np.asmatrix(np.zeros((sz, sz)))

            # add section title as a comment
            helper_list.append([
                '# axis %d: %s'
                % (axis, self.axis_labels[axis])
            ])
            # add a row for headings
            helper_list.append(['# datapoints'])

            # if the dataset has stat errors
            if self.__query_has_errors[axis]:
                # add a heading for second column
                helper_list[-1].append('uncor. err.')
                # if there are also correlations (syst errors)
                if self.__query_has_correlations[axis]:
                    # add a heading for the correlation matrix
                    helper_list[-1].append('correlation coefficients')

            for idx, val in enumerate(data):
                # append a new "row" to the helper list
                helper_list.append([])
                # append the coordinate of the data point
                helper_list[-1].append(format(val, format_string))
                # if the dataset has stat errors
                if self.__query_has_errors[axis]:
                    # append the stat error of the data point
                    helper_list[-1].append(
                        format(stat_errs[idx], format_string)
                    )
                    # if there are also correlations (syst errors)
                    if self.__query_has_correlations[axis]:
                        # go through the columns of the correlation matrix
                        for col in xrange(idx):
                            # append corr. coefficients to the helper list
                            helper_list[-1].append(
                                format(cor_mat[idx, col], format_string)
                            )

            helper_list.append([])  # append an empty list -> blank line
            output_list.append(helper_list)

        # turn the list into a string
        tmp_string = ''
        for row in output_list:
            for entry in row:
                tmp_string += join(entry, delimiter) + '\n'

        return tmp_string

    def write_formatted(self, file_path, format_string=".06e", delimiter='\t'):
        '''
        Writes the dataset to a plain-text file. For details on the format, see
        `get_formatted`_.

        **file_path** : string
            Path of the file object to write. **WARNING**: *overwrites existing
            files*!

        *format_string* : string (optional)
            A format string with which each entry will be rendered. Default is
            ``'.06e'``, which means the numbers are represented in scientific
            notation with six significant digits.

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
            # try to read the lines of the file
            tmp_lines = input_file.readlines()
        # this will fail if a file path string was passed, so alternatively:
        except AttributeError:
            # open the file pointed to by the path
            tmp_file = open(input_file, 'r')
            # and then read the lines of the file
            tmp_lines = tmp_file.readlines()

        # Parse the file
        #################

        tmp_data = []       # append data progressively to this list
        tmp_errors = []     # append error data progressively to this list
        tmp_cormat = []     # temporary storage of the correlation matrix

        tmp_rownumber = 0   # keep track of the current (error matrix) row
        tmp_axis = 0        # keep track of the current axis
        tmp_linenumber = 0  # keep track of the current line being read

        tmp_has_stat_errors = True  # assume statistical errors
        tmp_has_syst_errors = True  # assume correlations

        # don't assume that the file begins with a data block
        tmp_reading_data_block = False

        # go through the lines of the file
        for line in tmp_lines:

            tmp_linenumber += 1                 # update the line number

            if '#' in line:
                line = split(line, '#')[0]      # ignore anything after
                                                # a comment sign (#)

            if (not line) or (line.isspace()):  # if empty line encountered
                # if currenty reading a data block,
                # end reading it and commit data
                if tmp_reading_data_block:

                    # Commit the parsed data to the object
                    #######################################

                    # commit measurement data
                    self.set_data(tmp_axis, tmp_data)

                    if tmp_has_syst_errors:  # if there is a correlation matrix
                        # Turn the lists into a lower triangle matrix
                        tmp_cormat = zero_pad_lower_triangle(tmp_cormat)

                        # Symmetrize: copy the lower triangle to the upper half
                        tmp_cormat = make_symmetric_lower(tmp_cormat)

                        # commit covariance matrix
                        self.set_cov_mat(
                            tmp_axis,
                            cor_to_cov(tmp_cormat, tmp_errors)
                        )

                    # if there are just statistical errors
                    elif tmp_has_stat_errors:
                        # commit covariance matrix
                        self.set_cov_mat(
                            tmp_axis,
                            np.asmatrix(np.diag(tmp_errors)**2)
                        )
                    else:  # if there are no errors
                        self.set_cov_mat(tmp_axis, None)  # unset cov mat

                    # Reset temporary variables
                    #############################

                    tmp_data = []    # append data progressively to this list
                    tmp_errors = []  # append error data progressively to
                    tmp_cormat = []  # temp storage of the correlation matrix
                    tmp_has_stat_errors = True  # assume statistical errors
                    tmp_has_syst_errors = True  # assume correlations
                    tmp_rownumber = 0  # reset row number
                    tmp_axis += 1      # go to next axis
                    tmp_reading_data_block = False  # end data block

            else:  # else, if line is not empty, it must contain data

                if not tmp_reading_data_block:
                    # attest that we are currently reading a data block
                    tmp_reading_data_block = True

                # get the entries on the line as a list (whitespace-delimited)
                tmp_fields = split(line)

                # if there is only one entry,
                # we know it's just the measurement data
                if len(tmp_fields) == 1:
                    if tmp_has_stat_errors:
                        tmp_has_stat_errors = False  # no stat errs for axis
                    if tmp_has_syst_errors:
                        tmp_has_syst_errors = False  # no syst errs for axis

                # first field is the coordinate of the data point
                tmp_data.append(float(tmp_fields[0]))

                if tmp_has_stat_errors:
                    # second field is the error in that coordinate
                    tmp_errors.append(float(tmp_fields[1]))

                # if there are correlations
                if tmp_has_syst_errors:
                    # other fields are correlation coefficients
                    # (add 1.0 on main diagonal)
                    tmp_cormat.append(map(float, tmp_fields[2:]) + [1.0])

                # if there are not enough entries
                # for a valid correlation matrix
                if len(tmp_fields) != tmp_rownumber+2:
                    # attest that there is not a valid correlation matrix
                    tmp_has_syst_errors = False

                # update row number
                tmp_rownumber += 1

        # If EOF has been reached, commit data, if any
        if tmp_reading_data_block:

            # Commit the parsed data to the object
            #######################################

            # commit measurement data
            self.set_data(tmp_axis, tmp_data)

            if tmp_has_syst_errors:  # if there is a correlation matrix
                # Turn the lists into a lower triangle matrix
                tmp_cormat = zero_pad_lower_triangle(tmp_cormat)

                # Symmetrize: copy the lower triangle to the upper half
                tmp_cormat = make_symmetric_lower(tmp_cormat)

                # commit covariance matrix
                self.set_cov_mat(
                    tmp_axis,
                    cor_to_cov(tmp_cormat, tmp_errors)
                )

            # if there are just statistical errors
            elif tmp_has_stat_errors:
                # commit covariance matrix
                self.set_cov_mat(
                    tmp_axis,
                    np.asmatrix(np.diag(tmp_errors)**2)
                )
            else:  # if there are no errors
                self.set_cov_mat(tmp_axis, None)  # unset cov mat

        return True
