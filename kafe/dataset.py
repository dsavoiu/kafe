'''
.. module:: dataset
   :platform: Unix
   :synopsis: This sub-module defines a `Dataset` object, a container class for
        storing measurement data and error data. It also provides functions
        to build a Dataset object from pyhton arrays or read from or export
        data to a file

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

'''

# Changes:
# GQ 140724: fixed output format: uncor -> total
# DS 150610: add ErrorSource object
# ---------------------------------------------

from string import join, split

import numpy as np
from scipy.linalg import LinAlgError
import os

from numeric_tools import cov_to_cor, cor_to_cov, extract_statistical_errors, \
    zero_pad_lower_triangle, make_symmetric_lower

NUMBER_OF_AXES = 2

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')

class ErrorSource(object):
    '''
    This object stores the error information for a :py:obj:`Dataset` as a
    *covariance matrix* :math:`C` (sometimes also referred to as the *error
    matrix*). This has several advantages: it allows calculating the function
    to minimize (e.g. the chi-square) for a fit as a matrix product, and it
    allows specifying multiple error sources for a `Dataset` by simply adding
    up the corresponding matrices.

    The object contains methods to generate a covariance matrix for some
    simple cases, such as when all points have the same relative or absolute
    errors and the errors are either not correlated or fully correlated. For
    more complicated error models, a covariance matrix can be specified
    directly.
    '''

    def __init__(self):
        self.size = None            # undefined size
        self.error_type = None      # undefined type ('simple' or 'matrix')
        self.error_value = None     # float/floats or the covariance matrix

        # Boolean flags
        self.has_correlations = False  # assume no correlations

    def make_from_matrix(self, cov_mat, check_singular=False):
        """
        Sets the covariance matrix manually.

        Parameters
        ----------

        **cov_mat** : numpy.matrix
            A *square*, *symmetric* (and usually *regular*) matrix.

        Keyword Arguments
        -----------------

        check_singular : boolean, optional
            Whether to force singularity check. Defaults to ``False``.
        """

        # Check matrix suitability
        _mat = np.asmatrix(cov_mat)  # cast to matrix
        _shp = _mat.shape
        if not _shp[0] == _shp[1]:   # check square shape
            raise ValueError("Failed to make ErrorSource: matrix must be "
                               "square, got shape %r" % (_shp,))
        if (_mat == _mat.T).all():  # check if symmetric
            if check_singular:
                try:
                    _mat.I  # try to invert matrix
                except LinAlgError:
                    raise ValueError("Failed to make ErrorSource: singular "
                                       "matrix!")
        else:
            raise ValueError("Failed to make ErrorSource: covariance "
                               "matrix not symmetric")

        self.error_type = 'matrix'
        self.error_value = _mat
        self.size = _shp[0]
        self.has_correlations = (np.diag(np.diag(_mat)) == _mat).all()


    def make_from_val(self, err_val, fully_correlated=False):
        """
        Sets information required to construct the covariance matrix.

        Parameters
        ----------

        **err_val** : float or sequence of floats
            If all data points have the same uncertainty

        Keyword Arguments
        -----------------

        fully_correlated : boolean, optional
            Whether the errors are fully correlated. Defaults to ``False``.
        """
        try:
            iter(err_val)
        except TypeError:
            # object is not iterable, assume single float
            self.size = None
        else:
            # object is not iterable, assume sequence of floats
            self.size = len(err_val)  # store length

        self.error_type = 'simple'

        # set correlation flag
        if fully_correlated:
            self.has_correlations = True
        else:
            self.has_correlations = False

        self.error_value = err_val  # float or sequence of floats


    ## GET Methods ##

    def get_matrix(self, size=None):
        """
        Returns/Generates the covariance matrix for this ErrorSource.

        If the user specified the matrix using
        :py:meth:`~kafe.dataset.ErrorSource.make_from_matrix`,
        returns that matrix. If a simple error model is specified, a matrix is
        constructed as follows:

        For *uncorrelated* errors, the covariance matrix is always diagonal.

        If a single float :math:`\\sigma` is given as the error, the diagonal
        entries will be equal to :math:`\\sigma^2`. In this case, the matrix
        size needs to be specified via the ``size`` parameter.

        If a list of floats :math:`\\sigma_i` is given as the error, the
        *i*-th entry will be equal to :math:`{\\sigma_i}^2`. In this case,
        the size of the matrix is inferred from the size of the list.

        For *fully correlated* errors, the covariance matrix is the outer
        product of the error array :math:`\\sigma_i` with itself, i.e. the
        :math:`(i,j)`-th matrix entry will be equal to
        :math:`\\sigma_i\\sigma_j`.

        Keyword Arguments
        -----------------

        size : int (sometimes required)
            Size of the matrix to return. Only relevant if the error value
            is a single float, since in that case there is no way to deduce
            the matrix size.
        """

        if self.error_type == 'matrix':
            if size is not None and self.size != size:
                # 'matrix'-type errors are fixed-size -> warn about mismatch
                logger.warning("Ignoring requested size %d of "
                               "covariance matrix. Does not match "
                               "`matrix'-type error size %d"
                               % (size, self.size))
            return self.error_value

        elif self.error_type == 'simple':
            # single float case
            if isinstance(self.error_value, float):
                if size is None:
                    raise ValueError("Cannot generate covariance matrix "
                                     "for this simple error model without "
                                     "a specified size.")
                else:
                    # turn float value into array of identical values
                    _val = np.ones(size) * self.error_value
            # error list case
            else:
                # check if iterable
                try:
                    iter(self.error_value)
                except:
                    raise ValueError, ("Given error `%r' is not iterable."
                                       % (self.error_value,))
                else:
                    # use value given
                    _val = np.asarray(self.error_value)  # size is implicit
                    if size is not None and len(_val) != size:
                        # 'simple'-type error lists are fixed-size -> warn
                        logger.warning("Ignoring requested size %d of "
                                       "covariance matrix. Does not match "
                                       "`simple'-type error list length %d"
                                       % (size, self.size))

            # generate covariance matrix
            if self.has_correlations:
                _mat = np.outer(_val, _val)
            else:
                _mat = np.matrix(np.diag(_val ** 2))

            return _mat

        else:
            raise ValueError, "Unknown error type `%s'" % (self.error_type,)


class Dataset(object):
    '''
    The `Dataset` object is a data structure for storing measurement and error
    data.

    It contains the measurement `data` as *NumPy* arrays and the error
    information as a list of :py:class:`~kafe.dataset.ErrorSource` objects for
    each axis, each of which represents a separate contribution to the
    uncertainty of the measurement data, expressed as a *covariance matrix*.

    The `Dataset` object calculates a *total covariance matrix* by adding
    up all the individual `ErrorSource` covariance matrices. This
    *total covariance matrix* is the one used for fitting.

    A `Dataset` can be constructed directly from the measurement data, and can
    optionally be given a *title*, *axis labels* and *axis units*, as well as
    a *base name* for log or output files:

    >>> my_d = kafe.Dataset(data=[[0., 1., 2.], [1.23, 3.45, 5.62]])

    After constructing the `Dataset`, an error model may be added using
    :py:meth:`~kafe.dataset.Dataset.add_error_source` (here, an absolute
    *y*-uncertainty of 0.5):

    >>> my_d.add_error_source('y', 'simple', 0.5)  # y errors, all +/- 0.5

    The `Dataset` may then be used for fitting. For more information, see the
    :py:class:`~kafe.fit.Fit` object documentation.

    Keyword Arguments
    -----------------

    data : iterable, optional
        the measurement data. Either of the form (xdata, ydata) or
        [(x1, y1), (x2, y2),... (xn, yn)]

    title : string, optional
        the name of the `Dataset`. If omitted, the `Dataset` will be given the
        generic name 'Untitled Dataset'.

    axis_labels : list of strings, optional

        labels for the `x` and `y` axes. If omitted, these will be set to
        ``'x'`` and ``'y'``, respectively.

    axis_units : list of strings, optional

        units for the `x` and `y` axes. If omitted, these will be assumed to be
        dimensionless, i.e. the unit will be an empty string.

    basename : string

        base name of files generated by this Dataset/subsequent Fits...
    '''

    def __init__(self, data=None,
                       title="Untitled Dataset",
                       axis_labels=['x', 'y'], axis_units=['', ''], **kwargs):

        '''Create a Dataset'''

        # Definitions
        ##############

        self.__n_axes = 2
        """dimensionality of the `Dataset`. Currently, only 2D `Datasets` are
        supported"""
        self.n_datapoints = 0   #: number of data points in the `Dataset`
        self.data = [None, None]
        #: list containing measurement data (axis-ordering)

        self.cov_mats = [None, None]             #: covariance matrices for axes
        self.__cov_mat_up_to_date = False        # flag need to compute matrix

        self.err_src = [[], []]                  #: lists of ErrorSource objects
        self.__query_err_src_enabled = [[], []]  # ErrorSource objects enabled?
        self.__query_err_src_relative = [[], []] # ErrorSources relative?

        # Metadata
        #: axis labels
        if axis_labels is not None:
            self.axis_labels = list(axis_labels)
        else:
            self.axis_labels = ['x', 'y']

        if axis_units is not None:
            self.axis_units = list(axis_units)
        else:
            self.axis_units = ['', '']

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
        self.data_label = title

        # set the basename
        self.basename = kwargs.pop('basename', None)

        # check for deprecated input_file keyword and warn
        _input_file = kwargs.pop('input_file', None)
        if _input_file is not None:
            logger.warning("'input_file' argument to Dataset() is deprecated."
                           "Use Dataset.read_from_file() instead." )
            if data is not None:
                raise Exception("Cannot provide both 'data' and 'input_file' "
                                "arguments to Dataset().")
            else:
                self.read_from_file(_input_file)
        else:
            # set data, if any
            if data is not None:
                self.set_data(data)

    # Data
    ##############

    def set_data(self, data):
        """
        Set the measurement data for both axes.

        Each element of **data** must be iterable and be of the same length.
        The first element of the **data** tuple/list is assumed to be the `x`
        data, and the second to be the `y` data:

        >>> my_dataset.set_data(([0., 1., 2.], [1.23, 3.45, 5.62]))

        Alternatively, *x*-*y* value pairs can also be passed as **data**. The
        following is equivalent to the above:

        >>> my_dataset.set_data(([0.0, 1.23], [1.0, 3.45], [2.0, 5.62]))

        In case the `Dataset` contains two data points, the ordering is
        ambiguous. In this case, the first ordering (`x` data first, then `y`
        data) is assumed.

        Parameters
        ----------

        *data* : iterable
            the measurement data. Either of the form (xdata, ydata) or
            [(x1, y1), (x2, y2),... (xn, yn)]
        """

        # Load data
        ############

        # preliminary checks
        if len(data) != NUMBER_OF_AXES:
            # in case of xy value pairs, transpose
            data = np.asarray(data).T.tolist()
            if len(data) != NUMBER_OF_AXES:
                raise Exception(
                    "Unsupported number of axes: %d"
                    % (len(data),)
                )
            else:
                # set the transposed data as the read data
                #data = data
                pass

        for axis in xrange(self.__n_axes):  # go through the axes
            self.set_axis_data(axis, data[axis])  # load data for axis

    def set_axis_data(self, axis, data):
        '''
        Set the measurement data for a single axis.

        Parameters
        ----------

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
            _da = np.asarray(data)
            if axis == 0:
                self.data[axis] = _da
                # set the dataset's size
                self.n_datapoints = len(_da)
            else:
                # check the data size
                if len(_da) == self.n_datapoints:  # size mismatch
                    self.data[axis] = _da
                else:
                    raise ValueError("Cannot set data for axis %d. "
                                     "Size mismatch: expected %d, got %d."
                                     % (axis, self.n_datapoints, len(_da)))


    # Uncertainties
    ################

    def add_error_source(self, axis, err_type, err_val, relative=False,
                         correlated=False, recompute_cov_mat=True):
        """
        Add an error source for the data. A `Dataset` can have many
        error sources for each axis, each corresponding to a covariance matrix.
        The total error model for the axis is represented by the sum of
        these matrices.

        Note: whenever an ErrorSource is added, the total covariance matrix
        is (re-)calculated, unless *recompute_cov_mat* is ``False``.

        Parameters
        ----------

        **axis** : ``'x'`` or ``'y'``
            axis for which to add error source.

        **err_type**: ``'simple'`` or ``'matrix'``
            a ``'simple'`` error source is constructed from a single float or
            a list of *N* floats (*N* being the size of the `Dataset`),
            representing the uncertainty of the corresponding data points.

            A ``'matrix'`` error source is a user-constructed covariance
            matrix.

        **err_val**: float/list of floats *or* numpy.matrix
            for a ``'simple'`` error source, a float of a list of *N* floats
            (*N* being the size of the `Dataset`). The float/each float in the
            list represents the uncertainty of the corresponding data point.

            For a ``'matrix'`` error source, the user-constructed covariance
            matrix (type: `numpy.matrix`).

        Keyword Arguments
        -----------------

        relative: boolean, optional, default ``False``
            errors relative to the data (``True``) or absolute (``False``).

        correlated: boolean, optional, default ``False``
            errors fully correlated (``True``) or totally uncorrelated
            (``False``).

        recompute_cov_mat: boolean, optional, default ``True``
            recalculate the total covariance matrix after adding the error
            source

        Returns
        -------

        int
            this integer may later be used to remove or disable/enable the
            error source using
            :py:meth:`~kafe.dataset.Dataset.remove_error_source`,
            :py:meth:`~kafe.dataset.Dataset.disable_error_source` or
            :py:meth:`~kafe.dataset.Dataset.enable_error_source`.
        """

        # get axis id from an alias
        axis = self.get_axis(axis)

        # initialize ErrorSource
        _es = ErrorSource()

        # specify type of ErrorSource
        if err_type == 'simple':
            _es.make_from_val(err_val, fully_correlated=correlated)
        elif err_type == 'matrix':
            if correlated:
                logger.warn("Ignoring 'correlated' when adding a 'matrix' "
                            "error source. Correlation information is "
                            "contained within the matrix itself")
            _es.make_from_matrix(err_val)

        self.err_src[axis].append(_es)  # add ErrorSource
        self.__query_err_src_enabled[axis].append(True)  # enable ErrorSource
        self.__query_err_src_relative[axis].append(relative)  # relative flag
        self.__cov_mat_up_to_date = False  # flag need to recompute matrix

        if recompute_cov_mat:
            self.calc_cov_mats(axis)  # recompute covariance matrix

        # return error source ID
        _err_src_id = len(self.err_src[axis]) - 1
        return _err_src_id

    def remove_error_source(self, axis, err_src_id, recompute_cov_mat=True):
        """
        Remove the error source from the `Dataset`.

        Parameters
        ----------

        **axis** : ``'x'`` or ``'y'``
            axis for which to add error source.

        **err_src_id** : int
            error source ID, as returned by
            :py:meth:`~kafe.dataset.Dataset.add_error_source`.

        Keyword Arguments
        -----------------

        recompute_cov_mat: boolean, optional, default ``True``
            recalculate the total covariance matrix after removing the error
            source
        """

        # get axis id from an alias
        axis = self.get_axis(axis)

        self.err_src[axis][err_src_id] = None  # remove ErrorSource
        self.__query_err_src_enabled[axis][err_src_id] = False
        self.__query_err_src_relative[axis][err_src_id] = False
        self.__cov_mat_up_to_date = True  # flag need to recompute matrix

        if recompute_cov_mat:
            self.calc_cov_mats(axis)  # recompute covariance matrix

    def disable_error_source(self, axis, err_src_id):
        """
        Disables an ErrorSource by excluding it from the calculation of the
        total covariance matrix.

        Parameters
        ----------

        **axis** : ``'x'`` or ``'y'``
            axis for which to add error source.

        **err_src_id** : int
            error source ID, as returned by
            :py:meth:`~kafe.dataset.Dataset.add_error_source`.
        """
        # get axis id from an alias
        axis = self.get_axis(axis)

        self.__query_err_src_enabled[axis][err_src_id] = False

    def enable_error_source(self, axis, err_src_id):
        """
        Enables an ErrorSource by excluding it from the calculation of the
        total covariance matrix.

        Parameters
        ----------

        **axis** : ``'x'`` or ``'y'``
            axis for which to add error source.

        **err_src_id** : int
            error source ID, as returned by
            :py:meth:`~kafe.dataset.Dataset.add_error_source`.

        """
        # get axis id from an alias
        axis = self.get_axis(axis)

        self.__query_err_src_enabled[axis][err_src_id] = True

    def calc_cov_mats(self, axis='all'):
        """
        (Re-)Calculate the covariance matrix from the enabled error sources.

        Keyword Arguments
        -----------------

        axis : ``'x'`` or ``'y'`` or ``'all'``
            axis/axes for which to (re-)calcuate covariance matrix.
        """
        _size = self.n_datapoints
        _mats = [np.matrix(np.zeros((_size, _size))),
                 np.matrix(np.zeros((_size, _size)))]

        if axis is 'all':
            _axes_list = range(self.__n_axes)
        else:
            _axes_list = [self.get_axis(axis)]

        #print _axes_list

        for _axis in _axes_list:  # go through the axes
            for _idx, _es in enumerate(self.err_src[_axis]):  # go through the ErrorSources
                # skip removed error sources
                if _es is None:
                    continue

                # skip disabled error sources
                if not self.__query_err_src_enabled[_axis][_idx]:
                    continue

                if _es.size is not None:
                    # if ErrorSource size fixed
                    if _es.size == _size:
                        # if ErrorSource size matches Dataset size
                        _mat = _es.get_matrix()  # OK to get matrix
                    else:
                        # shouldn't happen for ErrorSources added with
                        # add_error_source(), but still...
                        raise ValueError("ErrorSource fixed size %d doesn't "
                                         "match Dataset size %d"
                                         % (_es.size, _size))
                else:
                    # get cov mat with specified size
                    _mat = _es.get_matrix(size=_size)

                if self.__query_err_src_relative[_axis][_idx]:
                    # for relative errors, "multiply" covariance matrix by data
                    _data = self.get_data(_axis)
                    _mat = np.asmatrix(np.asarray(_mat) *
                                       np.outer(_data, _data))

                _mats[_axis] += _mat  # add covariance matrix

        # set cov mats for all axes
        for _axis in _axes_list:  # go through the axes
            self.set_cov_mat(_axis, _mats[_axis])

        self.__cov_mat_up_to_date = True

    def set_cov_mat(self, axis, mat):
        '''
        Forcibly set the error matrix for an axis, ignoring :py:class:`~kafe.dataset.ErrorSource`
        objects. This is useful for adjusting the covariance matrix during the
        fit process.

        Parameters
        ----------

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
            self.__query_cov_mats_regular[axis] = False
        else:
            # else, mat is regular
            self.__query_cov_mats_regular[axis] = True

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

        # forced matrices considered not up to date
        self.__cov_mat_up_to_date = False

    # Get methods
    ##############

    def get_axis(self, axis_alias):
        '''
        Get axis id from an alias.

        Parameters
        ----------

        **axis_alias** : string or int
            Alias of the axis whose id should be returned. This is for example
            either ``'0'`` or ``'x'`` for the `x`-axis (id 0).

        Returns
        -------

        int
            the axis ID
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

        Returns
        -------

        int
            the number of datapoints in the `Dataset`.
        '''

        if self.data[0] is None:
            return 0
        else:
            return len(self.data[0])

    def get_data_span(self, axis, include_error_bars=False):
        '''
        Get the data span for an axis. The data span is a tuple (`min`, `max`)
        containing the smallest and highest coordinates for an axis.

        Parameters
        ----------

        **axis** : ``'x'`` or ``'y'``
            Axis for which to get the data span.

        Keyword Arguments
        -----------------

        include_error_bars : boolean, optional
            ``True`` if the returned span should be enlarged to
            contain the error bars of the smallest and largest datapoints
            (default: ``False``)

        Returns
        -------

        a tuple (`min`, `max`)
            the data span for the axis
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

        Parameters
        ----------

        **axis** : string
            Axis for which to get the measurement data. Can be ``'x'`` or
            ``'y'``.

        Returns
        -------

        *numpy.array*
            the measurement data for the axis
        '''

        # get axis id from an alias
        axis = self.get_axis(axis)

        return self.data[axis]

    def get_cov_mat(self, axis, fallback_on_singular=None):
        '''
        Get the error matrix for an axis.

        Parameters
        ----------

        **axis** :  ``'x'`` or ``'y'``
            Axis for which to load the error matrix.

        Keyword Arguments
        -----------------

        fallback_on_singular : `numpy.matrix` or string, optional
            What to return if the matrix is singular. If this is ``None``
            (default), the matrix is returned anyway. If this is a
            `numpy.matrix` object or similar, that is returned istead.
            Alternatively, the shortcuts ``'identity'`` or ``1`` and ``'zero'``
            or ``0`` can be used to return the identity and zero matrix
            respectively.

        Returns
        -------

        *numpy.matrix*
            the current covariance matrix
        '''

        # get axis id from an alias
        axis = self.get_axis(axis)
        _mat = self.cov_mats[axis]

        # compute and return zero matrix instead of ``None``
        if _mat is None:
            sz = self.get_size()
            _mat = np.asmatrix(np.zeros((sz, sz)))

        if fallback_on_singular is None:
            return _mat
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
                        return _mat
                    else:
                        raise SyntaxError(
                            "Cannot interpret fallback "
                            "matrix specification `%s`"
                            % (fallback_matrix,)
                        )

                return fallback_matrix  # return the fallback matrix
            else:
                # if not, return the (regular) matrix itself
                return _mat

    # Other methods
    ################

    def cov_mat_is_regular(self, axis):
        '''
        Returns `True` if the covariance matrix for an axis is regular and
        ``False`` if it is singular.

        Parameters
        ----------

        **axis** : ``'x'`` or ``'y'``
            Axis for which to check for regularity of the covariance matrix.

        Returns
        -------

        boolean
            ``True`` if covariance matrix is regular

        '''

        # get axis id from alias
        axis = self.get_axis(axis)

        return self.__query_cov_mats_regular[axis]

    def has_correlations(self, axis=None):
        '''
        Returns `True` if the specified axis has correlation data, ``False`` if
        not.

        Parameters
        ----------

        *axis* :  ``'x'`` or ``'y'`` or ``None``, optional
            Axis for which to check for correlations. If ``None``,
            returns true if there are correlations for at least one axis.

        Returns
        -------

        bool
            `True` if the specified axis has correlation data
        '''
        if axis is not None:
            # get axis id from alias
            axis = self.get_axis(axis)
            return self.__query_has_correlations[axis]
        else:
            return np.any(self.__query_has_correlations)

    def has_errors(self, axis=None):
        '''
        Returns `True` if the specified axis has any kind of error data.

        Parameters
        ----------

        *axis* :  ``'x'`` or ``'y'`` or ``None``, optional
            Axis for which to check for error data. If ``None``,
            returns true if there are errors for at least one axis.

        Returns
        -------

        bool
            `True` if the specified axis has any kind of error data.
        '''
        if axis is not None:
            # get axis id from alias
            axis = self.get_axis(axis)
            return self.__query_has_errors[axis]
        else:
            return np.any(self.__query_has_errors)


    def error_source_is_enabled(self, axis, err_src_id):
        """
        Returns ``True`` if an ErrorSource is enabled, that is if it is included
        in the total covariance matrix.

        Parameters
        ----------

        **axis** :  ``'x'`` or ``'y'``
            Axis for which to load the error matrix.

        **err_src_id** : int
            error source ID, as returned by
            :py:meth:`~kafe.dataset.Dataset.add_error_source`.

        Returns
        -------

        bool
            `True` if the specified error source is enables

        TODO: ##DocString##
        """
        return self.__query_err_src_enabled[axis][err_src_id]

    def get_formatted(self, format_string=".06e", delimiter='\t'):
        '''
        Returns the dataset in a plain-text format which is human-readable and
        can later be used as an input file for the creation of a new `Dataset`.

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

        Keyword Arguments
        -----------------

        format_string : string, optional
            A format string with which each entry will be rendered. Default is
            ``'.06e'``, which means the numbers are represented in scientific
            notation with six significant digits.

        delimiter : string, optional
            A delimiter used to separate columns in the output.

        Returns
        -------

        str
            a plain-text representation of the `Dataset`

        '''

        output_list = []

        # go through the axes
        for axis in xrange(self.__n_axes):
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
                # if there are also correlations (syst errors)
                if self.__query_has_correlations[axis]:
                    # add a heading for the correlation matrix
                    helper_list[-1].append('total err.')
                    helper_list[-1].append('correlation coefficients')
                else:
                    helper_list[-1].append('uncor. err.')

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
        :py:meth:`~kafe.dataset.Dataset.read_from_file`.

        Parameters
        ----------

        **file_path** : string
            Path of the file object to write. **WARNING**: *overwrites existing
            files*!

        Keyword Arguments
        -----------------

        format_string : string, optional
            A format string with which each entry will be rendered. Default is
            ``'.06e'``, which means the numbers are represented in scientific
            notation with six significant digits.

        delimiter : string, optional
            A delimiter used to separate columns in the output.

        '''

        # write the output of self.get_formatted to the file
        with open(file_path, 'w') as my_file:
            my_file.write(self.get_formatted(format_string))
            my_file.close()

    def read_from_file(self, input_file):
        '''
        Reads the `Dataset` object from a file.

        One way to construct a Dataset is to specify an input file containing
        a plain-text representation of the dataset:

        >>> my_dataset.read_from_file('/path/to/file')

        or

        >>> my_dataset.read_from_file(my_file_object)

        For details on the format, see
        :py:meth:`~kafe.dataset.Dataset.get_formatted`

        Parameters
        ----------

        **input_file** : str
            path to the file

        Returns
        -------

        boolean
            ``True`` if the read succeeded, ``False`` if not.
        '''
        try:
            # try to read the lines of the file
            tmp_lines = input_file.readlines()
        # this will fail if a file path string was passed, so alternatively:
        except AttributeError:
            # if not basename was provided in the arguments
            if self.basename is None:
                # get the basename from the path
                _basename = os.path.basename(input_file)
                # remove the last extension (usually '.dat')
                self.basename = '.'.join(_basename.split('.')[:-1])
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
                    self.set_axis_data(tmp_axis, tmp_data)

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
            self.set_axis_data(tmp_axis, tmp_data)

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

        # Turn covariance matrices into ErrorSource objects
        for axis in xrange(self.__n_axes):
            _mat = self.get_cov_mat(axis)

            # remove existing error model (all error sources)
            for err_src_id in xrange(len(self.err_src[axis])):
                self.remove_error_source(axis, err_src_id, recompute_cov_mat=False)

            if _mat is not None:
                # Replace error model with the computed matrix
                if self.err_src[axis]:
                    # Warn if error model is overwritten after call to read_from_file()
                    logger.warn("Overwriting existing error model for axis %d "
                                "of Dataset" % (axis,))
                # add error for axis as a single matrix error
                self.add_error_source(axis, 'matrix', _mat)
            else:
                if self.err_src[axis]:
                    # Warn if error model is removed after call to read_from_file()
                    logger.warn("Removing existing error model for axis %d "
                                "of Dataset" % (axis,))

        return True
