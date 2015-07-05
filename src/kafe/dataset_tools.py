'''
.. module:: dataset_tools
   :platform: Unix
   :synopsis: This sub-module defines some helper functions for creating a
       `Dataset` object.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

'''

# Changes:
# GQ 140724: fixed output format: uncor -> total
# DS 150610: add ErrorSource object
# ---------------------------------------------

from .dataset import Dataset

import numpy as np

NUMBER_OF_AXES = 2

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')


def build_dataset(xdata, ydata, cov_mats=None,
                  xabserr=0.0, xrelerr=0.0, xabscor=0.0, xrelcor=0.0,
                  yabserr=0.0, yrelerr=0.0, yabscor=0.0, yrelcor=0.0,
                  title=None,
                  axis_labels=None, axis_units=None, **kwargs):
    '''
    This helper function creates a `Dataset` from a series of keyword
    arguments.

    Parameters
    ----------

    **xdata** : list/tuple/`np.array` of floats
        This keyword argument is mandatory and should be an iterable
        containing *x*-axis the measurement data.

    **ydata** : list/tuple/`np.array` of floats
        This keyword argument is mandatory and should be an iterable
        containing *y*-axis the measurement data.

    *cov_mats* : ``None`` or 2-tuple, optional
        This argument defaults to ``None``, which means no covariance matrices
        are used. If covariance matrices are needed, a tuple with two entries
        (the first for `x` covariance matrices, the second for `y`) must be
        passed.

        Each element of this tuple may be either ``None`` or a NumPy matrix
        object containing a covariance matrix for the respective axis.

    Keyword Arguments
    -----------------

    error specification keywords : iterable or numeric (see below)
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

        >>> my_dataset = build_dataset(..., yabserr=0.3, yrelcor=0.1)

        creates a Dataset with an uncorrelated error of 0.3 for each `y`
        coordinate and a fully correlated (systematic) error of `y` of 0.1.

    title : string, optional
        The title of the `Dataset`.

    axis_labels : 2-tuple of strings, optional
        a 2-tuple containing the axis labels for the `Dataset`. This is
        relevant when plotting `Fits` of the `Dataset`, but is ignored when
        plotting more than one `Fit` in the same `Plot`.

    axis_units : 2-tuple of strings, optional
        a 2-tuple containing the axis units for the `Dataset`. This is
        relevant when plotting `Fits` of the `Dataset`, but is ignored when
        plotting more than one `Fit` in the same `Plot`.

    Returns
    -------

    ::py:class:`~kafe.dataset.Dataset`
        `Dataset` object constructed from data and error information
    '''

    # cast data to array
    data = (np.asarray(xdata), np.asarray(ydata))
    size = len(xdata)

    basename = kwargs.pop('basename', None)

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

    #
    # Construct cor mats from error specifications
    #

    error_keywords = {'xabserr': xabserr, 'xrelerr': xrelerr,
                      'xabscor': xabscor, 'xrelcor': xrelcor,
                      'yabserr': yabserr, 'yrelerr': yrelerr,
                      'yabscor': yabscor, 'yrelcor': yrelcor}

    # go through the keyword arguments
    for key, val in error_keywords.iteritems():

        err_spec = key
        err_val = val

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

    _dataset = Dataset(data=data,
                       title=title,
                       axis_labels=axis_labels, axis_units=axis_units,
                       basename=basename)

    _dataset.add_error_source('x', 'matrix', cov_mats[0])
    _dataset.add_error_source('y', 'matrix', cov_mats[1])

    return _dataset
