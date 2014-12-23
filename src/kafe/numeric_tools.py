'''
.. module:: numeric_tools
   :platform: Unix
   :synopsis: A submodule containing several numeric algorithms used by the
        fit package, such as methods for converting beween covariance and
        correlation matrices or extracting the statistical errors from a
        covariance matrix.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>

'''

## Changes:
#     07-Aug-14 GQ covariance matrix returned by Minuit contains zero-values
#                  for lines/colums corresponding to fixed parameters;
#                  made a special version of cov_to_cor,
#                  MinuitCov_to_cor for this case

import numpy as np

def cov_to_cor(cov_mat):
    r'''
    Converts a covariance matrix to a correlation matrix according to the
    formula

    .. math::

        \text{Cor}_{ij} = \frac{\text{Cov}_{ij}}
            {\sqrt{ \text{Cov}_{ii}\,\text{Cov}_{jj}}}

    **cov_mat** : `numpy.matrix`
        The covariance matrix to convert.
    '''

    diagonal = np.diag(cov_mat)  # extract the diagonal entries

    # a single zero in the diagonal makes the calculation impossible
    # Note: this can only happen if NO errors are defined for the
    #       axis in question
    if 0 in diagonal:
        raise ZeroDivisionError("Conversion to correlation matrix failed. \
                                 Zeroes present on error matrix diagonal...")

    # get the statistical error from the diagonal entries
    error_array = np.sqrt(diagonal)

    # construct the outer (dyadic) product between the error array
    # and itself. This is an numpy array, so '/' will stand for an
    # elementwise division
    stat_err_outer_prod = np.outer(error_array, error_array)

    # the correlation matrix is the original matrix divided
    # elementwise by the calculated matrix. Return a matrix.
    return np.asmatrix(np.asarray(cov_mat) / stat_err_outer_prod)



def MinuitCov_to_cor(cov_mat):
    r'''
    Converts a covariance matrix as returned by Minuit to the
    corresponding correlation matrix; note that the Minuit
    covariance matrix may contain lines/rows with zeroes if
    parameters are fixed

    **cov_mat** : `numpy.matrix`
        The Minuit covariance matrix to convert.
    '''

    err = np.sqrt(np.diag(cov_mat))  # extract the errors
    dim=len(err)
    cor_mat=np.zeros((dim,dim),np.float32)
    for i in range(0,dim):
      for j in range(0,dim):
          e2=err[i]*err[j]
          if e2 != 0. :
            cor_mat[i,j]=cov_mat[i,j]/e2
    return cor_mat

def cor_to_cov(cor_mat, error_list):
    r'''
    Converts a correlation matrix to a covariance matrix according to the
    formula

    .. math::

        \text{Cov}_{ij} = \text{Cor}_{ij}\, \sigma_i \, \sigma_j

    **cor_mat** : `numpy.matrix`
        The correlation matrix to convert.

    **error_list** : sequence of floats
        A sequence of statistical errors. Must be of the same length
        as the diagonal of `cor_mat`.

    '''

    # construct the outer (dyadic) product between the error array
    # and itself. This is an numpy array, so '*' will stand for an
    # elementwise multiplication
    stat_err_outer_prod = np.outer(error_list, error_list)

    # the covariance matrix is the correlation matrix multiplied
    # elementwise by the calculated array. Return a matrix.
    return np.asmatrix(np.asarray(cor_mat) * stat_err_outer_prod)


def extract_statistical_errors(cov_mat):
    '''
    Extracts the statistical errors from a covariance matrix. This means
    it returns the (elementwise) square root of the diagonal entries

    **cov_mat**
        The covariance matrix to extract errors from. Type: `numpy.matrix`
    '''

    diagonal = np.diag(cov_mat)  # extract the diagonal entries

    return np.sqrt(diagonal)  # return the array of square roots


def zero_pad_lower_triangle(triangle_list):
    '''
    Converts a list of lists into a lower triangle matrix. The list members
    should be lists of increasing length from 1 to N, N being the dimension of
    the resulting lower triangle matrix. Returns a `NumPy` matrix object.

    For example:

        >>> zero_pad_lower_triangle([ [1.0], [0.2, 1.0], [0.01, 0.4, 3.0] ])
        matrix([[ 1.  ,  0.  ,  0.  ],
                [ 0.2 ,  1.  ,  0.  ],
                [ 0.01,  0.4 ,  3.  ]])

    **triangle_list** : list
        A list containing lists of increasing length.

    returns : `numpy.matrix`
        The lower triangle matrix.
    '''

    dimension = len(triangle_list[-1])  # dimension of matrix given by last row
    result = []
    for row in triangle_list:
        result.append(row + (dimension - len(row)) * [0.0])

    return np.matrix(result)


def make_symmetric_lower(mat):
    '''
    Copies the matrix entries below the main diagonal to the upper triangle
    half of the matrix. Leaves the diagonal unchanged. Returns a `NumPy` matrix
    object.

    **mat** : `numpy.matrix`
        A lower diagonal matrix.

    returns : `numpy.matrix`
        The lower triangle matrix.
    '''

    # extract lower triangle from matrix (including diagonal)
    tmp_mat = np.tril(mat)

    # if the matrix given wasn't a lower triangle matrix, raise an error
    if (mat != tmp_mat).all():
        raise Exception('Matrix to symmetrize is not a lower diagonal matrix.')

    # add its transpose to itself, zeroing the diagonal to avoid doubling
    tmp_mat += np.triu(tmp_mat.transpose(), 1)

    return np.asmatrix(tmp_mat)
