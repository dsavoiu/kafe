"""
Unit tests for submodule ``numeric_tools``
"""

import numpy as np
from kafe import numeric_tools

#
# Reference objects for tests
#

# the reference values for these tests are taken from
# the W boson mass measurements featured in example 4

# the measurement uncertainties
REF_ERR_LIST = np.array(
    [ 0.05887274,  0.07596051,  0.07116882,  0.06168468,  0.0818352,
      0.10107918,  0.08955445,  0.08099383])

# the correlation matrix
REF_COR_MAT = np.matrix([
    [ 1.        ,  0.09861351,  0.10525302,  0.12143587,  0.12972557,
      0.10502775,  0.11854372,  0.13107317],
    [ 0.09861351,  1.        ,  0.08157573,  0.09411815,  0.10054303,
      0.08140113,  0.0918766 ,  0.10158748],
    [ 0.10525302,  0.08157573,  1.        ,  0.100455  ,  0.10731246,
      0.08688176,  0.09806253,  0.10842723],
    [ 0.12143587,  0.09411815,  0.100455  ,  1.        ,  0.12381195,
      0.10023999,  0.11313983,  0.12509812],
    [ 0.12972557,  0.10054303,  0.10731246,  0.12381195,  1.        ,
      0.23404724,  0.26416667,  0.29208771],
    [ 0.10502775,  0.08140113,  0.08688176,  0.10023999,  0.23404724,
      1.        ,  0.21387325,  0.23647854],
    [ 0.11854372,  0.0918766 ,  0.09806253,  0.11313983,  0.26416667,
      0.21387325,  1.        ,  0.26691085],
    [ 0.13107317,  0.10158748,  0.10842723,  0.12509812,  0.29208771,
      0.23647854,  0.26691085,  1.        ]])

# the covariance matrix
REF_COV_MAT = np.matrix([
    [ 0.003466,  0.000441,  0.000441,  0.000441,  0.000625,  0.000625,
      0.000625,  0.000625],
    [ 0.000441,  0.00577 ,  0.000441,  0.000441,  0.000625,  0.000625,
      0.000625,  0.000625],
    [ 0.000441,  0.000441,  0.005065,  0.000441,  0.000625,  0.000625,
      0.000625,  0.000625],
    [ 0.000441,  0.000441,  0.000441,  0.003805,  0.000625,  0.000625,
      0.000625,  0.000625],
    [ 0.000625,  0.000625,  0.000625,  0.000625,  0.006697,  0.001936,
      0.001936,  0.001936],
    [ 0.000625,  0.000625,  0.000625,  0.000625,  0.001936,  0.010217,
      0.001936,  0.001936],
    [ 0.000625,  0.000625,  0.000625,  0.000625,  0.001936,  0.001936,
      0.00802 ,  0.001936],
    [ 0.000625,  0.000625,  0.000625,  0.000625,  0.001936,  0.001936,
      0.001936,  0.00656 ]])



def test_cor_to_cov():
    """
    Test of numeric_tools.cor_to_cov.
    """
    global REF_ERR_LIST, REF_COV_MAT, REF_COR_MAT
    assert np.allclose(REF_COV_MAT, numeric_tools.cor_to_cov(REF_COR_MAT, REF_ERR_LIST))

def test_cov_to_cor():
    """
    Test of numeric_tools.cor_to_cov.
    """
    global REF_COV_MAT, REF_COR_MAT
    assert np.allclose(REF_COR_MAT, numeric_tools.cov_to_cor(REF_COV_MAT))

def test_extract_statistical_errors():
    """
    Test of numeric_tools.cor_to_cov.
    """
    global REF_ERR_LIST, REF_COV_MAT, REF_COR_MAT
    assert np.allclose(REF_ERR_LIST, numeric_tools.extract_statistical_errors(REF_COV_MAT))
