"""
Unit tests for basic fit functionality

Most tests here are based on the standard kafe examples.
"""

import numpy as np
import kafe

def test_W_boson_mass_averaging_with_y_cov_mat():
    W_mass_values = np.array([
        80.429, 80.339, 80.217, 80.449, 80.477, 80.310, 80.324, 80.353])
    W_mass_cov_mat = np.matrix([
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

    ref_pval = (80.3743268547,)
    ref_perr = (0.03513045624,)

    _dataset = kafe.Dataset(data=(range(len(W_mass_values)), W_mass_values))
    _dataset.add_error_source('y', 'matrix', W_mass_cov_mat)

    from kafe.function_library import constant_1par
    _fit = kafe.Fit(_dataset, constant_1par)
    _fit.do_fit()
    _pval, _perr = _fit.get_parameter_values(), _fit.get_parameter_errors()

    assert np.allclose(_pval, ref_pval)
    assert np.allclose(_perr, ref_perr)

def test_W_boson_mass_averaging_without_cov_mats():
    W_mass_values = np.array([
        80.429, 80.339, 80.217, 80.449, 80.477, 80.310, 80.324, 80.353])
    W_mass_errors = np.array([
        0.05887274,  0.07596051,  0.07116882,  0.06168468,  0.0818352,
        0.10107918,  0.08955445,  0.08099383])

    ref_pval = (80.3727519701,)
    ref_perr = (0.02629397757,)

    _dataset = kafe.Dataset(data=(range(len(W_mass_values)), W_mass_values))
    _dataset.add_error_source('y', 'simple', W_mass_errors)

    from kafe.function_library import constant_1par
    _fit = kafe.Fit(_dataset, constant_1par)
    _fit.do_fit()
    _pval, _perr = _fit.get_parameter_values(), _fit.get_parameter_errors()

    assert np.allclose(_pval, ref_pval)
    assert np.allclose(_perr, ref_perr)

#TODO: add more unit tests based on examples
