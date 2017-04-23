import unittest
import numpy as np
import kafe


from kafe.function_tools import FitFunction
'''
class Multfit_Test_fit_functionality(unittest.TestCase):

    def setUp(self):
        @FitFunction
        def IUmodel(U, R0=1., alph=0.004, p5=0.5, p4=0.9, p3=19.38):
            T=p5 *U*U + p4 *U + p3
            return U / (R0*(1. + T*alph))
        @FitFunction
        def quadric(U, p5=0.5 , p4=0.9, p3= 19.38):
            return p5*U*U + p4 *U + p3


        U = np.array([0.5,   1.,    1.5,   2.,    2.5,   3.,    3.5,   4.,    4.5,   5.,    5.5,   6.,
         6.5,   7.,    7.5,   8.,    8.5,   9.,    9.5,  10.])

        I=np.array([0.5,   0.89,  1.41,  1.67,  2.3,   2.59,  2.77,  3.57,  3.94,  4.24,  4.73,  4.87,
         5.35,  5.74,  5.77,  6.17,  6.32,  6.83,  6.87,  7.17])

        T=np.array([ 293.5,  293.8,  295.4,  296.8,  299.4,  301.,   303.,   307.4,  310.9,  315.1,
                    318.,   323.2,  327.4,  333.7,  338.2,  343.1,  350.,  354.7,  358.6,  367.9])

        T0 = 273.15
        sig = 0.1
        # Set first dataset
        kTUdata = kafe.Dataset(data=(U, T-T0) )
        kTUdata.add_error_source('x', 'simple', sig)
        kTUdata.add_error_source('y', 'simple', sig)
        # Set second dataset
        kIUdata = kafe.Dataset(data=(U, I))
        kIUdata.add_error_source('x', 'simple', sig)
        kIUdata.add_error_source('y', 'simple', sig)

        # Init Multifit

        self.Test_Multifit = kafe.Multifit([(kTUdata, quadric), (kIUdata, IUmodel)], quiet=True)
        self.Test_Multifit.autolink_parameters()
        self.Test_Multifit.do_fit(quiet=True)

    def test_fit_functionality_with_links_values(self):
        values = np.array([0.665247, 0.981657, 19.385533, 1.017867, 0.003898])
        assert np.allclose(self.Test_Multifit.current_parameter_values_minuit, values, atol=1e-4)

    def test_fit_functionality_with_links_errors(self):
        errors = np.array([0.019618, 0.162890, 0.215978, 0.023866, 0.000429])
        assert np.allclose(self.Test_Multifit.current_parameter_errors_minuit, errors, atol=1e-4)

class Multifit_test_fit_functionality_fix_parameters(unittest.TestCase):

    def setUp(self):
        @FitFunction
        def IUmodel(U, R0=1., alph=0.004, p5=0.5, p4=0.9, p3=19.38):
            T = p5 * U * U + p4 * U + p3
            return U / (R0 * (1. + T * alph))

        @FitFunction
        def quadric(U, p5=0.5, p4=0.9, p3=19.38):
            return p5 * U * U + p4 * U + p3

        U = np.array([0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.,
                      6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.])

        I = np.array([0.5, 0.89, 1.41, 1.67, 2.3, 2.59, 2.77, 3.57, 3.94, 4.24, 4.73, 4.87,
                      5.35, 5.74, 5.77, 6.17, 6.32, 6.83, 6.87, 7.17])

        T = np.array([293.5, 293.8, 295.4, 296.8, 299.4, 301., 303., 307.4, 310.9, 315.1,
                      318., 323.2, 327.4, 333.7, 338.2, 343.1, 350., 354.7, 358.6, 367.9])

        T0 = 273.15
        sig = 0.1
        # Set first dataset
        kTUdata = kafe.Dataset(data=(U, T - T0))
        kTUdata.add_error_source('x', 'simple', sig)
        kTUdata.add_error_source('y', 'simple', sig)
        # Set second dataset
        kIUdata = kafe.Dataset(data=(U, I))
        kIUdata.add_error_source('x', 'simple', sig)
        kIUdata.add_error_source('y', 'simple', sig)

        # Init Multifit

        self.Test_Multifit = kafe.Multifit([(kTUdata, quadric), (kIUdata, IUmodel)], quiet=True)

    def test_fit_functionality_with_links_and_fix_values(self):
        self.Test_Multifit.autolink_parameters()
        self.Test_Multifit.fix_parameters(["p3"], [19.0])
        self.Test_Multifit.do_fit(quiet=True)
        assert np.allclose(self.Test_Multifit.current_parameter_values_minuit[2], 19.0)

'''

