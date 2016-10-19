import unittest2
import kafe
import numpy as np

from kafe.function_tools import FitFunction, LaTeX


class Multfit_Test_no_Minimizer(unittest2.TestCase):

    def setUp(self):
        @FitFunction
        def IUmodel(U, R0=1., alph=0.004, p5=0.5, p4=0.9, p3=19.38):
            # use empirical temerature-dependence T(U) from T-U fit
            T = p5 * U * U + p4 * U + p3
            return U / (R0 * (1. + T * alph))

        @FitFunction
        def quadric(U, p2=0.5, p1=0.9, p0=19.38):
            return p2 * U * U + p1 * U + p0

        U=[0.5,   1.,    1.5]
        I=[0.5,   0.89,  1.41]
        T=[293.5-273.15,  293.8-273.15 , 295.4-273.15]
        # Set first dataset
        kTUdata = kafe.Dataset(data=(U, T))
        # Set second dataset
        kIUdata = kafe.Dataset(data=(U, I))
        self.Test_Multifit = kafe.Multifit([(kTUdata, quadric), (kIUdata, IUmodel)], minimizer_to_use=None, quiet=True)

    def test_current_parameter_values_minuit(self):
        parameter_values = [0.5,0.9,19.38,1.,0.004,0.5,0.9,19.38]
        self.assertEqual(self.Test_Multifit.current_parameter_values_minuit, parameter_values)

    def test_current_parameter_errors_minuit(self):
        parameter_values = [0.5,0.9,19.38,1.,0.004,0.5,0.9,19.38]
        parameter_errors = [x * 0.1 for x in parameter_values]
        assert np.allclose(parameter_errors,self.Test_Multifit.current_parameter_errors_minuit, atol=1e-4)

    def test_link_alias(self):
        self.Test_Multifit.link_parameters("p5", "p2")
        alias={"quadric.p2":"IUmodel.p5"}
        self.assertEqual(alias, self.Test_Multifit.parameter_space.alias)

    def test_fix_parameter_value(self):
        self.Test_Multifit.fix_parameters(["p5"], [0.0])
        id = self.Test_Multifit.parameter_space.get_parameter_ids(["p5"])[0]
        self.assertEqual(0, self.Test_Multifit.current_parameter_values_minuit[id])

    def test_set_parameter_value_without_function_keyword(self):
        self.Test_Multifit.set_parameter(p5 = [0, 0.1])
        id = self.Test_Multifit.parameter_space.get_parameter_ids(["p5"])[0]
        self.assertEqual(0, self.Test_Multifit.current_parameter_values_minuit[id])

    def test_set_parameter_value_with_function_keyword(self):
        @FitFunction
        def IUmodel(U, R0=1., alph=0.004, p5=0.5, p4=0.9, p3=19.38):
            # use empirical temerature-dependence T(U) from T-U fit
            T = p5 * U * U + p4 * U + p3
            return U / (R0 * (1. + T * alph))

        self.Test_Multifit.set_parameter(function=IUmodel, p5=[0, 0.1])
        id = self.Test_Multifit.parameter_space.get_parameter_ids(["p5"])[0]
        self.assertEqual(0, self.Test_Multifit.current_parameter_values_minuit[id])

    def test_set_parameter_error_without_function_keyword(self):
        self.Test_Multifit.set_parameter(p5 = [0, 0.1])
        id = self.Test_Multifit.parameter_space.get_parameter_ids(["p5"])[0]
        self.assertEqual(0.1, self.Test_Multifit.current_parameter_errors_minuit[id])

    def test_set_parameter_error_with_function_keyword(self):
        @FitFunction
        def IUmodel(U, R0=1., alph=0.004, p5=0.5, p4=0.9, p3=19.38):
            # use empirical temerature-dependence T(U) from T-U fit
            T = p5 * U * U + p4 * U + p3
            return U / (R0 * (1. + T * alph))

        self.Test_Multifit.set_parameter(function=IUmodel, p5=[0, 0.1])
        id = self.Test_Multifit.parameter_space.get_parameter_ids(["p5"])[0]
        self.assertEqual(0.1, self.Test_Multifit.current_parameter_errors_minuit[id])

    def test_set_all_parameter_values(self):
        parameter_values = [0.7,1,18,2,0.005,0.6,1,10]
        self.Test_Multifit.set_parameter(parameter_values, [x * 0.1 for x in parameter_values])
        self.assertEqual(parameter_values, self.Test_Multifit.current_parameter_values_minuit)

    def test_set_all_parameter_errors(self):
        parameter_values = [0.7, 1, 18, 2, 0.005, 0.6, 1, 10]
        parameter_errors = [x * 0.1 for x in parameter_values]
        self.Test_Multifit.set_parameter(parameter_values, parameter_errors)
        assert np.allclose(parameter_errors, self.Test_Multifit.current_parameter_errors_minuit, atol=1e-4)

class ParameterSpace_Test(unittest2.TestCase):

    def setUp(self):
        @LaTeX(name='I', parameter_names=(r'R_0', r'\alpha_T', 'p_2', 'p_1', 'p_0'),
               expression=r'U/ \left( R_0 (1 + t \cdot \alpha_T) \right)')
        @FitFunction
        def IUmodel(U, R0=1., alph=0.004, p5=0.5, p4=0.9, p3=19.38):
            # use empirical temerature-dependence T(U) from T-U fit
            T = p5 * U * U + p4 * U + p3
            return U / (R0 * (1. + T * alph))

        @LaTeX(name='T', parameter_names=('p_5', 'p_4', 'p_3'),
               expression=r'p_5 U^{2} + p_4 U + p_3')
        @FitFunction
        def quadric(U, p2=0.5, p4=0.9, p0=19.38):
            return p2 * U * U + p4 * U + p0

        # Get data from file
        U=[0.5,   1.,    1.5]
        I=[0.5,   0.89,  1.41]
        T=[293.5-273.15,  293.8-273.15 , 295.4-273.15]
        # Set first dataset
        kTUdata = kafe.Dataset(data=(U, T))
        # Set second dataset
        kIUdata = kafe.Dataset(data=(U, I))
        Fit1 = kafe.Fit(kTUdata,quadric)
        Fit2 = kafe.Fit(kIUdata, IUmodel)
        self.parameter_space = kafe.multifit._ParameterSpace([Fit1,Fit2])

    def test_number_of_parameters_init(self):
        self.assertEqual(self.parameter_space.total_number_of_parameters,8)

    def test_autolink_alias(self):
        self.parameter_space.autolink_parameters()
        dic = {"IUmodel.p4":"quadric.p4"}
        self.assertEqual(self.parameter_space.alias,dic)

    def test_link_parameter_number_of_parameters(self):
        self.parameter_space.link_parameters("p5","p2")
        self.assertEqual(self.parameter_space.total_number_of_parameters, 7)

    def test_link_parameter_changed_bool(self):
        self.parameter_space.link_parameters("p5", "p2")
        self.assertTrue(self.parameter_space.parameter_changed_bool)

    def test_link_parameter_alias(self):
        self.parameter_space.link_parameters("p2","p5")
        dic = {"IUmodel.p5": "quadric.p2"}
        self.assertEqual(self.parameter_space.alias, dic)

    def test_delink_parameter_alias(self):
        self.parameter_space.link_parameters("p2","p5")
        self.parameter_space.delink("p2","p5")
        self.assertFalse(self.parameter_space.alias)

    def test_delink_number_of_parameters(self):
        self.parameter_space.link_parameters("p2","p5")
        self.parameter_space.delink("p2","p5")
        self.assertEqual(self.parameter_space.total_number_of_parameters,8)

    def test_parameter_to_id_dic_no_links(self):
        dic = {'IUmodel.R0': 3, 'IUmodel.alph': 4,
               'IUmodel.p3': 7, 'IUmodel.p4': 6,
               'quadric.p4': 1, 'quadric.p2': 0,
               'IUmodel.p5': 5, 'quadric.p0': 2}
        self.parameter_space._update_parameter_to_id()
        self.assertEqual(dic,self.parameter_space.parameter_to_id)

    def test_parameter_to_id_dic_with_links(self):
        dic = {'IUmodel.R0': 3, 'IUmodel.alph': 4,
               'IUmodel.p3': 6, 'IUmodel.p4': 5,
               'quadric.p4': 1, 'quadric.p2': 0,
               'quadric.p0': 2}
        self.parameter_space.link_parameters("p2", "p5")
        self.parameter_space._update_parameter_to_id()
        self.assertEqual(dic,self.parameter_space.parameter_to_id)

    def test_get_current_parameter_values(self):
        list = [0, 1, 2, 3, 4, 5, 6, 7]
        current_quadric = self.parameter_space.get_current_parameter_values(list,
                                                                            self.parameter_space.fit_list[0].fit_function)
        self.assertEqual(current_quadric,[0,1,2])

    def test_fit_to_parameter_id(self):
        list = [0,1,2]
        ids = self.parameter_space.fit_to_parameter_id(self.parameter_space.fit_list[0])
        self.assertEqual(list,ids)

    def test_get_parameter_ids(self):
        list = [0,1,2]
        ids= self.parameter_space.get_parameter_ids(["p0","p2","p4"])
        self.assertEqual(sorted(ids),list)

    def test_build_current_parameter_names(self):
        dic = self.parameter_space.build_current_parameter()
        parameter_names = ['quadric.p2', 'quadric.p4',
                 'quadric.p0', 'IUmodel.R0',
                 'IUmodel.alph', 'IUmodel.p5',
                 'IUmodel.p4', 'IUmodel.p3']
        self.assertEqual(parameter_names,dic['names'])

    def test_build_current_parameter_latex_names(self):
        dic = self.parameter_space.build_current_parameter()
        latex_names =  ['p_5', 'p_4', 'p_3', 'R_0', '\\alpha_T', 'p_2', 'p_1', 'p_0']
        self.assertEqual(latex_names,dic['latex_names'])

    def test_build_current_parameter_values(self):
        dic = self.parameter_space.build_current_parameter()
        values = [0.5, 0.9, 19.38, 1.0, 0.004, 0.5, 0.9, 19.38]
        self.assertEqual(values, dic['values'])

    def test_build_current_parameter_errors(self):
        dic = self.parameter_space.build_current_parameter()
        errors = [0.05, 0.09, 1.938, 0.1, 0.0004, 0.05, 0.09, 1.938]
        self.assertEqual(errors, dic['errors'])


if __name__ == '__main__':
    unittest2.main()