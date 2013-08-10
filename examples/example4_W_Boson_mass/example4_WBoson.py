from kafe import *

import matplotlib.pyplot as plt
import numpy as np

def constant_1par(x, mean=80):
    
    return mean


myDataset_CM = parse_column_data('w_mittelung.dat',
                              field_order='x,y',
                              cov_mat_files=(None, 'w_mass.cov'),
                              title="W-Boson-Mass (mit KMen)")

myDataset_YE = parse_column_data('w_mittelung.dat',
                              field_order='x,y,yabsstat',
                              title="W-Boson-Masse (ohne KMen)")

myFit_CM = Fit(myDataset_CM, constant_1par, function_label="Mittelwert mit KMen")
myFit_CM.do_fit()

myFit_YE = Fit(myDataset_YE, constant_1par, function_label="Mittelwert ohne KMen")
myFit_YE.do_fit()

myPlot = Plot(myFit_CM, myFit_YE)
myPlot.axis_labels = ("Experiment Nr.", "W-Boson-Masse [GeV]")
myPlot.plot_all()
myPlot.show()

