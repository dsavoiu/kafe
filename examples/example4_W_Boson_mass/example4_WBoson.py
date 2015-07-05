#!/usr/bin/env python
# example4a_WBoson
## -------- example4a_WBboson ----------------------------------------
# Description: example fit with kafe package using key-driven parsing
#       parse_general_input_file
#
# all information on the data is located in the input file
#
##--------------------------------------------------------------------
# Author:      G. Quast   Jul. 2014
# dependencies: PYTHON v2.7, sys, numpy, matplotlib.pyplot,
#               kafe
# last modified: DS 01-OCT-14 <minor changes (output file name, etc.)>
#                GQ 27-JUL-14 <initial version>
#---------------------------------------------------------------------
# import everything we need from kafe
import kafe
from kafe.function_library import constant_1par
from kafe.file_tools import buildDataset_fromFile
#
# ---------------------------------------------------------

# begin execution
fname = 'WData.dat'
# build a kafe Dataset from input file
curDataset = buildDataset_fromFile(fname)

# perform fit
curFit = kafe.Fit(curDataset, constant_1par)
curFit.do_fit()

print "average:", curFit.get_parameter_values()
print "error :", curFit.get_parameter_errors()

myPlot = kafe.Plot(curFit)
myPlot.plot_all()
myPlot.save("kafe_example4.pdf")
myPlot.show()





