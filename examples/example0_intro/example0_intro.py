''' general example for fitting with kafe
      - construct dataset using helper function build_dataset
      - perform fit (2nd order polynomial)
      - show and save output
'''
from kafe import *
#from kafe.function_tools import FitFunction, LaTeX, ASCII
from kafe.function_library import quadratic_3par

#### build a Dataset instance:
myDataset = build_dataset(
    [0.05,0.36,0.68,0.80,1.09,1.46,1.71,1.83,2.44,2.09,3.72,4.36,4.60],
    [0.35,0.26,0.52,0.44,0.48,0.55,0.66,0.48,0.75,0.70,0.75,0.80,0.90],
    yabserr=[0.06,0.07,0.05,0.05,0.07,0.07,0.09,0.1,0.11,0.1,0.11,0.12,0.1],
    title='some data',
    axis_labels=['$x$', '$y=f(x)$'])

#### Create the Fit object
myFit = Fit(myDataset, quadratic_3par)
# Set initial values and error estimates
myFit.set_parameters((0., 1., 0.2), (0.5, 0.5, 0.5))
# Do the Fit
myFit.do_fit()

#### Create result plots and output them
myPlot = Plot(myFit)
myPlot.plot_all()
myPlot.save('kafe_example0.pdf') # to file

### Create (and save) contour and profile plots
contour1 = myFit.plot_contour(0, 1, dchi2=[1.,2.3])
contour2 = myFit.plot_contour(0, 2, dchi2=[1.,2.3])
contour3 = myFit.plot_contour(1, 2, dchi2=[1.,2.3])
#contour1.savefig('kafe_example0_contour1.pdf')
#contour2.savefig('kafe_example0_contour2.pdf')
#contour3.savefig('kafe_example0_contour3.pdf')
profile=myFit.plot_profile(2)
#profile.savefig('kafe_example0_profile.pdf')

myPlot.show()                    # show everything on screen
