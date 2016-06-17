''' general example for fitting with kafe
      - construct dataset
      - specify the error model using the add_error_source() method
      - perform fit (2nd order polynomial)
      - show and save output
'''
import kafe
#from kafe.function_tools import FitFunction, LaTeX, ASCII
from kafe.function_library import quadratic_3par

#### create a Dataset instance:
my_dataset = kafe.Dataset(
    data = ([0.05,0.36,0.68,0.80,1.09,1.46,1.71,1.83,2.44,2.09,3.72,4.36,4.60],
            [0.35,0.26,0.52,0.44,0.48,0.55,0.66,0.48,0.75,0.70,0.75,0.80,0.90]),
    title='some data',
    axis_labels=['$x$', '$y=f(x)$'])

#### specify the error model
my_dataset.add_error_source('y', 'simple',
            [0.06,0.07,0.05,0.05,0.07,0.07,0.09,0.10,0.11,0.10,0.11,0.12,0.10])

#### Create the Fit object
my_fit = kafe.Fit(my_dataset, quadratic_3par)
# Set initial values and error estimates
my_fit.set_parameters((0., 1., 0.2), (0.5, 0.5, 0.5))
# Do the Fit
my_fit.do_fit()

#### Create result plots and output them
my_plot = kafe.Plot(my_fit)
my_plot.plot_all()
my_plot.save('kafe_example0.pdf') # to file

### Create (and save) contour and profile plots
from kafe.fit import CL2Chi2
contour1 = my_fit.plot_contour(0, 1, dchi2=[1., CL2Chi2(.6827)])
contour2 = my_fit.plot_contour(0, 2, dchi2=[1., CL2Chi2(.6827)])
contour3 = my_fit.plot_contour(1, 2, dchi2=[1., CL2Chi2(.6827)])
#contour1.savefig('kafe_example0_contour1.pdf')
#contour2.savefig('kafe_example0_contour2.pdf')
#contour3.savefig('kafe_example0_contour3.pdf')
profile=my_fit.plot_profile(2)
#profile.savefig('kafe_example0_profile.pdf')

my_plot.show()                    # show everything on screen
