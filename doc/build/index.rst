.. meta::
   :description lang=en: kafe - a general, Python-based approach to fit a
      model function to two-dimensional data points with correlated
      uncertainties in both dimensions
   :robots: index, follow


===========================================
Welcome to KaFE (Karlsruhe Fit Environment)
===========================================

   **kafe** is a data fitting framework designed for use in undergraduate
   physics lab courses. It provides a basic `Python` toolkit for fitting
   models to data as well as visualization of the data and the model function.
   It relies on `Python` packages such as :py:mod:`numpy` and :py:mod:`matplotlib`, and uses
   the `Python` interface to the minimizer `Minuit` contained in the data
   analysis framework `ROOT`.


:py:mod:`kafe` Overview
=======================

.. figure:: _static/img/graph_example1.jpg
   :height: 300px
   :width: 300 px
   :scale: 100 %
   :alt: image not found
   :align: right

   `Graphical output generated with kafe`.

The :py:mod:`kafe` package provides a rather general approach to fitting of a model
function to two-dimensional data points with correlated uncertainties in both
dimensions. A typical use-case would be measurements of two quantities,
an `x`- and a `y`-value, with both uncorrelated (statistical) uncertainties
and correlated systematic uncertainties.

Use cases range from performing a simple average of measurements
to complex situations with correlated uncertainties on the measurements
of the x and y values. The `Python` API guarantees full flexibility
for data input. Helper functions, which also serve as examples for
own implementations,  are available to handle file-based examples.

The model function describes the y values as a function of the
x-values and a set of model parameters {p}, `y=f(x; {p})`. Full
flexibility exists as model functions are implemented as
`Python` code. Again, examples are provided, but user
implementations are supported as well.

Fitting is based on the χ²-method, assuming Gaussian errors and
correlations described by covariance matrices. The level of agreement
between data points and the fit model is expressed in terms of the
`χ² probability`, i. e. the probability to find less agreement between
data and model than actually observed. Full access to the covariance
matrix of the - typically correlated - model parameters is provided.

The graphical output visualizes the data and the fit model at the
best-fit-point of the parameters and also shows the uncertainty
of the fit model as a light band surrounding the line representing
the model function.


Code Structure
--------------

.. figure:: _static/img/kafeDiagram.jpg
   :height: 300px
   :width: 600 px
   :scale: 80 %
   :alt: image not found
   :align: right

   `Code structure of the kafe package`

The code of :py:mod:`kafe` is centered around very few classes to handle Data input,
fitting and plotting, as illustrated in the figure on the right-hand side.

Data, their uncertainties, and, optionally, the correlations of the
uncertainties - are passed through the interface of the :py:mod:`kafe` class
:py:class:`~kafe.dataset.Dataset`. Input can be included in the `Python` code or is read
from files in standardized or user-defined formats. The representation
of the data within the :py:class:`~kafe.dataset.Dataset` class is minimalistic, consisting
of the x and y values and the full covariance matrices of their
uncertainties. Correlated errors between x and y values are not
supported yet, as such use cases are extremely rare.

A helper function, :py:func:`~kafe.dataset.build_dataset`, is available
to transform various error models, like a combination of independent
and correlated errors or common absolute or relative errors, to this
basic format.

Adding a model function, taken either from a prepared set of fit
functions within kafe or from a user's own `Python` implementation,
results in a :py:class:`~kafe.fit.Fit` object, which controls the minimizer :py:class:`~kafe.minuit.Minuit`
and provides the results through access methods.

One or multiple fit objects, i. e. the input data and model
functions(s) at the best-fit point in parameter-space, are
visualized by the class :py:class:`~kafe.plot.Plot` with the help of :py:mod:`matplotlib`
functionality. The :py:mod:`plot` module also contains functionality to
display the model uncertainty by surrounding the model function
at the best-fit values of the parameters by a light band, the one-σ
uncertainty band, which is obtained by propagation of the uncertainties
of the fit parameters, taking into account their correlations.



Example
-------

Only very few lines of Python code are needed to perform fits with kafe.
The snippet of code shown below performs a fit of a quadratic
function to some data points with uncertainties:

.. code-block:: python

    from kafe import *
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

    #### Create the plots and output them
    myPlot = Plot(myFit)
    myPlot.plot_all()
    myPlot.save('kafe_example0.pdf') # to file
    myPlot.show()                    # to screen

The output in text form (also available via various :py:meth:`get_...` methods
of the :py:class:`~kafe.fit.Fit` class) contains the values of the parameters at the best-fit
point, their correlation matrix and the fit probability. The example produces
the following graphical output:

.. figure:: _static/img/kafe_example0.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   Example: `Data points with one-dimensional error bars compared
   to a quadratic model function with` **kafe**.

More and advanced examples - like fitting different models
to one data set, comparison of different data sets with model
functions, averaging of correlated measurements or multi-parameter
fits - are provided as part of the `kafe` distribution and are
described in the section `Examples` below. They may serve as
a starting point for own applications.


Installation
------------

To install `kafe`, unpack the archive `kafe-<version>.tgz` , change to
the sub-directory  `kafe-<version>/src/`  and follow the installation
instructions below.

1.) Install using `pip`:

   To install kafe using the `Pip` installer
   (http://www.pip-installer.org/), simply
   run the helper script as root:

     ``./install.sh``

   If you don't have Pip installed, use:

     ``easy_install pip``

   To remove kafe using pip, just run the helper script:

      ``./uninstall.sh``

2. Install using `setuptools`:

   Installing using Python's `setup` tools works, but does not
   provide a clean uninstall. Use this method if installing
   with `Pip` is not possible:

     ``python setup.py install``

:py:mod:`kafe` needs a working version of the CERN data analysis framework ``root``,
freely available at  http://root.cern.ch


Dependencies
............

The recommended versions of external packages for :py:mod:`kafe` are as follows,
the version numbers in parentheses refer to the minimum requirements::

  Python packages:
    * SciPy >= 0.12.0 (0.9.0), which includes
        - NumPy >= 1.7.1 (1.6.1) and
        - matplotlib >= 1.2.0 (1.1.1)

  Other dependencies:
    * ROOT >= 5.34 (http://root.cern.ch)
    * Qt4 >= 4.8.5 `(could work with other versions)`
    * PyQt >= 3.18.1 `(could work with other versions)`
    * A LaTeX distribution `(tested with texlive)`

Be sure that the version of `ROOT` you use is compiled with `PyROOT` support.
For `Python` to see the `PyROOT` bindings, the following environment variables
must be set correctly::

    export ROOTSYS=<directory where ROOT is installed>
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$ROOTSYS/lib
    export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH

For more info, refer to [http://root.cern.ch/drupal/content/pyroot].

`Qt` is needed because it is the supported interactive front-end for
:py:mod:`matplotlib`. Other front-ends are not supported and can cause weird behavior.

`LaTeX` is used by :py:mod:`matplotlib` for displaying labels and mathematical
expressions on graphs.


Fit examples, utilities, tips and tricks
=========================================

A wide range of applications of the :py:mod:`kafe` core and the usage of
the helper functions is exemplified here. All of them
are contained in the sub-directory ``examples/`` of the
:py:mod:`kafe` distribution and are intended to serve as a basis for
user projects.


Example 1 - model comparison
----------------------------

To decide whether a model is adequate to describe a given
set of data, typically several models have to be fit to the
same data. Here is the code for a comparison of a data set
to two models, namely a linear and an exponential function:

.. code-block:: python

    # import everything we need from kafe
    from kafe import *
    # additionally, import the two model functions we want to fit:
    from kafe.function_library import linear_2par, exp_2par

    ############
    # Load the Dataset from the file
    my_dataset = Dataset(input_file='dataset.dat', title="Example Dataset")
    ### Create the Fits
    my_fits = [Fit(my_dataset, exp_2par),
               Fit(my_dataset, linear_2par)]
    ### Do the Fits
    for fit in my_fits:
    fit.do_fit()
    ### Create the plots, save and show output
    my_plot = Plot(my_fits[0], my_fits[1])
    my_plot.plot_all(show_data_for=0) # show data only once (it's the same data)
    my_plot.save('plot.pdf')
    my_plot.show()


The file `dataset.dat` contains x and y data in the standard :py:mod:`kafe` data
format, where values and errors (and optionally also correlation coefficients)
are given for each axis separately. `#` indicates a comment line, which
is ignored when reading the data::

    # axis 0: x
    # datapoints  uncor. err.
    0.957426  3.0e-01
    2.262212  3.0e-01
    3.061167  3.0e-01
    3.607280  3.0e-01
    4.933100  3.0e-01
    5.992332  3.0e-01
    7.021234  3.0e-01
    8.272489  3.0e-01
    9.250817  3.0e-01
    9.757758  3.0e-01

    # axis 1: y
    # datapoints  uncor. err.
    1.672481  2.0e-01
    1.743410  2.0e-01
    1.805217  2.0e-01
    2.147802  2.0e-01
    2.679615  2.0e-01
    3.110055  2.0e-01
    3.723173  2.0e-01
    4.430122  2.0e-01
    4.944116  2.0e-01
    5.698063  2.0e-01


The resulting output is shown below. As can be seen already
from the graph, the exponential model better describes the
data. The χ² probability in the printed output shows, however,
that the linear model would be marginally acceptable as well::

    linear_2par
    chi2prob 0.052
    HYPTEST  accepted (CL 5%)

    exp_2par
    chi2prob 0.96
    HYPTEST  accepted (CL 5%)


.. figure:: _static/img/kafe_example1.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Output of example1 - compare two models`


Example 2 - two fits and models
-------------------------------

Another typical use case consists of comparing two sets
of measurements and the models derived from them. This is
very similar to the previous example with minor
modifications:

.. code-block:: python

    ...

    ############
    # Workflow #
    ############
    # Load two Datasets from files
    my_datasets = [Dataset(input_file='dataset1.dat', title="Example Dataset 1"),
                   Dataset(input_file='dataset2.dat', title="Example Dataset 2")]
    # Create the Fits
    ...
    # Do the Fits
    ...
    # Create the plots
    my_plot.plot_all()  # this time without any arguments, i.e. show everything
    ...


This results in the following output:

.. figure:: _static/img/kafe_example2.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Output of example2 - comparison of two linear fits.`

Although the parameters extracted from the two data sets agree within
errors, the uncertainty bands of the two functions do not overlap
in the region where the data of Dataset 2 are located, so the data
are most probably incompatible with the assumption of an underlying
single linear model.


Example 3 - properties of a Gauss curve
---------------------------------------

This example creates a dummy :py:class:`~kafe.dataset.Dataset` object whose points lie exactly
on a Gaussian curve. The :py:class:`~kafe.fit.Fit` will then converge toward that very same
Gaussian. When plotting, the data points used to "support" the curve
can be omitted.

This example shows how to access the :py:mod:`kafe` plot objects
to annotate plots with :py:mod:`matplotlib` functionality.

.. figure:: _static/img/kafe_example3.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Output of example 3 - properties of a Gauss curve.`



Example 4 - average of correlated measurements
----------------------------------------------

The average of a set of measurements can be considered as a fit
of a constant to a set of input data. This example illustrates
how correlated errors are handled in :py:mod:`kafe`.
Measurements can have a common error, which may be absolute
or relative, i. e. depend on the input value.  In more complicated
cases the full covariance matrix must be constructed.

:py:mod:`kafe` has a helper function, :py:func:`~kafe.dataset.build_dataset` in module :py:mod:`fit`,
which aids in setting up the covariance matrix and transforming
the input to the default format used by the :py:class:`~kafe.dataset.Dataset` and :py:class:`~kafe.fit.Fit`
classes. Two further helper functions in module :py:mod:`file_tools`
aid in reading the appropriate information from data files.

  1. The function  :py:func:`~kafe.file_tools.parse_column_data` reads the input values and their
     independent errors from one file, and optionally covariance
     matrices for the x and y axes from additional files. The field ordering
     is defined by a control string.

  2. Another helper function, :py:func:`~kafe.file_tools.buildDataset_fromFile`, specifies
     input values or blocks of input data from a single file with
     keywords.

The second version needs only very minimal additional user
code, as illustrated here:

.. code-block:: python

    from kafe import *
    from kafe.function_library import constant_1par
    from kafe.file_tools import buildDataset_fromFile
    #
    # ---------------------------------------------------------
    fname = 'WData.dat'
    curDataset = buildDataset_fromFile(fname) # Dataset from input file
    curFit = Fit(curDataset, constant_1par)   # set up the fit object
    curFit.do_fit()

    myPlot = Plot(curFit)
    myPlot.plot_all()
    myPlot.save("plot.pdf")
    myPlot.show()


The input file is necessarily more complicated, but holds
the full information on the data set in one place. Refer to
the documentation of the function :py:func:`~kafe.file_tools.parse_general_inputfile`
in module :py:mod:`file_tools` for a full description of the
currently implemented keywords. The input file for the
averaging example is here::

    # Measurements of W boson mass (combined LEP2, 2013)
    # --------------------------------------------------
    # example to use parse_general_inputfile from kafe;
    #  covariance matrix build from common errors
    # --
    #  Meta data for plotting
    *TITLE measurements of the W boson mass
    *xLabel number of measurement
    *yLabel $m_\matrhm{W}$
    *yUnit GeV/$c^2$

    # x data need not be given for averaging

    # ------------------------------------------------------------
    #  Measurements of W mass by ALEPH, DELPI, L3 and OPAL
    #                              from from LEP2 Report Feb. 2013
    #  common errors within channels
    #                     2q2l: 0.021 GeV,
    #                       4q: 0.044 GeV,
    #     and between channels: 0.025 GeV
    # ------------------------------------------------------------

    *yData_SCOV
    # W_mass  err     syst    sqrt of the off-diagonal
    # 2q2l channel                           elements of the
    80.429  0.055   0.021          #         covariance matrix
    80.339  0.073   0.021   0.021
    80.217  0.068   0.021   0.021 0.021
    80.449  0.058   0.021   0.021 0.021 0.021
    # 4q  channel
    80.477  0.069   0.044   0.025 0.025 0.025 0.025 0.044
    80.310  0.091   0.044   0.025 0.025 0.025 0.025 0.044 0.044
    80.324  0.078   0.044   0.025 0.025 0.025 0.025 0.044 0.044 0.044
    80.353  0.068   0.044   0.025 0.025 0.025 0.025 0.044 0.044 0.044 0.044


Example 5 - multi-parameter fit (damped oscillation)
----------------------------------------------------

This example shows the fitting of a more complicated model function
to data collected from a damped harmonic oscillator. In such
non-linear fits, stetting the initial values is sometimes crucial
to let the fit converge at the global minimum. The :py:class:`~kafe.fit.Fit` object
provides the method :py:meth:`~kafe.fit.Fit.set_parameters` for this purpose. As the
fit function for this problem is not a standard one, it is defined
explicitly making use of the decorator functions available in :py:mod:`kafe`
to provide nice type setting of the parameters. This time, the
function :py:func:`~kafe.file_tools.parse_column_data` is used to read the input,
which is given as separate columns with the fields

  ``<time>  <Amplitude>    <error on time>   <error on Amplitude>``


Here is the example code:

.. code-block:: python

    ...
    from kafe import *
    from numpy import exp, cos
    # Model function definition #
    # ---------------------------
    # Set an ASCII expression for this function
    @ASCII(x_name="t", expression="A0*exp(-t/tau)*cos(omega*t+phi)")
    # Set some LaTeX-related parameters for this function
    @LaTeX(name='A', x_name="t",
           parameter_names=('a_0', '\\tau{}', '\\omega{}', '\\varphi{}'),
           expression="a_0\\,\\exp(-\\frac{t}{\\tau})\,"
                      "\cos(\\omega{}\\,t+\\varphi{})")
    @FitFunction
    def damped_oscillator(t, a0=1, tau=1, omega=1, phi=0):
        return a0 * exp(-t/tau) * cos(omega*t + phi)

    # ---- Workflow #
    # load the experimental data from a file
    my_dataset = parse_column_data('damped_oscillation.dat',
        field_order="x,y,xabserr,yabserr", title="Damped Oscillator",
        axis_labels=['Time t', 'Amplitude'])
    # --- Create the Fit
    my_fit = Fit(my_dataset, damped_oscillator)
    # Set the initial values for the fit:
    #                      a_0 tau omega phi
    my_fit.set_parameters((1., 3., 6.28, 0.))
    my_fit.do_fit()
    # --- Create and output the plots
    my_plot = Plot(my_fit)
    my_plot.plot_all()
    my_plot.save('plot.pdf')
    my_plot.show()

.. figure:: _static/img/kafe_example5.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Example 5 - fit of the time dependence of the amplitude of a damped harmonic oscillator.`


Example 6 - another multi-parameter fit
---------------------------------------


This example is not much different from the previous one, except that
the fit function, a standard fourth-degree polynomial from the module
:py:mod:`function_library`, is modified to reflect the names of the problem
given, and :py:mod:`matplotlib` functionality is used to influence the
output of the plot, e.g. axis names and linear or logarithmic scale.

It is also shown how to circumvent a problem that
often arises when errors depend on the measured values.
For a counting rate, the (statistical) error is typically estimated
as the square root of the (observed) number of entries in each bin.
For large numbers of entries, this is not a problem,
but for small numbers, the correlation between the observed
number of entries and the error derived from it leads to a
bias when fitting functions to the data. This problem can be
avoided by iterating the fit procedure:

In a pre-fit, a first approximation of the model function is
determined, which is then used to calculate
the expected errors, and the original errors are
replaced before performing the final fit. Note that the numbers
of entries in the bins must be sufficiently large to justify
a replacement of the (asymmetric) Poisson uncertainties by
the symmetric uncertainties implied by the χ²-method.

The implementation of this  procedure needs accesses some
more fundamental methods of the `Dataset`, `Fit` and
`FitFunction` classes. The code shown below demonstrates
how this can be done with :py:mod:`kafe`, using some of its lower-level,
internal interfaces:

.. code-block:: python

    from kafe.function_library import poly4
    # modify function's independent variable name to reflect its nature:
    poly4.x_name = 'x=cos(t)'
    poly4.latex_x_name = 'x=\\cos(\\theta)'
    ...

    # Set the axis labels appropriately
    my_plot.axis_labels = ['$\\cos(\\theta)$', 'counting rate']
    ...
    # load the experimental data from a file
    my_dataset = parse_column_data(
      'counting_rate.dat',
      field_order="x,y,yabserr",
      title="Counting Rate per Angle")

    ### pre-fit
    # error for bins with zero contents is set to 1.
    covmat = my_dataset.get_cov_mat('y')
    for i in range(0, len(covmat)):
        if covmat[i, i] == 0.:
            covmat[i, i] = 1.
    my_dataset.set_cov_mat('y', covmat) # write it back

    # Create the Fit
    my_fit = Fit(my_dataset, poly4)
    #            fit_label="Linear Regression " + dataset.data_label[-1])

    # perform an initial fit with temporary errors (minimal output)
    my_fit.call_minimizer(final_fit=False, verbose=False)

    # set errors using model at pre-fit parameter values:
    #       sigma_i^2=cov[i, i]=n(x_i)
    fdata = my_fit.fit_function.evaluate(my_fit.xdata,
                                       my_fit.current_parameter_values)
    np.fill_diagonal(covmat, fdata)
    my_fit.current_cov_mat = covmat  # write new covariance matrix
    ### end pre-fit - rest is as usual
    my_fit.do_fit()
    # Create the plots and --
    my_plot = Plot(my_fit)
    # -- set the axis labels
    my_plot.axis_labels = ['$\\cos(\\theta)$', 'counting rate']
    # -- set scale linear / log
    my_plot.axes.set_yscale('linear')
    ...


This is the resulting output:

.. figure:: _static/img/kafe_example6.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Output of example 6 - counting rate.`


Example 7 - non-linear multi-parameter fit
------------------------------------------

Again, not much new in this example, except that the
model is now very non-linear, the intensity distribution
of light after passing through a double-slit. The
non-standard model definition again makes use of the
decorator mechanism to provide nice output - the decorators
(expressions beginning with '@') can safely be omitted if `LaTeX`
output is not needed. Setting of appropriate initial
conditions is absolutely mandatory for this example,
because there  exist many local minima of the χ² function.

Another problem becomes obvious when carefully inspecting
the fit function definition: only two of the three parameters g,
b or k can be determined, and therefore one must be kept fixed,
or an external constraint must be applied.
Failing to do so will result in large, correlated errors
on the parameters g, b and k as an indication of the problem.

Fixing parameters of a model function is achieved by the method
:py:meth:`~kafe.fit.Fit.fix_parameters`, and a constraint within a given uncertainty
is achieved by the method :py:meth:`~kafe.fit.Fit.constrain_parameters`
of the :py:class:`~kafe.fit.Fit` class.

Here are the interesting pieces of code::

    ...
    # Model function definition #
    # Set an ASCII expression for this function
    @ASCII(x_name="x", expression="I0*(sin(k/2*b*sin(x))/(k/2*b*sin(x))"
                                  "*cos(k/2*g*sin(x)))^2")
    # Set some LaTeX-related parameters for this function
    @LaTeX(name='I', x_name="\\alpha{}",
           parameter_names=('I_0', 'b', 'g', 'k'),
           expression="I_0\\,\\left(\\frac{\\sin(\\frac{k}{2}\\,b\\,\\sin{\\alpha})}"
                      "{\\frac{k}{2}\\,b\\,\\sin{\\alpha}}"
                      "\\cos(\\frac{k}{2}\\,g\\,\\sin{\\alpha})\\right)^2")
    @FitFunction
    def double_slit(alpha, I0=1, b=10e-6, g=20e-6, k=1.e7):
        k_half_sine_alpha = k/2*sin(alpha)  # helper variable
        k_b = k_half_sine_alpha * b
        k_g = k_half_sine_alpha * g
        return I0 * (sin(k_b)/(k_b) * cos(k_g))**2

    ...

    # Set the initial values for the fit
    #                      I   b      g        k
    my_fit.set_parameters((1., 20e-6, 50e-6, 9.67e6))
    # fix one of the (redundant) parameters, here 'k'
    my_fit.fix_parameters('k')

    ...


If the parameter `k` in the example above has a (known) uncertainty,
is is more appropriate to constrain it within its uncertainty (which
may be known from an independent measurement or from the specifications
of the laser used in the experiment). To take into account a
wave number `k` known with a precision of 10'000, the
last line in the example above should be replaced by::

    ...
    my_fit.constrain_parameters(['k'], [9.67e6], [1.e4])
    ...


This is the resulting output:

.. figure:: _static/img/kafe_example7.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Example 7 - fit of the intensity distribution of light behind a double slit with fixed or constrained wave length.`


Example 8 - fit of a Breit-Wigner Resonance to data with correlated errors
--------------------------------------------------------------------------

This example illustrates how to define the data and the fit function
in a single file - provided by the helper function :py:func:`~kafe.file_tools.buildFit_fromFile`
in module :py:mod:`file_tools`. Parsing of the input file is done by the
function :py:func:`~kafe.file_tools.parse_general_inputfile`, which had already been introduced
in Example 4. The definition of the fit function as `Python` code
including the :py:mod:`kafe` decorators in the input file, however, is new.
Note: because spaces are used to to separate data  fields in the
input file, spaces needed for proper `Python` indentation have to be
replaced by '~'. The last key in the file defines the start values
of the parameters and their initial ranges.

The advantage of this approach is the location of all data
and the fit model in one place, which is strictly separated
from the `Python` code. The `Python` code below is thus very general
and can handle a large large variety of problems without
modification (except for the file name, which could easily be
passed on the command line)::

    from kafe import *
    from kafe.file_tools import buildFit_fromFile
    # ---------------------------------------------------------
    fname = 'LEP-Data.dat'
    # initialize fit object from file
    BWfit = buildFit_fromFile(fname)
    BWfit.do_fit()
    #
    BWplot = Plot(BWfit)
    BWplot.plot_all()
    BWplot.save("plot.pdf")
    BWplot.show()

The magic happens in the input file, which now has to provide
all the information needed to perform the fit::

    # Fit of a Breit-Wigner function to
    #      measurements of hadronic Z cross sections at LEP
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #  Meta-data for plotting
    *TITLE  LEP Hadronic Cross Section ($\sigma^0_\mathrm{had}$)
    *xLabel $E_CM$
    *xUnit  $\mathrm{GeV}$
    *yLabel $\sigma^0_{\mathrm{had}}$
    *yUnit  $\mathrm{nb}$

    #----------------------------------------------------------------------
    # DATA: average of hadronic cross sections measured by
    #  ALEPH, DELPHI, L3 and OPAL around 7 energy points at the Z resonance
    #----------------------------------------------------------------------

    # CMenergy E err
    *xData
     88.387  0.005
     89.437  0.0015
     90.223  0.005
     91.238  0.003
     92.059  0.005
     93.004  0.0015
     93.916  0.005
    # Centre-of-mass energy has a common uncertainty
    *xAbsCor 0.0017

    # sig^0_h  sig err     #  rad.cor  sig_h measured
    *yData
     6.803   0.036      #  1.7915    5.0114
     13.965  0.013      #  4.0213    9.9442
     26.113  0.075      #  7.867    18.2460
     41.364  0.010      #  10.8617  30.5022
     27.535  0.088      #  3.9164   23.6187
     13.362  0.015      # -0.6933   14.0552
      7.302  0.045      # -1.8181    9.1196
    # cross-sections have a common relative error
    *yRelCor 0.0007

    *FITLABEL Breit-Wigner-Fit {\large{( with s-dependent width )}}
    *FitFunction
    # Breit-Wigner with s-dependent width
    @ASCII(expression='s0*E^2*G^2/[(E^2-M^2)^2+(E^4*G^2/M^2)]')
    @LaTeX(name='f', parameter_names=('\\sigma^0', 'M_Z', '\\Gamma_Z'),
    expression='\\frac{\\sigma^0\\, M_Z^2\\Gamma^2}'
                   '{((E^2-M_Z^2)^2+(E^4\\Gamma^2 / M_Z^2))}')
    @FitFunction
    def fitf(E, M=91.2, G=2.5, s0=41.0):
    ~~~~return s0*E*E*G*G/((E*E-M*M)**2+(E**4*G*G/(M*M)))

    *InitialParameters    # set initial values and ranges
    91.2 0.1
    2.5  0.1
    41.  0.5

Here is the output:

.. figure:: _static/img/kafe_BreitWignerFit.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Output of example 8 - Fit of a Breit-Wigner function.`

This example also contains a code snippet demonstrating how to plot
contours with :py:mod:`kafe`. A *contour* is a region of the parameter
space containing all parameter values consistent with the fit, within
uncertainty.

.. TODO: Need short, correct explanation of contours.

Contours are useful because they provide a way to visualize parameter
uncertainties and correlations. :py:mod:`kafe` supports plotting
2D :math:`1\sigma` contours for any pair of parameters by getting the
relevant data from :cpp:class:`TMinuit` and plotting it using :py:mod:`matplotlib`.
Here is a possible way to do it:

.. code-block:: python

    # plot 1-sigma contour of first two parameters into a separate figure
    x, y = BWfit.minimizer.get_contour(0, 1, n_points=100)  # get contour
    cont_fig = plt.figure()  # create new figure for contour
    cont_ax = cont_fig.gca()  # get/create axes object for current figure
    # set axis labels
    cont_ax.set_xlabel('$%s$' % (BWfit.latex_parameter_names[0],))
    cont_ax.set_ylabel('$%s$' % (BWfit.latex_parameter_names[1],))
    # plot the actual contour
    cont_ax.fill(x, y, alpha=0.25, color='red')
    # save to file
    cont_fig.savefig("kafe_BreitWignerFit_contour12.pdf")

.. figure:: _static/img/kafe_BreitWignerFit_contour12.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Contour generated in example 8 - Fit of a Breit-Wigner function.`

Example 9 - fit of a function to histogram data
-----------------------------------------------

This example brings us to the limit of what is currently
possible with :py:mod:`kafe`. Here, the data represent the
center of a histogram bins ad the number of entries, :math:`n_i`,
in each bin. The (statistical) error is typically estimated
as the square root of the (observed) number of entries in each bin.
For large numbers of entries, this is not a problem,
but for small numbers, especially for bins with 0 entries,
the correlation between the observed number of entries and
the error derived from it leads to a bias when fitting
functions to the histogram data. In particular, bins with
zero entries cannot be handled in the χ²-function, and are
typically omitted to cure the problem.  However, a bias
remains, as bins with downward fluctuations of the
observed numbers of events get assigned smaller errors
and hence larger weights in the fitting procedure - leading
to the aforementioned bias.

These problems are avoided by using a likelihood method for
such use cases, where the Poisson distribution of the uncertainties
and their dependence on the values of the fit model is properly
taken into account. However, the χ²-method can be saved to some
extend if the fitting procedure is iterated. In a pre-fit, a
first approximation of the model function is determined, where
the error in bins with zero entries is set to one. The model
function determined from the pre-fit is then used to calculate
the expected errors for each bin, and the original errors are
replaced before performing the final fit. Note that the numbers
of entries in the bins must be sufficiently large to justify
a replacement of the (asymmetric) Poisson uncertainties by
the symmetric uncertainties implied by the χ²-method.

The code shown below demonstrates
how to get a grip on such more complex procedures with
more fundamental methods of the `Dataset`, `Fit` and
`FitFunction` classes::

    ...
    # Load Dataset from file
    hdataset = Dataset(input_file='hdataset.dat', title="Data for example 9")

    # error for bins with zero contents is set to 1.
    covmat = hdataset.get_cov_mat('y')
    for i in range(0, len(covmat)):
        if covmat[i, i] == 0.:
            covmat[i, i] = 1.
    hdataset.set_cov_mat('y', covmat) # write it back

    # Create the Fit instance
    hfit = Fit(hdataset, gauss, fit_label="Fit of a Gaussian to histogram data")
    #
    # perform an initial fit with temporary errors (minimal output)
    hfit.call_minimizer(final_fit=False, verbose=False)
    #
    #re-set errors using model at pre-fit parameter values:
    #        sigma_i^2=cov[i, i]=n(x_i)
    fdata=hfit.fit_function.evaluate(hfit.xdata, hfit.current_parameter_values)
    np.fill_diagonal(covmat, fdata)
    hfit.current_cov_mat = covmat # write back new covariance matrix
    #
    # now do final fit with full output
    hfit.do_fit()
    # and create, draw, save and show plot
    ...


Here is the output, which shows that the parameters of the
normal distribution, from which the data were generated, are
well reproduced within the uncertainties by the fit result:

.. figure:: _static/img/kafe_example9.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: image not found
   :align: center

   `Output of example 9 - Fit of a Gaussian distribution to histogram data`


:py:mod:`kafe` Documentation -- module descriptions
===================================================
The following documentation of functions and methods
of relevance to the user interface was generated from
the `DocStrings` contained in the `Python` code of the
:py:mod:`kafe` package.
For further information or if in doubt about the exact
functionality, users are invited to consult the source
code.

:py:mod:`__init__` Module
------------------------------
.. automodule:: kafe.__init__
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`_version_info` Module
------------------------------

.. automodule:: kafe._version_info
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`dataset` Module
------------------------

.. automodule:: kafe.dataset
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`file_tools` Module
---------------------------

.. automodule:: kafe.file_tools
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`fit` Module
--------------------

.. automodule:: kafe.fit
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`function_library` Module
---------------------------------

.. automodule:: kafe.function_library
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`function_tools` Module
-------------------------------

.. automodule:: kafe.function_tools
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`minuit` Module
-----------------------

.. automodule:: kafe.minuit
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`config` Module
-----------------------

.. automodule:: kafe.config
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`plot` Module
---------------------

.. automodule:: kafe.plot
    :members:
    :undoc-members:
    :show-inheritance:

:py:mod:`latex_tools` Module
----------------------------

.. automodule:: kafe.latex_tools
    :members:
    :undoc-members:
    :show-inheritance:


:py:mod:`numeric_tools` Module
------------------------------

.. automodule:: kafe.numeric_tools
    :members:
    :undoc-members:
    :show-inheritance:


:py:mod:`stream` Module
-----------------------

.. automodule:: kafe.stream
    :members:
    :undoc-members:
    :show-inheritance:

