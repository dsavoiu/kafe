"""

**kafe** *-- a Python package for fitting and plotting for use in physics lab
courses.*

This Python package allows fitting of user-defined functions to data. A dataset
is represented by a `Dataset` object which stores measurement data as `NumPy`
arrays. The uncertainties (errors) of the data are also stored in the `Dataset`
as a list of one or more `ErrorSource` objects, each of which stores a part of
the uncertainty information as a so-called *covariance matrix* (also called an
*error matrix*). This allows **kafe** to work with uncertainties of different
kinds for a `Dataset`, particularly when there is a degree of correlation
between the uncertainties of the datapoints.

Fitting with **kafe** in a nutshell goes like this:

    1) create a `Dataset` object from your measurement data

    >>> my_d = kafe.Dataset(data=[[0., 1., 2.], [1.23, 3.45, 5.62]])

    2) add errors (uncertainties) to your `Dataset`

    >>> my_d.add_error_source('y', 'simple', 0.5)  # y errors, all +/- 0.5

    3) import a model function from `kafe.function_library` (or define one
       yourself)

    >>> from kafe.function_library import linear_2par

    4) create a `Fit` object from your `Dataset` and your model function

    >>> my_f = kafe.Fit(my_d, linear_2par)

    5) do the fit

    >>> my_f.do_fit()

    6) *(optional)* if you want to see a plot of the result, use the `Plot`
       object

    >>> my_p = kafe.Plot(my_f)
    >>> my_p.plot_all()
    >>> my_p.show()

For more in-depth information on **kafe**'s features, feel free to consult the
documentation.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>

"""

# Import config variables
from . import config

# Import version info
from . import _version_info

# Import and create logging tools and related stuff
import logging

_logger = logging.getLogger('kafe')  # create logger
_ch = logging.StreamHandler()  # create console handler (ch)
_fmt = "%(name)s %(asctime)s :: " \
       "%(levelname)s :: %(message)s"

if config.D_DEBUG_MODE:
    _mode = logging.DEBUG
else:
    _mode = logging.WARNING

_logger.setLevel(_mode)
_ch.setLevel(_mode)
logging.basicConfig(filename=config.log_file('kafe.log'),
                    level=_mode, format=_fmt)

# create formatter
_formatter = logging.Formatter("%(name)s %(asctime)s :: "
                               "%(levelname)s :: %(message)s")
_ch.setFormatter(_formatter)  # add formatter to ch
_logger.addHandler(_ch)  # add ch to logger

__version__ = _version_info._get_version_string()

# Import matplotlib and set backend
import matplotlib as _mpl

# get configured fallback behavior on backend setting failure
_orig_backend_fallback_setting = _mpl.rcParams['backend_fallback']
try:
    # try to use backend specified in config
    # note: this may set 'backend_fallback' to False, triggering
    # ImportErrors further down the line
    _mpl.use(config.G_MATPLOTLIB_BACKEND)
except ValueError as e:
    # matplotlib does not provide requested backend
    _logger.error("matplotlib error: %s" % (e,))
    _logger.warning("Failed to load requested backend '%s' for matplotlib. "
                    "Current backend is '%s'."
                    % (config.G_MATPLOTLIB_BACKEND, _mpl.get_backend()))
except ImportError as e:
    # pyplot has probably been imported before kafe and backends
    # could not be switched
    _logger.error("matplotlib error: %s" % (e,))
    _logger.warning("Could not switch from curently running matplotlib backend '%s' "
                    "to requested '%s'. Plots may look or behave differently. This can "
                    "usually be solved by ensuring that `kafe` is imported before `matplotlib.pyplot`."
                    % (_mpl.get_backend(), config.G_MATPLOTLIB_BACKEND))
finally:
    # ensure the original fallback behavior is kept despite call to `use`
    _mpl.rcParams['backend_fallback'] = _orig_backend_fallback_setting

# Import main kafe components
from .dataset import Dataset
from .dataset_tools import build_dataset
from .fit import Fit, chi2
from .plot import Plot, PlotStyle
from .file_tools import (parse_column_data,
                         buildDataset_fromFile, buildFit_fromFile)
from .numeric_tools import cov_to_cor, cor_to_cov
from .function_tools import FitFunction, LaTeX, ASCII
from .multifit import Multifit
from .multiplot import Multiplot
