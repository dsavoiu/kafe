"""

**kafe** *-- a Python package for fitting and plotting for use in physics lab courses.*

This Python package allows fitting of user-defined functions to data. A dataset
is represented by a `Dataset` object which stores measurement data as `NumPy`
arrays. The uncertainties of the data are also stored in the `Dataset` as an
`error matrix`, allowing for both correlated and uncorrelated errors to be
accurately represented.

The constructor of a `Dataset` object accepts several keyword arguments and can
be used to construct a `Dataset` from input data which has been loaded into
`Python` as `NumPy` arrays. Alternatively, a plain-text representations of a
`Dataset` can be read from a file.

Also provided are helper functions which construct a `Dataset` object from a
file containing column data (one measurement per row, column order can be
specified), or from a keyword-driven input format.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

"""

# Import config variables
import config

# Import version info
import kafe._version_info

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
logging.basicConfig(filename='kafe.log', level=_mode, format=_fmt)

# create formatter
_formatter = logging.Formatter("%(name)s %(asctime)s :: "
                              "%(levelname)s :: %(message)s")
_ch.setFormatter(_formatter)  # add formatter to ch
_logger.addHandler(_ch)  # add ch to logger

_version_suffix = ""  # for suffixes such as 'rc' or 'beta' or 'alpha'

__version__ = kafe._version_info._get_version_string()
__version__ += _version_suffix

# Import matplotlib and set backend
import matplotlib
try:
    matplotlib.use(config.G_MATPLOTLIB_BACKEND)
except ValueError, e:
    # matplotlib does not provide requested backend
    logger.error("matplotlib error: %s" % (e,))
    logger.warning("Failed to load requested backend '%s' for matplotlib. "
                   "Current backend is '%s'."
                   % (config.G_MATPLOTLIB_BACKEND, matplotlib.get_backend()))

import matplotlib.pyplot

# Import main kafe components
from kafe.dataset import Dataset, build_dataset
from kafe.fit import Fit, chi2
from kafe.plot import Plot, PlotStyle
from kafe.file_tools import (parse_column_data,
    buildDataset_fromFile, buildFit_fromFile)
from kafe.numeric_tools import cov_to_cor, cor_to_cov
from function_tools import FitFunction, LaTeX, ASCII



