"""
kafe
====

A Python package for fitting and plotting for use in physics lab courses.

This Python package allows fitting of user-defined functions to data. A dataset
is represented by a `Dataset` object which stores measurement data as `NumPy`
arrays. The uncertainties of the data are also stored in the `Dataset` as an
`error matrix`, allowing for both correlated and uncorrelated errors to be
accurately represented.

The constructor of a `Dataset` object accepts several keyword arguments and can
be used to construct a `Dataset` out of data which has been loaded into
`Python` as `NumPy` arrays. Alternatively, a plain-text representation of a
`Dataset` can be read from a file.

Also provided are helper functions which construct a `Dataset` object from a
file containing column data (one measurement per row, column order can be
specified).

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

"""

# Import main kafe components
from kafe.dataset import Dataset, build_dataset
from kafe.fit import Fit, chi2
from kafe.plot import Plot, PlotStyle
from kafe.file_tools import parse_column_data,  parse_matrix_file
from kafe.numeric_tools import cov_to_cor, cor_to_cov
from function_tools import FitFunction, LaTeX, ASCII

# Import version info
from kafe._version_info import major, minor, revision

# Import and create logging tools and related stuff
import logging
from constants import D_DEBUG_MODE

logger = logging.getLogger('kafe')  # create logger
ch = logging.StreamHandler()  # create console handler (ch)

if D_DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
    ch.setLevel(logging.WARNING)

# create formatter
formatter = logging.Formatter('%(name)s %(asctime)s :: \
                               %(levelname)s :: %(message)s')
ch.setFormatter(formatter)  # add formatter to ch
logger.addHandler(ch)  # add ch to logger

_version_info = (major, minor, revision)
_version_suffix = ""  # for suffixes such as 'rc' or 'beta' or 'alpha'

__version__ = "%d.%d.%d" % _version_info
__version__ += _version_suffix
