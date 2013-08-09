"""
A Python package for fitting and plotting for use in physics lab courses.

This Python package allows fitting of user-defined functions to data. A dataset is
represented by a `Dataset` object which stores measurement data as `NumPy` arrays.
The uncertainties of the data are also stored in the `Dataset` as an `error matrix`,
allowing for both correlated and uncorrelated errors to be accurately represented.

The constructor of a `Dataset` object accepts several keyword arguments and can be used
to construct a `Dataset` out of data which has been loaded into `Python` as `NumPy` arrays.
Alternatively, a plain-text representation of a `Dataset` can be read from a file.

Also provided are helper functions which construct a `Dataset` object from a
file containing column data (one measurement per row, column order can be specified).

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

"""

# Import main kafe components
from kafe.dataset import *
from kafe.fit import *
from kafe.plot import *
from kafe.file_tools import *

__version__ = "0.3alpha1"

if __name__ == "__main__":
    import scipy, numpy, matplotlib, ROOT
    
    print scipy.__version__
    print numpy.__version__
    print matplotlib.__version__