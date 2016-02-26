.. -*- mode: rst -*-

**********************************
*kafe* - Karlsruhe Fit Environment
**********************************

=====
About
=====

**kafe** is a data fitting framework designed for use in undergraduate
physics lab courses. It is open-source software licensed under the GNU
Public License.

It provides a basic Python toolkit for fitting models to data as well as
for visualizing the fit result. It relies on Python packages such as *NumPy*
and *matplotlib*, and uses the Python interface to the minimizer *MINUIT*
contained in CERN's data analysis framework ROOT or to *iminuit*, which is
available as a separate Python package.

The software originated as part of a bachelor's thesis in physics *Institut
für Experimentelle Kernphysik* (IEKP) at the *Karlsruhe Insitute of Technology*
(KIT). Development is currently slow, but ongoing.

Contributors:

    * Günter Quast <g (dot) quast (at) kit (dot) edu>
    * Daniel Savoiu <daniel (dot) savoiu (at) cern (dot) ch>


For more information, please consult the HTML documentation in:

    ``doc/html/index.html``

============
Requirements
============

*kafe* needs some additional Python packages. The recommended versions of these are
as follows. The version numbers between parentheses refer to the minimum requirements:

    * `SciPy <http://www.scipy.org>`_ >= 0.12.0 (0.9.0)
    * `NumPy <http://www.numpy.org>`_ >= 1.7.1 (1.6.1)
    * `matplotlib <http://matplotlib.org>`_ >= 1.5.0 (1.3.0)


Additionally, a function minimizer is needed. *kafe* implements interfaces to two
function minimizers and requires at least one of them to be installed:

    * *MINUIT*, which is included in *CERN*'s data analysis package `ROOT <http://root.cern.ch>`_ (>= 5.34), or
    * `iminuit <https://github.com/iminuit/iminuit>`_ (>= 1.1.1), which is independent of ROOT


Finally, *kafe* requires a number of external programs:

    * Qt4 (>= 4.8.5) and the Python bindings PyQt4 (>= 3.18.1) are needed because *Qt* is the supported
      interactive frontend for matplotlib. Other frontends are not supported and may cause unexpected behavior.
    * A *LaTeX* distribution (tested with `TeX Live <https://www.tug.org/texlive/>`_), since *LaTeX* is
      used by matplotlib for typesetting labels and mathematical expressions.
    * `dvipng <http://www.nongnu.org/dvipng/>`_ for converting DVI files to PNG graphics

==========================
Installation notes (Linux)
==========================


Most of the above packages and programs can be installed through the package manager on most Linux
distributions.

*kafe* was developed for use on Linux desktop systems. Please note that all
commands below should be run as root.

-----------------------------------------
Install *NumPy*, *SciPy* and *matplotlib*
-----------------------------------------

These packages should be available in the package manager.

In Ubuntu/Mint/Debian:

    .. code:: bash

        apt-get install python-numpy python-scipy python-matplotlib

In Fedora/RHEL/CentOS:

    .. code:: bash

        yum install numpy scipy python-matplotlib


--------------
Install *ROOT*
--------------

ROOT and its Python bindings can be obtained via the package manager in
Ubuntu/Mint/Debian:

    .. code:: bash

        apt-get install root-system libroot-bindings-python5.34 libroot-bindings-python-dev

Or, in Fedora/RHEL/CentOS:

    .. code:: bash

        yum install root root-python


This setup is usually sufficient. However, you may decide to build ROOT yourself. In this case,
be sure to compile with *PyROOT* support. Additionally, for Python to see the *PyROOT* bindings,
the following environment variables have to be set correctly (:
 
    .. code:: bash

        export ROOTSYS=<directory where ROOT is installed>
        export LD_LIBRARY_PATH=$ROOTSYS/lib:$PYTHONDIR/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH


For more info, refer to `<http://root.cern.ch/drupal/content/pyroot>`_.


-----------------
Install `iminuit`
-----------------

*iminuit* is a Python wrapper for the Minuit minimizer which is
independent of ROOT. If compiling/installing ROOT is not possible,
this minimizer can be used instead.

To install the *iminuit* package for Python, the `Pip installer
<http://www.pip-installer.org/>`_ is recommended:

    .. code:: bash

        pip install

If you don't have *Pip* installed, get it from the package manager.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install python-pip

In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python-pip

or use ``easy_install`` (included with `setuptools <https://pypi.python.org/pypi/setuptools>`_):

    .. code:: bash

        easy_install pip

You might also need to install the Python headers for *iminuit* to
compile properly.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install libpython2.7-dev

In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python-devel


Read the README file for more information on other dependencies
(there should be adequate packages for your Linux distribution
to satisfy these).

--------------
Install *kafe*
--------------

To install *kafe* using *Pip*, simply run the helper script as root:

    .. code:: bash

        ./install.sh

To remove kafe using *Pip*, just run the helper script:

    .. code:: bash

        ./uninstall.sh


Alternatively, installing using Python's *setuptools* also works, but does not provide a
clean uninstall. Use this method if installing with *Pip* is not possible:

    .. code:: bash

        python setup.py install

============================
Installation notes (Windows)
============================

*kafe* can be installed under Windows, but requires some additional configuration.
Pre-compiled binaries/installers for all dependencies should be available for the most part.


*(More information to be added)*
