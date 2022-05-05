.. -*- mode: rst -*-

**********************************
*kafe* - Karlsruhe Fit Environment
**********************************

.. image:: https://badge.fury.io/py/kafe.svg
    :target: https://badge.fury.io/py/kafe

.. image:: https://readthedocs.org/projects/kafe/badge/?version=latest
    :target: https://kafe.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/dsavoiu/kafe.svg?branch=master
    :target: https://travis-ci.org/dsavoiu/kafe

**Legacy Version** - development is discontinued.  
see https://github.com/dsavoiu/kafe2 for actually maintained successor. 

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

* Günter Quast <guenter (dot) quast (at) online (dot) de>
* Daniel Savoiu <daniel (dot) savoiu (at) cern (dot) ch>


For more information, please consult the HTML documentation in ``doc/html/index.html``
or on `ReadTheDocs <http://kafe.readthedocs.org/en/latest/>`_.


============
Requirements
============

*kafe* runs under both Python 2 (`>=2.7.9`) and Python 3 (tested with `3.5.2`).

Some additional Python packages are required. The recommended versions of these are
as follows. Please note that more recent versions of these packages should work
as well:

* `SciPy <http://www.scipy.org>`_ >= 0.17.0
* `NumPy <http://www.numpy.org>`_ >= 1.11.2
* `matplotlib <http://matplotlib.org>`_ >= 1.5.0


Additionally, a function minimizer is needed. *kafe* implements interfaces to two
function minimizers and requires at least one of them to be installed:

* *MINUIT*, which is included in *CERN*'s data analysis package `ROOT <http://root.cern.ch>`_ (>= 5.34), or
* `iminuit <https://github.com/iminuit/iminuit>`_ (>= 1.2, < 2.0), which is independent of ROOT


Finally, *kafe* requires a number of external programs:

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

**Note**: This section is written with ROOT version 5.34 in mind.
When using this version, take care that the Python bindings (PyROOT)
are compiled for the version of Python you intend to use (either 2.x or 3.x),
as it is not possible to use both. For newer versions of ROOT (i.e. 6 and
above), this should no longer be an issue.


ROOT and its Python bindings can be obtained via the package manager in
Ubuntu/Mint/Debian:

    .. code:: bash

        apt-get install root-system libroot-bindings-python5.34 libroot-bindings-python-dev

Or, in Fedora/RHEL/CentOS:

    .. code:: bash

        yum install root root-python


This setup is usually sufficient. However, you may decide to build ROOT yourself. In this case,
be sure to compile with *PyROOT* support. Additionally, for Python to see the *PyROOT* bindings,
the following environment variables have to be set correctly:

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

        pip install iminuit

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


Alternatively, installing using Python's *setuptools* also works, but may not
provide a clean uninstall. Use this method if installing with *Pip* is not possible:

    .. code:: bash

        python setup.py install

============================
Installation notes (Windows)
============================

*kafe* can be installed under Windows, but requires some additional configuration.

The recommended Python distribution for working with *kafe* under Windows is
`WinPython <https://winpython.github.io/>`_, which has the advantage that it is
portable and comes with a number of useful pre-installed packages. Particularly,
*NumPy*, *SciPy* and *matplotlib* are all pre-installed in *WinPython*.

-----------------
Install `iminuit`
-----------------

After installing *WinPython*, start 'WinPython Command Prompt.exe' in the
*WinPython* installation directory and run

    .. code:: bash

        pip install iminuit

--------------
Install `kafe`
--------------

Now *kafe* can be installed from PyPI by running:

    .. code:: bash

        pip install kafe

Alternatively, it may be installed directly using *setuptools*. Just run
the following in 'WinPython Command Prompt.exe' after switching to the
directory into which you have downloaded *kafe*:

    .. code:: bash

        python setup.py install

--------------------------------------
Using *kafe* with ROOT under Windows
--------------------------------------

If you want *kafe* to work with ROOT's ``TMinuit`` instead of using
*iminuit*, then ROOT has to be installed. Please note that ROOT releases
for Windows are 32-bit and using the PyROOT bindings on a 64-bit *WinPython*
distribution will not work.

A pre-built verson of ROOT for Windows is available on the ROOT homepage as a Windows
Installer package. The recommended version is
`ROOT 5.34 <https://root.cern.ch/content/release-53434>`_.
During the installation process, select "Add ROOT to the system PATH for all users"
when prompted. This will set the ``PATH`` environment variable to include
the relevant ROOT directories. The installer also sets the ``ROOTSYS`` environment
variable, which points to the directory where ROOT in installed. By default,
this is ``C:\root_v5.34.34``.

Additionally, for Python to find the *PyROOT* bindings, the ``PYTHONPATH``
environment variable must be modified to include the ``bin`` subdirectory
of path where ROOT is installed. On Windows 10, assuming ROOT has been installed
in the default directory (``C:\root_v5.34.34``), this is achieved as follows:

  1)  open the Start Menu and start typing "environment variables"
  2)  select "Edit the system environment variables"
  3)  click the "Environment Variables..." button
  4)  in the lower part, under "System variables", look for the "PYTHONPATH" entry

  5)  modify/add the "PYTHONPATH" entry:

      * if it doesn't exist, create it by choosing "New...",
        enter PYTHONPATH as the variable name
        and ``C:\root_v5.34.34\bin`` as the variable value
      * if it already exists and contains only one path, edit it via "Edit..." and
        insert ``C:\root_v5.34.34\bin;`` at the beginning of the variable value.
        (Note the semicolon!)
      * if the variable already contains several paths, choosing "Edit..." will
        show a dialog box to manage them. Choose "New" and write
        ``C:\root_v5.34.34\bin``

  6)  close all opened dialogs with "OK"


Now you may try to ``import ROOT`` in the *WinPython* interpreter to check
if everything has been set up correctly.

For more information please refer to ROOT's official
`PyROOT Guide <https://root.cern.ch/pyroot>`_.
