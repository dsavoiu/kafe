#!/usr/bin/env python2

import kafe  # from this directory
import unittest
import sys
import os

from setuptools import setup

def discover_kafe_tests():
    _tl = unittest.TestLoader()
    _ts = _tl.discover(os.path.join('kafe','tests'), 'test_*.py')
    return _ts


def read_local(filename):
    _path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(_path):
        return open(_path).read()
    else:
        return ""

setup(
    name='kafe',
    version=kafe.__version__,
    author='Daniel Savoiu, Guenter Quast, Joerg Schindler',
    author_email='daniel.savoiu@cern.ch',
    packages=['kafe'],
    package_data={'kafe': ['config/*.conf']},
    scripts=[],
    url='https://github.com/dsavoiu/kafe',
    license='GNU Public Licence v3',
    description='A Python Package for Introduction to \
        Data Analysis in Physics Lab Courses',
    long_description=read_local('README.rst'),
    setup_requires=[
        "NumPy >= 1.11.2",
        "SciPy >= 0.17.0",
        "matplotlib >= 1.5.0",
        "iminuit >= 1.2, <2",
    ],
    test_suite='setup.discover_kafe_tests',
    keywords = "data analysis lab courses education students physics fitting minimization",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
