import kafe  # from this directory
import sys
import os

from setuptools import setup
from setuptools.command.test import test as TestCommand


# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'README.rst')) as f:
    long_description = f.read()

setup(
    name='kafe',
    version=kafe.__version__,
    author='Daniel Savoiu, Guenter Quast, Joerg Schindler',
    author_email='danielsavoiu@gmail.com',
    packages=['kafe'],
    package_data={'kafe': ['config/*.conf']},
    scripts=[],
    url='https://github.com/dsavoiu/kafe',
    license='GNU Public Licence',
    description='A Python Package for Introduction to \
        Data Analysis in Physics Lab Courses',
    long_description=long_description,
    setup_requires=[
        "NumPy >= 1.7.1",
        "SciPy >= 0.12.0",
        "matplotlib >= 1.5.0",
    ],
    test_suite = 'unittest.collector',
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
