import kafe  # from this directory
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

# class for running unit tests
# from: https://pytest.org/latest/goodpractices.html
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.pytest_args)
        sys.exit(errcode)

setup(
    name='kafe',
    version=kafe.__version__,
    author='Daniel Savoiu, Guenter Quast',
    author_email='danielsavoiu@gmail.com',
    packages=['kafe'],
    package_data={'kafe': ['config/*.conf']},
    scripts=[],
    url='https://github.com/dsavoiu/kafe',
    license='GNU Public Licence',
    description='A Python Package for Introduction to \
        Data Analysis in Physics Lab Courses',
    long_description='todo: add long description',  # open('README.txt').read()
    setup_requires=[
        "NumPy >= 1.7.1",
        "SciPy >= 0.12.0",
        "matplotlib >= 1.5.0",
    ],
    tests_require=['pytest'],
    cmdclass = { 'test' : PyTest }
)
