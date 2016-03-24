from setuptools import setup

import kafe  # from this directory

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
    ]
)
