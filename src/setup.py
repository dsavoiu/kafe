from distutils.core import setup

setup(
    name='kafe',
    version='0.3alpha1',
    author='Daniel Savoiu',
    author_email='danielsavoiu@gmail.com',
    packages=['kafe'],
    scripts=[],
    url='https://ekptrac.physik.uni-karlsruhe.de/trac/dsavoiu/browser',
    license='LICENSE.txt',
    description='A Python Package for Introduction to Data Analysis in Physics Lab Courses',
    long_description='todo: add long description',#open('README.txt').read(),
    install_requires=[
        "NumPy >= 1.7.1",
	    "SciPy >= 0.12.0",
	    "matplotlib >= 1.2.0",
        "ROOT >= 5.34",
        "PyROOT >= 1.1.1"]
)
