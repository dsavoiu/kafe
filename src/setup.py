from distutils.core import setup

setup(
    name='kafe',
    version='0.3alpha1',
    author='Daniel Savoiu',
    author_email='danielsavoiu@gmail.com',
    packages=['kafe'],
    scripts=[],
    url='https://ekptrac.physik.uni-karlsruhe.de/trac/dsavoiu/browser',
    license='LICENSE',
    description='A Python Package for Introduction to Data Analysis in Physics Lab Courses',
    long_description='todo: add long description',#open('README.txt').read(),
    setup_requires=[
        "NumPy >= 1.6.1",
        "SciPy >= 0.9.0",
        "matplotlib >= 1.1.1"
    ]
)
