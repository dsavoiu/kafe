#!/bin/bash

# Install script for kafe using pip

# Build the source distribution
python setup.py sdist

# Install using pip
pip install kafe --no-index --find-links dist
