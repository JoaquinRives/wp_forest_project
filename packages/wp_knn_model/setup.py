#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup
from datetime import datetime


now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M%S")

# Package meta-data.
NAME = 'knn_model'
DESCRIPTION = 'k nearest neighbors regression model for prediction of the water permeability' \
              'level of forest soil. The model is intended to be use in the forest industry to' \
              'guide routing decisions in harvesting operations.'

# TODO
URL = 'https://github.com/JoaquinRives/'
EMAIL = 'joaquin.rives01@email.com'
AUTHOR = 'Joaquin Rives'
REQUIRES_PYTHON = '>=3.6.0'


# Requirements
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Setup
setup(
    name=NAME,
    version=dt_string,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'knn_model': [dt_string]},
    install_requires=None,
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
