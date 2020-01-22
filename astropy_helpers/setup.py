#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: most of the configuration, including the version number,
# is defined in setup.cfg

import sys
from distutils.version import LooseVersion

import setuptools
from setuptools import setup

if LooseVersion(setuptools.__version__) < '30.3':
    sys.stderr.write("ERROR: setuptools 30.3 or later is required by astropy-helpers\n")
    sys.exit(1)

from astropy_helpers.version_helpers import generate_version_py  # noqa
version = generate_version_py()

setup(version=version)
