# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys
from distutils.version import LooseVersion
from hoki.constants import *

__minimum_python_version__ = "3.7"

__all__ = []


class UnsupportedPythonError(Exception):
    pass


if LooseVersion(sys.version) < LooseVersion(__minimum_python_version__):
    raise UnsupportedPythonError(
        "hoki does not support Python < {}".format(__minimum_python_version__)
    )
