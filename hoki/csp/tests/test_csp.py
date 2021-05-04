"""
Author: Max Briel

Tests for the CSP parent class
"""
import numpy as np
import pytest

from hoki.csp.csp import CSP
from hoki.csp.sfh import SFH
from hoki.utils.exceptions import HokiFormatError, HokiKeyError, HokiTypeError
from hoki.constants import HOKI_NOW

#################################################
# Test Complex Stellar Populations Parent Class #
#################################################


class TestCSP(object):

    # Test `now` attribute
    def test_init(self):
        csp = CSP()
        assert csp.now == HOKI_NOW, "CSP parent class initialisation failed."

    def test_type_check_history(self):
        csp = CSP()

        # define pyton callable
        def x(i): return i

        # define SFH object
        time_axis = np.linspace(0, 13e9, 1000)
        sfh = SFH(time_axis, "b", {"constant": 10, "T0": 5e9})

        # Check Types
        with pytest.raises(HokiTypeError):
            csp._type_check_histories([10], [0])
        with pytest.raises(HokiTypeError):
            csp._type_check_histories([10], 0)
        with pytest.raises(HokiTypeError):
            csp._type_check_histories([x, x], [x, 10])

        # Check Format
        with pytest.raises(HokiFormatError):
            csp._type_check_histories([x], [x, x])
        with pytest.raises(HokiFormatError):
            csp._type_check_histories([x, x], [x])
        with pytest.raises(HokiFormatError):
            csp._type_check_histories([x], [])

        # Checking if the correct input does run
        assert csp._type_check_histories([x], [x]) == ([x], [x])
        assert csp._type_check_histories([x, x], [x, x]) == ([x, x], [x, x])
        assert csp._type_check_histories(x, x) == ([x], [x])
        assert csp._type_check_histories([x], x) == ([x], [x])
        assert csp._type_check_histories(sfh, x) == ([sfh], [x])
        assert csp._type_check_histories([sfh], [x]) == ([sfh], [x])
