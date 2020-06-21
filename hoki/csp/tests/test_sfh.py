"""
Test the stellar formation history object
"""

from hoki.csp.sfh import SFH
import numpy as np
import pkg_resources
import pytest
from hoki.constants import *
from hoki.utils.exceptions import HokiKeyError, HokiTypeError

data_path = pkg_resources.resource_filename('hoki', 'data')
sfr = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
time_points = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")


class TestSFH(object):

    def test_intialisation(self):
        _ = SFH("custom",time_points, sfr)

    def test_model_not_recognised(self):
        with pytest.raises(HokiKeyError):
            _ = SFH("afs", time_points, sfr)

    def test_input_sizes(self):
        with pytest.raises(HokiTypeError):
            _ = SFH("custom", time_points[1:], sfr)

    sfh =  SFH("custom", time_points, sfr)

    def test_stellar_formation_rate(self):
        assert np.isclose(
                    [self.sfh.stellar_formation_rate(i) for i in time_points],
                    sfr).all(),\
            "Something is wrong with the stellar formation rate."

    def test_mass_per_bin(self):
        mass_per_bin = np.loadtxt(f"{data_path}/csp_test_data/mass_per_bin.txt")
        assert np.isclose(self.sfh.mass_per_bin(np.linspace(0, HOKI_NOW, 101)),
                          mass_per_bin).all(),\
            "The mass per bin is calculated incorrectly."
