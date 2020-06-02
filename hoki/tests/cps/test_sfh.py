"""
Test the stellar formation history object
"""

from hoki.csp.sfh import SFH
import numpy as np
import pkg_resources
import pytest

data_path = pkg_resources.resource_filename('hoki', 'data')
test_sfr_file =np.loadtxt(f"{data_path}/test_SFH.txt")
time_bins = test_sfr_file[0]*1e9
sfr = test_sfr_file[1]
print(sfr)

class TestSFH(object):

    def test_intialisation(self):
        _ = SFH(time_bins, sfr, model_type="custom")

    def test_model_not_recognised(self):
        with pytest.raises(TypeError):
            _ = SFH(time_bins, sfr, model_type="afs")

    def test_input_sizes(self):
        with pytest.raises(TypeError):
            _ = SFH(time_bins[1:], sfr, model_type="custom")

    sfh =  SFH(time_bins, sfr, model_type="custom")
    def test_stellar_formation_rate(self):
        assert np.sum(
            [np.isclose(self.sfh.stellar_formation_rate(time_bins[0]),
                       sfr[0]),
            np.isclose(self.sfh.stellar_formation_rate(time_bins[10]),
                       sfr[10])]) == 2, \
            "Something is wrong with the stellar formation rate."

    def test_mass_per_bin(self):
        mass_per_bin = np.loadtxt(f"{data_path}/mass_per_bin.txt")
        assert np.isclose(self.sfh.mass_per_bin(np.linspace(0,13.8*1e9,100)),
                          mass_per_bin).all(),\
            "The mass per bin is calculated incorrectly."
