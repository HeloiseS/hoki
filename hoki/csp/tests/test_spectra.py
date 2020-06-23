"""
Tests for the CSP spectra subpackage.
"""
import numpy as np
import pkg_resources
from  hoki.csp.spectra import CSPSpectra
import matplotlib.pyplot as plt
data_path = pkg_resources.resource_filename('hoki', 'data')

test_sfh = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
test_metallicity = np.loadtxt(f"{data_path}/csp_test_data/metallicity.txt")
time_axis = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")

# Make test functions from data
sfh_fnc = lambda x : np.interp(x, time_axis,  test_sfh)
Z_fnc = lambda x : np.interp(x, time_axis, test_metallicity)

# This test only works when the spectra folder is present in the data folder
# this folder should contain all imf135_300 BPASS spectra files.
class TestCSPSSpectra(object):

    def test_init(self):
        _ = CSPSpectra(f"{data_path}/spectra",  "imf135_300")

    CSP = CSPSpectra(f"{data_path}/spectra",  "imf135_300")

    def test_calculate_spec_at_time(self):
        spectra = self.CSP.calculate_spec_at_time([sfh_fnc],
                                                  [Z_fnc],
                                                   0)

    spectra = CSP.calculate_spec_at_time([sfh_fnc],
                                            [Z_fnc],
                                             0, 1000)

    def test_calculate_spec_over_time(self):
        spectra = self.CSP.calculate_spec_over_time([sfh_fnc],
                                                    [Z_fnc],
                                                    100)
        # A relative high tolerance, because the _over_time calculation is a
        # slow and therefore has a larger error than the _at_time calculation
        # By increasing the bins in the _over_time calculation, the tolerance
        # can be decreased. Right now, it's extremely high
        assert np.isclose(self.spectra[0], spectra[0][0], rtol=1).all()
