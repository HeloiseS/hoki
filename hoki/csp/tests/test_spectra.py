"""
Tests for the CSP spectra subpackage.
"""
import numpy as np
import pkg_resources
from  hoki.csp.spectra import CSPSpectra
from scipy import interpolate


data_path = pkg_resources.resource_filename('hoki', 'data')

test_sfh = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
test_metallicity = np.loadtxt(f"{data_path}/csp_test_data/metallicity.txt")
time_points = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")

# This test only works when the spectra folder is present in the data folder
"""
class TestCSPSSpectra(object):

    def test_init(self):
        _ = CSPSpectra(f"{data_path}/spectra")

    CSP = CSPSpectra(f"{data_path}/spectra")
    fnc_Z = interpolate.splrep(time_axis, test_metallicity, k=1)
    fnc_sfh = interpolate.splrep(time_axis, test_sfh, k=1)

    def test_calculate_spec_over_time(self):
        spectra, edges = self.CSP.calculate_spec_over_time([self.fnc_Z],
                                                           [self.fnc_sfh],
                                                           10,
                                                           return_edges=True)


"""
