"""
Tests for the CSP spectra subpackage.
"""
import numpy as np
import numpy.testing as npt
import pkg_resources

from hoki.csp.spectra import CSPSpectra

data_path = pkg_resources.resource_filename('hoki', 'data')


# Test functions
def sfh_fnc(x): return 1
def Z_fnc(x): return 0.00001


# Using the unittest.mock package, it should be possible to run these test with
# the CI. However, for now this will have to be run locally with a spectra
# folder containing the BPASS spectra for the imf135_300 IMF.
#

# class TestCSPSSpectra(object):
#
#     def test_init(self):
#         CSP = CSPSpectra(f"{data_path}/spectra",  "imf135_300")
#         assert CSP.shape(51, 13, 100000)
#
#
#     CSP = CSPSpectra(f"{data_path}/spectra",  "imf135_300")
#
#
#     def test_calculate_spec_at_time(self, mock_np):
#         spectra = self.CSP.calculate_spec_at_time([sfh_fnc],
#                                                   [Z_fnc],
#                                                    0)
#         assert spectra.shape == (1, 100000)
#         npt.assert_allclose(spectra, 1)
#
#     def test_calculate_spec_over_time(self, mock_np):
#         spectra = self.CSP.calculate_spec_over_time([sfh_fnc],
#                                                     [Z_fnc],
#                                                     100)
#
#         assert np.isclose(self.spectra[0], spectra[0][0], rtol=1).all()
#
#     def test_remove_test_files(self):
#
#       os.remove(f"{data_path}/all_spectra-bin-imf135_300.pkl")
