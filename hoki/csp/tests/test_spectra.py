"""
Tests for the CSP spectra subpackage.
"""
import os
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pkg_resources

from hoki.csp.spectra import CSPSpectra
from hoki.load import model_output

data_path = pkg_resources.resource_filename('hoki', 'data')

# Load Test spectra. Not an actual spectra.
test_spectra = np.loadtxt(f"{data_path}/csp_test_data/test_spectra.txt")


# Test functions
def sfh_fnc(x): return 1
def Z_fnc(x): return 0.00001


class TestCSPSSpectra(object):

    # Load data to remove I/O for testing.
    data = model_output(
        f"{data_path}/spectra-bin-imf135_300.z002.dat").loc[:, slice("6.0", "11.0")]

    with patch("hoki.data_compilers.model_output") as mock_model_output:
        mock_model_output.return_value = data
        CSP = CSPSpectra(f"{data_path}",  "imf135_300")

    @patch("hoki.data_compilers.model_output")
    def test_init(self, mock_model_output):
        mock_model_output.return_value = self.data
        CSP = CSPSpectra(f"{data_path}",  "imf135_300")
        assert CSP.bpass_spectra.shape == (
            13, 51, 100000), "Output shape is wrong."

    def test_compiled_file(self):
        assert os.path.isfile(f"{data_path}/all_spectra-bin-imf135_300.npy"),
        "No compiled file is made."

    def test_calculate_spec_at_time(self):
        spectra = self.CSP.calculate_spec_at_time([sfh_fnc], [Z_fnc], 0)
        assert spectra.shape == (1, 100000)
        npt.assert_allclose(
            spectra[0], test_spectra, err_msg="calculate_spec_at_time output is wrong")

    def test_calculate_spec_over_time(self):
        spectra = self.CSP.calculate_spec_over_time([sfh_fnc],
                                                    [Z_fnc],
                                                    100)
        assert np.isclose(test_spectra[0], spectra[0][0], rtol=1).all(),\
            "Calculate_spec_over_time output is wrong."

    def test_remove_test_files(self):
        os.remove(f"{data_path}/all_spectra-bin-imf135_300.npy")
