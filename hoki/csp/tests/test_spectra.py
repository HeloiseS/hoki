"""
Author: Max Briel

Tests for the CSP spectra subpackage.
"""
import os
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pkg_resources

from hoki.csp.spectra import CSPSpectra
from hoki.load import model_output
from hoki.constants import BPASS_NUM_METALLICITIES
import itertools
data_path = pkg_resources.resource_filename('hoki', 'data')

# Load Test spectra. Not an actual spectra.
test_spectra = np.loadtxt(f"{data_path}/csp_test_data/test_spectra.txt")
time_points = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")

# Test functions
def sfh_fnc(x): return 1

def Z_fnc(x): return 0.00001


class TestCSPSSpectra(object):

    # Load data to remove I/O for testing.
    data = model_output(
        f"{data_path}/spectra-bin-imf135_300.z002.dat").to_numpy()

    with patch("hoki.data_compilers.np.loadtxt") as mock_model_output:
        with patch("hoki.data_compilers.isfile") as mock_os:
            mock_os.return_value = True
            mock_model_output.return_value = data
            CSP = CSPSpectra(f"{data_path}",  "imf135_300")

    @patch("hoki.data_compilers.np.loadtxt")
    @patch("hoki.data_compilers.isfile")
    def test_init(self, mock_os, mock_model_output):
        mock_model_output.return_value = self.data
        mock_os.return_value = True
        CSP = CSPSpectra(f"{data_path}",  "imf135_300")
        assert CSP.bpass_spectra.shape == (
            13, 51, 100000), "Output shape is wrong."

    def test_compiled_file(self):
        assert os.path.isfile(f"{data_path}/all_spectra-bin-imf135_300.npy"),\
        "No compiled file is made."

    def test_calculate_spec_at_time(self):
        spectra = self.CSP.at_time([sfh_fnc], [Z_fnc], 0, sample_rate=-1)
        assert spectra.shape == (1, 100000)
        npt.assert_allclose(
            spectra[0], test_spectra, err_msg="calculate_spec_at_time output is wrong")

    def test_calculate_spec_over_time(self):
        spectra = self.CSP.over_time([sfh_fnc], [Z_fnc], 100)
        assert np.isclose(test_spectra[0], spectra[0][0], rtol=1).all(),\
            "Calculate_spec_over_time output is wrong."

    def test_grid_at_time(self):
        nr_time_points = len(time_points)
        SFH = np.zeros((1,13, nr_time_points), dtype=np.float64)
        bpass_Z_index = np.array([np.argmin(np.abs(Z_fnc(i) - BPASS_NUM_METALLICITIES)) for i in time_points])
        SFH[0,bpass_Z_index, range(nr_time_points)] += np.array([sfh_fnc(i) for i in time_points])


        out = self.CSP.grid_at_time(SFH , time_points, 0, sample_rate=100)
        out2 = self.CSP.at_time([sfh_fnc], [Z_fnc], 0, sample_rate=100)

        assert out.shape == (1, 13, 100000), "Output shape is wrong"
        assert np.allclose(out[0].sum(axis=0),out2[0]), "grid not the same as funtion"

    def test_grid_over_time(self):
        nr_time_points = len(time_points)
        SFH = np.zeros((1,13, nr_time_points), dtype=np.float64)
        bpass_Z_index = np.array([np.argmin(np.abs(Z_fnc(i) - BPASS_NUM_METALLICITIES)) for i in time_points])
        SFH[0,bpass_Z_index, range(nr_time_points)] += np.array([sfh_fnc(i) for i in time_points])


        out = self.CSP.grid_over_time(SFH , time_points, 10)
        out2 = self.CSP.over_time([sfh_fnc], [Z_fnc], 10)

        assert out.shape == (1, 13, 10, 100000), "grid_over_time Output shape is wrong"
        assert np.allclose(out[0].sum(axis=0),out2[0]), "grid not the same at function"

    def test_remove_test_files(self):
        os.remove(f"{data_path}/all_spectra-bin-imf135_300.npy")
