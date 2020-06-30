"""
Test for the CPS utility subpackages
"""
import os
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pkg_resources
import pytest

import hoki.csp.utils as utils
import hoki.load
from hoki.constants import *
from hoki.csp.sfh import SFH
from hoki.utils.exceptions import HokiFormatError, HokiKeyError, HokiTypeError

data_path = pkg_resources.resource_filename('hoki', 'data')

#################################################
# Test Complex Stellar Populations Parent Class #
#################################################


class TestCSP(object):

    # Test `now` attribute
    def test_init(self):
        csp = utils.CSP()
        assert csp.now == HOKI_NOW, "CSP parent class initialisation failed."

    def test_type_check_history(self):
        csp = utils.CSP()

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


#############################
# Test Calculations per bin #
#############################


def test_mass_per_bin():
    x = np.linspace(0, 100, 101)
    y = np.zeros(101) + 1

    def sfh_func(i): return np.interp(i, x, y)

    mass_per_bin = utils.mass_per_bin(sfh_func, np.linspace(0, 10, 11))
    assert np.isclose(mass_per_bin, np.zeros(10) + 1).all(),\
        "mass_per_bin calculation wrong."


def test_metallicity_per_bin():
    x = np.linspace(0, 100, 101)

    def Z_func(i): return np.interp(i, x, x)

    out = utils.metallicity_per_bin(Z_func, x)
    expected = np.arange(0.5, 100, 1)
    assert np.isclose(out, expected).all(), "Z per bin has failed"

#############################
#  Test BPASS File Loading  #
#############################


class TestLoadRates(object):

    # Setup files to load
    data = hoki.load.model_output(
        f"{data_path}/supernova-bin-imf135_300.zem5.dat")

    # Check if function loads rates
    @patch("hoki.load.model_output")
    def test_load_rates(self, mock_model_output):
        mock_model_output.return_value = self.data
        x = utils.load_rates(f"{data_path}", "imf135_300"),\
            "The rates cannot be initialised."

    # Load rates
    with patch("hoki.load.model_output") as mock_model_output:
        mock_model_output.return_value = data
        x = utils.load_rates(f"{data_path}", "imf135_300")

    # Test wrong inputs
    def test_file_not_present(self):
        with pytest.raises(AssertionError):
            _ = utils.load_rates(f"{data_path}", "imf135_300"),\
                "The file is not present, but the load function runs."

    def test_wrong_imf(self):
        with pytest.raises(HokiKeyError):
            _ = utils.load_rates(f"{data_path}", "i"),\
                "An unsupported IMF is taken as an input."

    # Test output
    def test_output_shape(self):
        assert type(self.x) == pd.DataFrame
        assert (self.x.columns.get_level_values(0).unique() ==
                np.array(BPASS_EVENT_TYPES)).all(),\
            "wrong headers read from the file."
        assert (self.x.columns.get_level_values(1).unique() ==
                np.array(BPASS_NUM_METALLICITIES)).all(),\
            "wrong metallicity header"

    def test_output(self):
        assert np.isclose(self.x.loc[:, ("Ia", 0.00001)],
                          self.data["Ia"]).all(),\
            "Models are not loaded correctly."


class TestLoadSpectra(object):

    # Initialise model_output DataFrame
    # This reduces I/O readings
    data = hoki.load.model_output(
        f"{data_path}/spectra-bin-imf135_300.z002.dat").loc[:, slice("6.0", "11.0")]

    # Patch the model_output function
    @patch("hoki.data_compilers.model_output")
    def test_compile_spectra(self, mock_model_output):

        # Set the model_output to the DataFrame
        mock_model_output.return_value = self.data

        spec = utils.load_spectra(f"{data_path}", "imf135_300")

        # Check if compiled file is created
        assert os.path.isfile(f"{data_path}/all_spectra-bin-imf135_300.npy"),\
            "No compiled file is created."

        # Check output numpy array
        npt.assert_allclose(spec[3], self.data.T,
                            err_msg="Loading of files has failed.")

    def test_load_pickled_file(self):

        spec = utils.load_spectra(f"{data_path}", "imf135_300")

        # Check output numpy array
        npt.assert_allclose(spec[3], self.data.T,
                            err_msg="Loading of compiled file has failed.")

        os.remove(f"{data_path}/all_spectra-bin-imf135_300.npy")

################################
#  Test Normasise BPASS Files  #
################################


def test_normalise_rates():
    rates = pd.DataFrame(np.linspace(0, 100, 51))
    out = utils._normalise_rates(rates)
    expected = pd.DataFrame(np.linspace(0, 100, 51) /
                            1e6 / BPASS_LINEAR_TIME_INTERVALS)
    assert np.isclose(out, expected).all(), "Rate normalisation failed"


def test_normalise_spectrum():
    spectrum = pd.DataFrame(np.linspace(0, 100, 51))
    out = utils._normalise_spectrum(spectrum)
    expected = pd.DataFrame(np.linspace(0, 1e-4, 51))
    npt.assert_allclose(out, expected, err_msg="Spectrum is not normalised")


################################
#   Test BPASS Metallicities   #
################################

def test_find_bpass_metallicities():
    test_metallicities = [0.0001, 0.00003, 0.044, 0.021, 0.05]
    expected = [0.0001, 0.00001, 0.04, 0.02, 0.04]
    out = utils._find_bpass_metallicities(test_metallicities)
    assert np.isclose(out, expected).all(),\
        "find_bpass_metallicities not working"


#############################################
# Test Complex Stellar History Calculations #
#############################################

class TestRateCalculations(object):

    edges = np.linspace(0, 10, 11)
    Z_values = np.zeros(10) + 0.00001
    mass_values = np.zeros(10) + 1
    rates = np.zeros((1, 51)) + 1

    def test_over_time(self):
        out = utils._over_time(self.Z_values,
                               self.mass_values,
                               self.edges,
                               self.rates)
        npt.assert_allclose(np.linspace(10, 1, 10), out,
                            err_msg="_over_time calculation has failed.")

    def test_at_time_now(self):
        out = utils._at_time(self.Z_values,
                             self.mass_values,
                             self.edges,
                             self.rates)
        npt.assert_allclose(10, out, err_msg="_at_time has failed for t=0.")

    def test_at_time_past(self):
        out = utils._at_time(self.Z_values[2:],
                             self.mass_values[2:],
                             self.edges[2:],
                             self.rates)
        npt.assert_allclose(
            8, out, err_msg="_at_time has failed for a past time.")


class TestSpectraCalculations(object):

    edges = np.linspace(0, 10, 11)
    Z_values = np.zeros(10) + 0.00001
    mass_values = np.zeros(10) + 1

    def test_over_time(self):
        spectra = np.zeros((1, 100000, 51)) + 1
        out = utils._over_time_spectrum(self.Z_values,
                                        self.mass_values,
                                        self.edges,
                                        spectra)
        npt.assert_allclose(out[4], np.zeros(100000) + 6,
                            err_msg="_over_time_spectra calculation has failed.")

    def test_at_time_now(self):
        spectra = np.zeros((51, 1, 100000)) + 1
        out = utils._at_time_spectrum(self.Z_values,
                                      self.mass_values,
                                      self.edges,
                                      spectra)
        print(out)
        npt.assert_allclose(out, np.zeros(100000) + 10,
                            err_msg="_at_time_spectra calculation has failed for the now.")

    def test_at_time_past(self):
        spectra = np.zeros((51, 1, 100000)) + 1
        out = utils._at_time_spectrum(self.Z_values[5:],
                                      self.mass_values[5:],
                                      self.edges[5:],
                                      spectra)
        npt.assert_allclose(out, np.zeros(100000) + 5,
                            err_msg="_at_time_spectra calculation has failed for in the past.")


###############################
#    TEST HELPER FUNCTIONS    #
###############################
class TestIntegral(object):
    edges = np.linspace(0, 100, 101)
    values = np.zeros(100) + 1
    bin_widths = np.diff(edges)

    def test_within_bin(self):
        assert np.isclose(utils._integral(0.1, 0.9, self.edges, self.values, self.bin_widths), 0.8),\
            "The integral within a bin is wrong."

    def test_edge_to_edge(self):
        assert np.isclose(utils._integral(0, 1, self.edges, self.values, self.bin_widths), 1),\
            "The integral between bin edges is wrong."

    def test_adjecent_bins(self):
        assert np.isclose(utils._integral(0.5, 1.5, self.edges, self.values, self.bin_widths), 1),\
            "The integral between adjacent bins is wrong."

    def test_with_middle_bins(self):
        assert np.isclose(utils._integral(10.2, 20.2, self.edges, self.values, self.bin_widths), 10),\
            "The integral over multiple bins is wrong."


class TestGetBinIndex(object):
    edges = np.linspace(0, 100, 101)

    def test_below_lowest_edge(self):
        with pytest.raises(HokiFormatError):
            utils._get_bin_index(-1, self.edges)

    def test_above_highest_edge(self):
        with pytest.raises(HokiFormatError):
            utils._get_bin_index(101, self.edges)

    def test_in_bin(self):
        assert np.sum(np.isclose([utils._get_bin_index(0.05, self.edges),
                                  utils._get_bin_index(1.04, self.edges),
                                  utils._get_bin_index(99.95, self.edges)], [0, 1, 99]
                                 )) == 3, "The wrong bin for within the bin is given."

    def test_on_edge(self):
        assert utils._get_bin_index(1, self.edges) == 1,\
            "The wrong bin is returned for an edge."

    def test_inclusive_left_edge_bottom_bin(self):
        assert utils._get_bin_index(0, self.edges) == 0,\
            "The left edge of the bottom bin is not inclusive."

    def test_inclusive_right_edge_top_bin(self):
        assert utils._get_bin_index(100, self.edges) == 99,\
            "The right edge of the top bin in not inclusive."
