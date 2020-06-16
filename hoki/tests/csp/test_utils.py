"""
Test for the CPS utility subpackages
"""
import hoki.csp.utils as utils
import pkg_resources
import numpy as np
import hoki.load
import pytest
from scipy import interpolate
import pandas as pd
from hoki.constants import *

data_path = pkg_resources.resource_filename('hoki', 'data')


class TestLoadFiles(object):

    def test_load_file(self):
        _ = utils._load_files(f"{data_path}/supernova", "supernova")

    def test_file_not_present(self):
        with pytest.raises(AssertionError):
            _ = utils._load_files(f"{data_path}", "supernova")

    x = utils._load_files(f"{data_path}/supernova", "supernova")

    def test_output_shape(self):
        assert type(self.x) == pd.DataFrame
        assert (self.x.columns.get_level_values(0).unique() ==
                np.array(BPASS_EVENT_TYPES)).all(),\
                "wrong headers read from the file."
        assert (self.x.columns.get_level_values(1).unique() ==
                np.array(BPASS_NUM_METALLICITIES)).all(),\
                "wrong metallicity header"

    def test_output(self):
        expected = hoki.load.model_output(f"{data_path}/supernova/supernova-bin-imf135_300.zem5.dat")
        assert np.isclose(self.x.loc[:,("Ia",0.00001)],
            expected["Ia"]).all(),\
            "Models are not loaded correctly."

class TestGetBinIndex(object):
    edges = np.linspace(0,100, 101)

    def test_below_lowest_edge(self):
        with pytest.raises(Exception):
            utils._get_bin_index(-1, self.edges)

    def test_above_highest_edge(self):
        with pytest.raises(Exception):
            utils._get_bin_index(101, self.edges)

    def test_in_bin(self):
        assert np.sum(np.isclose([utils._get_bin_index(0.05, self.edges),
                                  utils._get_bin_index(1.04, self.edges),
                                  utils._get_bin_index(99.95, self.edges)],[0,1,99]
        )) == 3, "The wrong bin for within the bin is given."

    def test_on_edge(self):
        assert utils._get_bin_index(1, self.edges) == 1,\
            "The wrong bin is returned for an edge."

    def test_inclusive_left_edge_bottom_bin(self):
        assert utils._get_bin_index(0,self.edges) == 0,\
            "The left edge of the bottom bin is not inclusive."

    def test_inclusive_right_edge_top_bin(self):
        assert utils._get_bin_index(100,self.edges) == 99,\
            "The right edge of the top bin in not inclusive."


class TestIntegral(object):
    edges = np.linspace(0,100,101)
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


def test_mass_per_bin():
    fnc_mass = interpolate.splrep(np.linspace(0,100, 101), np.zeros(101)+1, k=1)
    assert np.isclose(utils.mass_per_bin(fnc_mass, np.linspace(0,10, 11)),
                      np.zeros(10)+1).all(), "mass_per_bin calculation wrong."


def test_metallicity_per_bin():
    x = np.linspace(0,100,101)
    fnc_metallicity = interpolate.splrep(x,
                                         x,
                                         k=1)

    out = utils.metallicity_per_bin(fnc_metallicity, x)
    expected = np.arange(0.5, 100, 1)
    assert np.isclose(out, expected).all(), "Z per bin has failed"


def test_normalise_rates():
    rates = pd.DataFrame(np.linspace(0,100, 51))
    out = utils._normalise_rates(rates)
    expected = pd.DataFrame(np.linspace(0,100,51)/1e6/BPASS_LINEAR_TIME_INTERVALS)
    assert np.isclose(out, expected).all(), "Rate normalisation failed"


def test_find_bpass_metallicities():
    test_metallicities = [0.0001, 0.00003, 0.044, 0.021, 0.05]
    expected = [0.0001, 0.00001, 0.04, 0.02, 0.04]
    out = utils._find_bpass_metallicities(test_metallicities)
    assert np.isclose(out, expected).all(), "find_bpass_metallicities not working"
