"""
Author: Max Briel

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
from hoki.utils.exceptions import HokiFormatError, HokiKeyError, HokiTypeError

data_path = pkg_resources.resource_filename('hoki', 'data')

#############################
# Test Calculations per bin #
#############################

def test_optimised_trapezodial_rule():
    x = np.linspace(0,100,101)
    y = x
    out = utils._optimised_trapezodial_rule(y,x)
    assert np.isclose(out, (100**2)/2), "optimised trapezodial rule failed"

def test_optimised_mass_per_bin():
    x = np.linspace(0, 100, 101)
    y = np.zeros(101) + 1
    out = utils._optimised_mass_per_bin(x, y, np.linspace(0,10,11), 25)
    assert np.allclose(out, np.zeros(10) + 1), "optmised mass calculation is wrong"

def test_mass_per_bin():
    x = np.linspace(0, 100, 101)
    y = np.zeros(101) + 1

    def sfh_func(i): return np.interp(i, x, y)

    mass_per_bin = utils.mass_per_bin(sfh_func, np.linspace(0, 10, 11))
    assert np.allclose(mass_per_bin, np.zeros(10) + 1),\
        "mass_per_bin calculation wrong."

def test_mass_per_bin_vector_input():
    x = np.linspace(0, 100, 101)
    y = np.zeros(101) + 1

    def sfh_func(i): return np.zeros(len(i)) + 1

    mass_per_bin = utils.mass_per_bin(sfh_func, np.linspace(0, 10, 11))
    assert np.allclose(mass_per_bin, np.zeros(10) + 1),\
        "mass_per_bin calculation wrong."


def test_metallicity_per_bin():
    x = np.linspace(0, 100, 101)

    def Z_func(i): return np.interp(i, x, x)

    out = utils.metallicity_per_bin(Z_func, x)
    expected = np.arange(0.5, 100, 1)
    assert np.isclose(out, expected).all(), "Z per bin has failed"



################################
#  Test Normalise BPASS Files  #
################################


def test_normalise_rates():
    rates = pd.DataFrame(np.linspace(0, 100, 51))
    out = utils._normalise_rates(rates)
    expected = pd.DataFrame(np.linspace(0, 100, 51) /
                            1e6 / BPASS_LINEAR_TIME_INTERVALS)
    assert np.isclose(out, expected).all(), "Rate normalisation failed"


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
    rates = np.zeros((13, 51)) + 1

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
