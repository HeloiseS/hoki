"""
Test for the Complex Stellar Population event rate calculations
"""

from unittest.mock import patch

import numpy as np
import pkg_resources
import pytest

import hoki.csp.eventrate as er
import hoki.csp.utils as utils
from hoki.constants import *
from hoki.load import model_output
from hoki.utils.exceptions import HokiFatalError, HokiFormatError

data_path = pkg_resources.resource_filename('hoki', 'data')

test_sfh = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
test_metallicity = np.loadtxt(f"{data_path}/csp_test_data/metallicity.txt")
time_points = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")


# Make test functions from data
# sfh_fnc = lambda x : np.interp(x, time_points,  test_sfh)
# Z_fnc = lambda x : np.interp(x, time_points, test_metallicity)
def sfh_fnc(x): return 1
def Z_fnc(x): return 0.00001


class TestCSPEventRate():

    data = model_output(f"{data_path}/supernova-bin-imf135_300.zem5.dat")

    # Check initalisation
    @patch("hoki.load.model_output")
    def test_init(self, mock_model_output):
        mock_model_output.return_value = self.data
        _ = er.CSPEventRate(f"{data_path}", "imf135_300")

    # Load model_output with a single supernova rate file
    with patch("hoki.load.model_output") as mock_model_output:
        mock_model_output.return_value = data
        CSP = er.CSPEventRate(f"{data_path}", "imf135_300")

    out, time_edges = CSP.calculate_rate_over_time([sfh_fnc],
                                                   [Z_fnc],
                                                   ["Ia"],
                                                   100,
                                                   return_edges=True)

    def test_bins(self):
        assert np.isclose(self.time_edges,  np.linspace(0, HOKI_NOW, 101)).all(),\
            "time edges are not properly set."

    def test_event_rate_calculation(self):
        expected = np.loadtxt(f"{data_path}/csp_test_data/type_Ia_rates.txt")
        assert np.isclose(self.out["Ia"], expected).all(),\
            "The event rate calculation is wrong."

    def test_event_rate_wrong_input(self):
        with pytest.raises(HokiFormatError):
            _ = self.CSP.calculate_rate_over_time([sfh_fnc],
                                                  [Z_fnc, Z_fnc],
                                                  ["Ia"],
                                                  100)
        with pytest.raises(KeyError):
            _ = self.CSP.calculate_rate_over_time([sfh_fnc],
                                                  [Z_fnc],
                                                  ["B"],
                                                  100)
        with pytest.raises(HokiFormatError):
            _ = self.CSP.calculate_rate_over_time([sfh_fnc, sfh_fnc],
                                                  [Z_fnc],
                                                  ["B"],
                                                  100)

    def test_event_rate_calculation_multi_type(self):
        out = self.CSP.calculate_rate_over_time(
            [sfh_fnc], [Z_fnc], ["Ia", "II"], 100)
        assert len(out) == 1
        assert len(out[0]) == 2

    def test_event_rate_calculation_multi(self):
        out = self.CSP.calculate_rate_over_time(
            [sfh_fnc, sfh_fnc], [Z_fnc, Z_fnc], ["Ia"], 100)
        assert len(out) == 2
        assert len(out[0]) == 1

    def test_event_rate_at_time(self):
        x = self.CSP.calculate_rate_at_time([sfh_fnc], [Z_fnc], ["Ia"], 0)
        assert np.isclose(x[0]["Ia"], 0.002034966495416449)
