"""
Test for the Complex Stellar Population event rate calculations
"""

import pkg_resources
import numpy as np
import pytest

from hoki.constants import *
from hoki.utils.exceptions import HokiFatalError
import hoki.csp.eventrate as er
import hoki.csp.utils as utils

data_path = pkg_resources.resource_filename('hoki', 'data')
test_sfh = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
test_metallicity = np.loadtxt(f"{data_path}/csp_test_data/metallicity.txt")
time_points = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")
test_mass_per_bin = np.loadtxt(f"{data_path}/csp_test_data/mass_per_bin.txt")


# ADD A TEST TO CHECK THE TYPE OF THE INPUT FUNCTIONS!
class TestCSPEventRate():

    def test_init(self):
        _ = er.CSPEventRate(f"{data_path}/supernova")

    CSP = er.CSPEventRate(f"{data_path}/supernova")
    out, time_edges = CSP.calculate_rate_over_time([time_points],
                                                   [test_metallicity],
                                                   [test_sfh],
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
        with pytest.raises(HokiFatalError):
            _ = self.CSP.calculate_rate_over_time([self.fnc_Z],
                                             [self.fnc_sfh, self.fnc_sfh],
                                             ["Ia"],
                                             100)
        with pytest.raises(KeyError):
            _ = self.CSP.calculate_rate_over_time([self.fnc_Z],
                                             [self.fnc_sfh],
                                             ["B"],
                                             100)
        with pytest.raises(HokiFatalError):
            _ = self.CSP.calculate_rate_over_time([self.fnc_Z, self.fnc_Z],
                                             [self.fnc_sfh],
                                             ["B"],
                                             100)


    def test_event_rate_calculation_multi_type(self):
        out = self.CSP.calculate_rate_over_time([self.fnc_Z], [self.fnc_sfh], ["Ia", "II"], 100)
        assert len(out) == 1
        assert len(out[0]) == 2

    def test_event_rate_calculation_multi(self):
        out = self.CSP.calculate_rate_over_time([self.fnc_Z, self.fnc_Z], [self.fnc_sfh, self.fnc_sfh], ["Ia"], 100)
        assert len(out) == 2
        assert len(out[0]) == 1

    def test_event_rate_at_time(self):
        x = self.CSP.calculate_rate_at_time([self.fnc_Z], [self.fnc_sfh], ["Ia"], 0)
        assert np.isclose(x[0]["Ia"], 0.01140687)
