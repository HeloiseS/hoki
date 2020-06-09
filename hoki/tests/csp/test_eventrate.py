"""
Test for the Complex Stellar Population event rate calculations
"""

import hoki.csp.eventrate as er
import pkg_resources
import numpy as np
from hoki.constants import *
import hoki.csp.utils as utils
from scipy import interpolate
import matplotlib.pyplot as plt

data_path = pkg_resources.resource_filename('hoki', 'data')
test_sfh = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
test_metallicity = np.loadtxt(f"{data_path}/csp_test_data/metallicity.txt")
test_mass_per_bin = np.loadtxt(f"{data_path}/csp_test_data/mass_per_bin.txt")
time_points = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")



class TestCSPEventRate():

    def test_init(self):
        _ = er.CSPEventRate(f"{data_path}/supernova")

    CSP = er.CSPEventRate(f"{data_path}/supernova")
    fnc_Z = interpolate.splrep(time_points, test_metallicity, k=1)
    fnc_sfh = interpolate.splrep(time_points, test_sfh, k=1)
    out = CSP.calculate_rate_over_time([fnc_Z], [fnc_sfh], ["Ia"],100)


    def test_bins(self):
        assert self.CSP.nr_bins == 100, "nr_bins not properly set."
        assert np.isclose(self.CSP.time_edges,  np.linspace(0,HOKI_NOW, 101)).all(),\
            "time edges are not properly set."

    def test_metallicity(self):
        assert np.isclose(self.CSP.metallicity_per_bin[0],
            np.loadtxt(f"{data_path}/csp_test_data/metallicity_per_bin.txt")).all(),\
            "The metallicity per bin is set incorrectly."

    def test_mass_per_bin(self):
        assert np.isclose(self.CSP.mass_per_bin[0],
            np.loadtxt(f"{data_path}/csp_test_data/mass_per_bin.txt")).all(),\
            "The mass per bin is incorrect."

    def test_event_rate_calculation(self):
        expected = np.loadtxt(f"{data_path}/csp_test_data/type_Ia_rates.txt")
        assert np.isclose(self.CSP.event_rates["Ia"], expected).all(),\
            "The event rate calculation is wrong."

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
