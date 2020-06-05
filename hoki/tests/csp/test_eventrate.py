"""
Test for the Complex Stellar Population event rate calculations
"""

import hoki.csp.eventrate as er
import pkg_resources
import numpy as np
from hoki.constants import *
import hoki.csp.utils as utils
from scipy import interpolate


data_path = pkg_resources.resource_filename('hoki', 'data')
test_sfh = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
test_metallicity = np.loadtxt(f"{data_path}/csp_test_data/metallicity.txt")
test_mass_per_bin = np.loadtxt(f"{data_path}/csp_test_data/mass_per_bin.txt")
time_points = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")



class TestCSPEventRateOverTime():

    def test_init(self):
        CSP = er.CSPEventRateOverTime(f"{data_path}/supernova",100)
        assert CSP.nr_bins == 100, "nr_bins not properly set."
        assert np.isclose(CSP.time_edges,  np.linspace(0,HOKI_NOW, 101)).all(),\
            "time edges are not properly set."

    CSP = er.CSPEventRateOverTime(f"{data_path}/supernova",100)
    fnc_Z = interpolate.splrep(time_points, test_metallicity, k=1)
    fnc_sfh = interpolate.splrep(time_points, test_sfh, k=1)
    CSP.calculate_rate([fnc_Z], [fnc_sfh], ["Ia"])

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
        assert np.isclose(self.CSP.event_rates, expected).all(),\
            "The event rate calculation is wrong."
