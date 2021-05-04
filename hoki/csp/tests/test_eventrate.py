"""
Author: Max Briel

Test for the Complex Stellar Population event rate calculations
"""

from unittest.mock import patch

import numpy as np
import pkg_resources
import pytest

import hoki.csp.eventrate as er
import hoki.csp.utils as utils
from hoki.constants import *
from hoki.csp.sfh import SFH
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

def vec_sfh(x): return np.zeros(len(x)) +1
def vec_Z(x): return np.zeros(len(x)) + 0.00001

# define SFH object
time_axis = np.linspace(0, HOKI_NOW, 1000)
sfh = SFH(time_axis, "c", {"constant": 1})


class TestCSPEventRate():

    data = model_output(f"{data_path}/supernova-bin-imf135_300.zem5.dat")

    # Check initalisation
    @patch("hoki.load.model_output")
    def test_init(self, mock_model_output):
        mock_model_output.return_value = self.data
        _ = er.CSPEventRate(f"{data_path}", "imf135_300")

    def test_input_functions_at_time(self):
        assert np.isclose(self.CSP.at_time([sfh_fnc], [Z_fnc], ["Ia"], 0, sample_rate=-1)[0]["Ia"],
                          0.002034966495416449), "Correct input is not taken."

        assert np.isclose(self.CSP.at_time(sfh_fnc, Z_fnc, ["Ia"], 0, sample_rate=-1)[0]["Ia"],
                          0.002034966495416449), "Correct input is not taken."
        assert np.isclose(self.CSP.at_time(sfh, Z_fnc, ["Ia"], 0, sample_rate=-1)[0]["Ia"],
                          0.002034966495416449), "Correct input is not taken."

        assert np.isclose(self.CSP.at_time([sfh], Z_fnc, ["Ia"],0)[0]["Ia"],
                          0.0018987009588956765), "Correct input is not taken."
        assert np.isclose(self.CSP.at_time([sfh, sfh], [Z_fnc, Z_fnc], ["Ia"], 0)[0]["Ia"],
                          0.0018987009588956765), "Correct input is not taken."

    def test_event_rate_wrong_input(self):
        with pytest.raises(HokiFormatError):
            _ = self.CSP.over_time([sfh_fnc], [Z_fnc, Z_fnc], ["Ia"],100)
        with pytest.raises(ValueError):
            _ = self.CSP.over_time([sfh_fnc], [Z_fnc], ["B"], 100)
        with pytest.raises(HokiFormatError):
            _ = self.CSP.over_time([sfh_fnc, sfh_fnc], [Z_fnc], ["B"], 100)

    @patch("hoki.load.model_output")
    def test_input_over_time(self,mock_model_output):
        # Load model_output with a single supernova rate file
        mock_model_output.return_value = self.data
        CSP = er.CSPEventRate(f"{data_path}", "imf135_300")

        test_out, time_edges = CSP.over_time([sfh_fnc], [Z_fnc], ["Ia"],
                                             100, return_time_edges=True)


    # Load model_output with a single supernova rate file
    with patch("hoki.load.model_output") as mock_model_output:
        mock_model_output.return_value = data
        CSP = er.CSPEventRate(f"{data_path}", "imf135_300")

        test_out, time_edges = CSP.over_time([sfh_fnc], [Z_fnc], ["Ia"],
                                             100, return_time_edges=True)

    def test_bins(self):
        assert np.isclose(self.time_edges,  np.linspace(0, HOKI_NOW, 101)).all(),\
            "time edges are not properly set."

    def test_event_rate_calculation(self):
        expected = np.loadtxt(f"{data_path}/csp_test_data/type_Ia_rates.txt")
        assert np.isclose(self.test_out["Ia"], expected).all(),\
            "The event rate calculation is wrong."

    def test_event_rate_calculation_multi_type(self):
        out = self.CSP.over_time(
            [sfh_fnc], [Z_fnc], ["Ia", "II"], 100)
        assert len(out) == 1, "The output of calculate_over_time is wrong."
        assert len(out[0]) == 2, "The output of calculate_over_time is wrong."

    def test_event_rate_calculation_multi(self):
        out = self.CSP.over_time(
            [sfh_fnc, sfh_fnc], [Z_fnc, Z_fnc], ["Ia"], 100)
        assert len(out) == 2, "The output ofcalculate_over_time is wrong."
        assert len(out[0]) == 1, "The output of calculate_over_time is wrong."

    def test_event_rate_at_time(self):
        x = self.CSP.at_time([sfh_fnc], [Z_fnc], ["Ia"], 0)
        assert np.isclose(x[0]["Ia"], 0.0018987009588956765),\
            "The output of CSP.at_time is wrong."

    def test_vector_input(self):
        assert np.isclose(self.CSP.at_time([vec_sfh], [vec_Z], ["Ia"], 0, sample_rate=-1)[0]["Ia"],
                0.002034966495416449),"Correct input is not taken."

    def test_full_grid_over_time(self):

        # Build mock 2D grid
        nr_time_points = len(time_points)
        SFH = np.zeros((1,13, nr_time_points), dtype=np.float64)
        bpass_Z_index = np.array([np.argmin(np.abs(Z_fnc(i) - BPASS_NUM_METALLICITIES)) for i in time_points])
        SFH[0,bpass_Z_index, range(nr_time_points)] += np.array([sfh_fnc(i) for i in time_points])

        out = self.CSP.grid_over_time(SFH ,time_points , ["Ia", "IIP"], 100)
        assert out.shape == (1, 13, 2, 100), "Output shape is wrong"
        assert np.allclose(out[0][:,0].sum(axis=0), self.test_out["Ia"]), "Not the same as over_time"

    def test_full_grid_at_time(self):

        # Build mock 2D grid
        nr_time_points = len(time_points)
        SFH = np.zeros((1,13, nr_time_points), dtype=np.float64)
        bpass_Z_index = np.array([np.argmin(np.abs(Z_fnc(i) - BPASS_NUM_METALLICITIES)) for i in time_points])
        SFH[0,bpass_Z_index, range(nr_time_points)] += np.array([sfh_fnc(i) for i in time_points])

        out = self.CSP.grid_at_time(SFH , time_points, ["Ia", "IIP"], 0)
        assert out.shape == (1, 13, 2), "Output shape is wrong"
        assert np.isclose(out[0][:,0].sum(axis=0),  0.0018987009588956765), "Not the same as over_time"
