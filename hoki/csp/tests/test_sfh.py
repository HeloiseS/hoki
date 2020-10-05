"""
Author: Max Briel & Heloise Stevance

Test the stellar formation history object
"""

import numpy as np
import pkg_resources
import pytest

from hoki.constants import *
from hoki.csp.sfh import SFH
from hoki.utils.exceptions import HokiFormatError, HokiKeyError, HokiTypeError

data_path = pkg_resources.resource_filename('hoki', 'data')
sfr = np.loadtxt(f"{data_path}/csp_test_data/mass_points.txt")
time_axis = np.loadtxt(f"{data_path}/csp_test_data/time_points.txt")


class TestSFHCustom(object):

    def test_intialisation(self):
        _ = SFH(time_axis, sfh_type="custom", sfh_arr=sfr)

    def test_model_not_recognised(self):
        with pytest.raises(HokiTypeError):
            _ = SFH(time_axis, sfh_type="afs", sfh_arr=sfr)

    def test_input_sizes(self):
        with pytest.raises(HokiFormatError):
            _ = SFH(time_axis[1:], sfh_type="custom", sfh_arr=sfr)

    sfh = SFH(time_axis, sfh_type="custom", sfh_arr=sfr)

    def test_sfr_at(self):
        assert np.isclose(
            [self.sfh(i) for i in time_axis],
            sfr).all(),\
            "Something is wrong with the stellar formation rate."

    def test_mass_per_bin(self):
        mass_per_bin = np.loadtxt(
            f"{data_path}/csp_test_data/mass_per_bin.txt")
        result = self.sfh.mass_per_bin(np.linspace(0, HOKI_NOW, 101), sample_rate=100)
        assert np.allclose(result, mass_per_bin),\
            "The mass per bin is calculated incorrectly."


class TestSFHParametric(object):
    def test_intialisation_c(self):
        _ = SFH(time_axis/1e9, sfh_type="c", parameters_dict={'constant': 1})

    def test_intialisation_b(self):
        _ = SFH(time_axis/1e9, sfh_type="b",
                parameters_dict={'T0': 1, 'constant': 1})

    def test_intialisation_e(self):
        _ = SFH(time_axis/1e9, sfh_type="e",
                parameters_dict={'tau': 1, 'T0': 1, 'constant': 1})

    def test_intialisation_de(self):
        _ = SFH(time_axis/1e9, sfh_type="de",
                parameters_dict={'tau': 1, 'T0': 1, 'constant': 1})

    def test_intialisation_dpl(self):
        _ = SFH(time_axis/1e9, sfh_type="dpl",
                parameters_dict={'tau': 1, 'alpha': 1, 'beta': 1, 'constant': 1})

    def test_intialisation_ln(self):
        _ = SFH(time_axis/1e9, sfh_type="ln",
                parameters_dict={'tau': 1, 'T0': 1, 'constant': 1})

    sfh = SFH(time_axis, sfh_type="c", parameters_dict={'constant': 1})

    def test_sfr_at(self):
        # this only works for constant sfr = 1
        assert np.isclose([self.sfh(i) for i in time_axis], [1]*len(time_axis)).all(),\
            "Something is wrong with the stellar formation rate."

    def test_mass_per_bin(self):
        assert np.isclose(self.sfh.mass_per_bin([2, 3, 6, 8]), [1., 3., 2.]).all(),\
            "The mass per bin is calculated incorrectly."
