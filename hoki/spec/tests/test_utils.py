import pandas as pd
from hoki import load
from hoki.spec import utils as su
import numpy as np
import os
import pkg_resources
import pytest
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiFormatWarning

data_path = pkg_resources.resource_filename('hoki', 'data')

spectra = load.model_output(data_path+'/spectra-bin-imf135_300.z008.dat')
spectra_b=spectra[(spectra.WL>4000) & (spectra.WL<6000)]
lower, line_bound, upper =[5743,5760], [5760,5855], [5855,5872]


def test_dopcor():
    ngc4993 = pd.read_csv(data_path+'/ngc4993_spec_tot.dat')
    wl0_init = ngc4993.wl[0]
    su.dopcor(ngc4993, 0.009783)
    assert np.isclose((wl0_init-ngc4993.wl[0]), 46.4655, atol=1e-05), "The doppler correction isn't right"


def test_pseudo_continuum():
    cont = su.pseudo_continuum(spectra_b.WL, spectra_b['6.6'], lower, upper)
    assert np.isclose(cont[0], 12099.23805719865), "Continuum is wrong"


def test_error_pseudo_continuum():
    with pytest.raises(HokiFormatError):
        __, __ = su.pseudo_continuum(spectra_b.WL, spectra_b['6.6'], '2', upper)


def test_equivalent_width_line_bound():
    ew = su.equivalent_width(spectra_b.WL, spectra_b['6.6'], lower, upper, line_bound)
    assert np.isclose(ew, -9.727277922413643), "EW is wrong"

def test_equivalent_width_no_line_bound():
    ew = su.equivalent_width(spectra_b.WL, spectra_b['6.6'], lower, upper)
    assert np.isclose(ew, -9.727277922413643), "EW is wrong"