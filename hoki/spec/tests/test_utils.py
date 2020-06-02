import pandas as pd
from hoki import load
from hoki.spec.utils import dopcor
import numpy as np
import os
import pkg_resources

data_path = pkg_resources.resource_filename('hoki', 'data')

#wavelengths and throughput/transmission manually read from the LICK.LICK.U file
wl = [3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100]
transmission = [0.015, 0.140,0.347,0.440,0.625,0.685,0.708,0.643,0.458,0.170,0.029]
data = load.model_output(f"{data_path}/spectra-bin-imf135_300.z002.dat")

def test_dopcor():
    ngc4993 = pd.read_csv(data_path+'/ngc4993_spec_tot.dat')
    wl0_init = ngc4993.wl[0]
    dopcor(ngc4993,  0.009783)
    assert np.isclose((wl0_init-ngc4993.wl[0]), 46.4655, atol=1e-05), "The doppler correction isn't right"