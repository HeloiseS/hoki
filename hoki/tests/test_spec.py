import pandas as pd
from hoki import spec
import numpy as np
import pkg_resources

data_path = pkg_resources.resource_filename('hoki', 'data')

def test_dopcor():
    ngc4993 = pd.read_csv(data_path+'/ngc4993_spec_tot.dat')
    wl0_init = ngc4993.wl[0]
    spec.dopcor(ngc4993,  0.009783)
    assert np.isclose((wl0_init-ngc4993.wl[0]), 46.4655, atol=1e-05), "The doppler correction isn't right"