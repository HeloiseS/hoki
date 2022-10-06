import numpy as np
import hoki.hrdiagrams as hr
import hoki.load as load
import pkg_resources
from hoki.utils.exceptions import HokiFatalError
import pytest

data_path = pkg_resources.resource_filename('hoki', 'data')

hr_file = data_path+'/hrs-sin-imf_chab100.zem4.dat'

hrtl = load._hrTL(hr_file)
hrtg = load._hrTg(hr_file)
hrttg = load._hrTTG(hr_file)

def test_error():
    with pytest.raises(HokiFatalError):
        test_plot = hrtl.plot(abundances=(0,0,0))

# I cant' test whether matplotlib is doing the right thing but I can check it runs.
def test_stack():
    hrtl.stack(log_age_min=6.1)
    hrtl.stack(log_age_max=7.0)
    hrtl.stack(log_age_min=6.2, log_age_max=6.8)


def test_at_log_age():
    hrtg.at_log_age(6.7)


def test_plot_hrtl():
    test_plot = hrtl.plot()
    test_plot = hrtl.plot(log_age=6.8)
    test_plot = hrtl.plot(age_range=(6.2, 6.75), levels=2)
    test_plot = hrtl.plot(log_age=6.8, loc=221, abundances = (1,1,1), cmap='Reds_r')
    del test_plot


def test_plot_hrtg():
    test_plot = hrtg.plot()
    test_plot = hrtg.plot(log_age=6.8)
    test_plot = hrtg.plot(age_range=(6.2, 6.75), levels=2)
    test_plot = hrtg.plot(log_age=6.8, loc=221, abundances = (1,1,1), cmap='Reds_r')
    del test_plot


def test_plot_hrttg():
    test_plot = hrttg.plot()
    test_plot = hrttg.plot(log_age=6.8)
    test_plot = hrttg.plot(age_range=(6.2, 6.75), levels=2)
    test_plot = hrttg.plot(log_age=6.8, loc=221, abundances = (1,1,1), cmap='Reds_r')
    del test_plot
