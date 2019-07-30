import numpy as np
import hoki.hrdiagrams as hr
import hoki.load as load
import pkg_resources

data_path = pkg_resources.resource_filename('hoki', 'data')

hr_file = data_path+'/hrs-sin-imf_chab100.zem4.dat'

hrtl = load.hrTL(hr_file)
hrtg = load.hrTg(hr_file)
hrttg = load.hrTTG(hr_file)


class testHRDiagrams():
    # I cant' test whether matplotlib is doing the right thing but I can check it runs.
    def test_plot_hrtl(self):
        test_plot = hrtl.plot()
        test_plot = hrtl.plot(log_age=6.8)
        test_plot = hrtl.plot(age_range=(6.2, 6.75), levels=2)
        test_plot = hrtl.plot(log_age=6.8, loc=221, abundances = (1,1,1), cmap='Reds_r')
        del test_plot

    def test_plot_hrtg(self):
        test_plot = hrtg.plot()
        test_plot = hrtg.plot(log_age=6.8)
        test_plot = hrtg.plot(age_range=(6.2, 6.75), levels=2)
        test_plot = hrtg.plot(log_age=6.8, loc=221, abundances = (1,1,1), cmap='Reds_r')
        del test_plot

    def test_plot_hrttg(selfs):
        test_plot = hrttg.plot()
        test_plot = hrttg.plot(log_age=6.8)
        test_plot = hrttg.plot(age_range=(6.2, 6.75), levels=2)
        test_plot = hrttg.plot(log_age=6.8, loc=221, abundances = (1,1,1), cmap='Reds_r')
        del test_plot
