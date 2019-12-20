from hoki.cmd import CMD
from hoki.load import unpickle
import pkg_resources
import os

data_path = pkg_resources.resource_filename('hoki', 'data')
cmd_path = data_path+'/cmd_bv_z002_bin_imf135_300'


class TestCMD(object):
    cmd = CMD(data_path+'/input_bpass_z020_bin_imf135_300')

    def test_init_(self):
        #print(os.listdir(data_path))
        assert self.cmd is not None, "object not instanciated"
        assert sum(self.cmd.grid.flatten()) == 0, "CMD Grid should be empty"
    """
    # NOTE: These tests are to do locally in dev - they require 50GB of full model data to test
    # they cannot easily be implemented into Travis CI.
    def test_make(self):
        print(os.listdir(data_path))
        self.cmd.make(filter1='B', filter2='V')
        assert sum(self.cmd.grid.flatten()) != 0, 'CMD grid is still empty'

    def test_plot(self):
        assert sum(self.cmd.grid.flatten()) != 0, 'CMD grid is still empty'
        self.cmd.plot()

    def test_plot_bin0(self):
        self.cmd.plot(log_age=6.0)
    """

    def test_plot_pickle(self):
        cmd = unpickle(cmd_path)
        myplot = cmd.plot(log_age=7)
        myplot.set_xlabel('bla')
        del myplot
        del cmd
