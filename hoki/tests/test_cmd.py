from hoki.cmd import CMD
import pkg_resources

data_path = pkg_resources.resource_filename('hoki', 'data')


class TestCMD(object):
    cmd = CMD(data_path+'/input_bpass_z020_bin_imf135_300', path = data_path+'/sample_stellar_models/')

    def test_init_(self ):
        assert self.cmd is not None, "object not instanciated"
        assert sum(self.cmd.grid.flatten()) == 0, "CMD Grid should be empty"

    def test_make(self):
        self.cmd.make(filter1='B', filter2='V')
        #assert sum(self.cmd.grid.flatten()) == 0, 'CMD grid is not empty'
        assert sum(self.cmd.grid.flatten()) != 0, 'CMD grid is still empty'

    def test_plot(self):
        myplot = self.cmd.plot(log_age=7)
