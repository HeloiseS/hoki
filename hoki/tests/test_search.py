from hoki.search import DataCompiler, bpass_input_z_list
import hoki.search as s
import pytest
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError
import numpy as np
import pkg_resources

data_path = pkg_resources.resource_filename('hoki', 'data')
data_path+="/"
#models_path=data_path+"sample_stellar_models/"
#print(models_path)


class TestSelectInputFiles(object):
    def test_given_z(self):
        filename = s.select_input_files(['z014'])[0]
        assert filename[-19:-15] =='z014', "metallicity not in right place"
    def test_different_tail(self):
        filename = s.select_input_files(['z014'], imf='imf135_100')[0]
        assert filename[-10:] == 'imf135_100', ""


def test_compile_input_files_to_dataframe():
    filename = s.select_input_files(['z001'], directory=data_path)
    s.compile_input_files_to_dataframe(filename)


class TestDataCompiler(object):
    def test_bad_input_metalicity(self):
        with pytest.raises(HokiFormatError):
            __ = DataCompiler(z_list=['bla'], columns=['M1'])

    def test_bad_input_columns(self):
        with pytest.raises(HokiFormatError):
            __ = DataCompiler(z_list=['z020'], columns=['bla'])

"""
    def test_compiling_small_dataset(self):
        small_set = DataCompiler(z_list=['z020'],
                                 columns=['M1'],
                                 binary=False,
                                 single=True,
                                 input_files_path=data_path,
                                 models_path=data_path)

        assert np.isclose(small_set.data.iloc[0,0], 11, atol=1e-3), "wrong first mass"
        assert int(small_set.data.shape[0])==613, "wrong shape"
"""

