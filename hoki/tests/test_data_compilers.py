"""
Author: Max Briel & Heloise Stevance

Tests for the data_compiler package
"""

import os.path
import pytest
import numpy as np
from unittest.mock import patch

import numpy.testing as npt
import pkg_resources

import hoki.data_compilers as dc
from hoki.data_compilers import ModelDataCompiler, SpectraCompiler, EmissivityCompiler
from hoki.load import model_output
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError

data_path = pkg_resources.resource_filename('hoki', 'data')
data_path+="/"
#models_path=data_path+"sample_stellar_models/"
#print(models_path)

class TestSelectInputFiles(object):
    def test_given_z(self):
        filename = dc._select_input_files(['z014'])[0]
        assert filename[-19:-15] =='z014', "metallicity not in right place"
    def test_different_tail(self):
        filename = dc._select_input_files(['z014'], imf='imf135_100')[0]
        assert filename[-10:] == 'imf135_100', ""


def test_compile_input_files_to_dataframe():
    filename = dc._select_input_files(['z001'], directory=data_path)
    dc._compile_input_files_to_dataframe(filename)


class TestDataCompiler(object):
    def test_bad_input_metalicity(self):
        with pytest.raises(HokiFormatError):
            __ = ModelDataCompiler(z_list=['bla'], columns=['M1'])

    def test_bad_input_columns(self):
        with pytest.raises(HokiFormatError):
            __ = ModelDataCompiler(z_list=['z020'], columns=['bla'])


class TestSpectraCompiler(object):

    # Initialise model_output DataFrame return a smaller single dataframe
    # This reduces I/O readings
    data = model_output(
        f"{data_path}/spectra-bin-imf135_300.z002.dat")

    # Patch the model_output function
    @patch("hoki.data_compilers.np.loadtxt")
    @patch("hoki.data_compilers.isfile")
    def test_compiler(self, mock_isfile, mock_model_output):

        # Set the model_output to the DataFrame
        mock_model_output.return_value = self.data.to_numpy()
        mock_isfile.return_value = True

        spec = SpectraCompiler(f"{data_path}",
                               f"{data_path}",
                               "imf135_300")

        # Check if pkl file is created
        assert os.path.isfile(f"{data_path}/all_spectra-bin-imf135_300.npy")

        # Check output dataframe
        npt.assert_allclose(
            spec.output[3],
            self.data.loc[:, slice("6.0", "11.0")].T.to_numpy(),
            err_msg="Complied spectra is wrong."
        )

        # Remove created pickle
        os.remove(f"{data_path}/all_spectra-bin-imf135_300.npy")


class TestEmissivityCompiler(object):

    # Initialise model_output DataFrame return a smaller single dataframe
    # This reduces I/O readings
    data = model_output(
        f"{data_path}/ionizing-bin-imf135_300.z002.dat")

    # Patch the model_output function
    @patch("hoki.data_compilers.np.loadtxt")
    @patch("hoki.data_compilers.isfile")
    def test_compiler(self, mock_isfile, mock_model_output):

        # Set the model_output to the DataFrame
        mock_model_output.return_value = self.data.to_numpy()
        mock_isfile.return_value = True

        res = EmissivityCompiler(f"{data_path}",
                                 f"{data_path}",
                                 "imf135_300")

        assert os.path.isfile(f"{data_path}/all_ionizing-bin-imf135_300.npy")

        npt.assert_allclose(
            res.output[3],
            self.data.drop(columns='log_age').to_numpy(),
            err_msg="Compiled emissivities is wrong."
        )
        os.remove(f"{data_path}/all_ionizing-bin-imf135_300.npy")



"""
    def test_compiling_small_dataset(self):
        small_set = ModelDataCompiler(z_list=['z020'],
                                 columns=['M1'],
                                 binary=False,
                                 single=True,
                                 input_files_path=data_path,
                                 models_path=data_path)

        assert np.isclose(small_set.data.iloc[0,0], 11, atol=1e-3), "wrong first mass"
        assert int(small_set.data.shape[0])==613, "wrong shape"
"""
