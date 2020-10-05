"""
Author: Max Briel

Tests for the data_compiler package
"""

import os.path
from unittest.mock import patch

import numpy.testing as npt
import pkg_resources

from hoki.data_compilers import SpectraCompiler, EmissivityCompiler
from hoki.load import model_output

data_path = pkg_resources.resource_filename('hoki', 'data')


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
