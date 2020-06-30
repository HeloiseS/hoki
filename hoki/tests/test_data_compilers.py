"""
Tests for the data_compiler package
"""

import os.path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pkg_resources

from hoki.data_compilers import SpectraCompiler
from hoki.load import model_output

data_path = pkg_resources.resource_filename('hoki', 'data')


class TestSpectraCompiler(object):

    # Initialise model_output DataFrame return a smaller single dataframe
    # This reduces I/O readings
    data = model_output(
        f"{data_path}/spectra/spectra-bin-imf135_300.z002.dat").loc[:, slice("6.0", "11.0")]

    # Patch the model_output function
    @patch("hoki.data_compilers.model_output")
    def test_compiler(self, mock_model_output):

        # Set the model_output to the DataFrame
        mock_model_output.return_value = self.data

        spec = SpectraCompiler(f"{data_path}",
                               f"{data_path}",
                               "imf135_300")

        # Check if pkl file is created
        assert os.path.isfile(f"{data_path}/all_spectra-bin-imf135_300.npy")

        # Check output dataframe
        npt.assert_allclose(
            spec.spectra[3], self.data.T, err_msg="Complied spectra is wrong.")

        # Remove created pickle
        os.remove(f"{data_path}/all_spectra-bin-imf135_300.npy")
