"""
Tests for the data_compiler package
"""

import os.path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pkg_resources

from hoki.constants import BPASS_NUM_METALLICITIES
from hoki.data_compilers import SpectraCompiler
from hoki.load import model_output

data_path = pkg_resources.resource_filename('hoki', 'data')


class TestSpectraCompiler(object):

    # Initialise a smaller pandas dataframe
    columns = pd.MultiIndex.from_product([BPASS_NUM_METALLICITIES, np.linspace(1, 10, 10)],
                                         names=["Metallicicty", "Wavelength"])
    df = pd.DataFrame(index=np.linspace(6, 11, 51),
                      columns=columns, dtype=np.float64)

    # Initialise model_output DataFrame return a smaller single dataframe
    # This reduces I/O readings
    data = model_output(
        f"{data_path}/spectra-bin-imf135_300.z002.dat").loc[:, slice("6.0", "11.0")]

    # Patch the Dataframe creation function
    @patch("hoki.data_compilers.pd.DataFrame")
    # Patch the model_output function
    @patch("hoki.data_compilers.model_output")
    def test_compiler(self, mock_model_output, mock_dataframe):

        # Set the smaller output DataFrame as pandas DataFrame creation output
        mock_dataframe.return_value = self.df

        # Set the model_output to the DataFrame
        mock_model_output.return_value = self.data

        spec = SpectraCompiler(f"{data_path}",
                               f"{data_path}",
                               "imf135_300")

        # Check if pkl file is created
        assert os.path.isfile(f"{data_path}/all_spectra-bin-imf135_300.pkl")
        # Check output dataframe
        npt.assert_allclose(
            spec.spectra.loc[:, (0.002, slice(0, 10))], self.data.T)

        # Remove created pickle
        os.remove(f"{data_path}/all_spectra-bin-imf135_300.pkl")
