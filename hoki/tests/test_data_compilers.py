"""
Tests for the data_compiler package
"""

import os
import shutil
import pkg_resources
import pandas as pd
import numpy as np
import numpy.testing as npt
from hoki.data_compilers import SpectraCompiler
from hoki.load import model_output
from hoki.constants import BPASS_METALLICITIES, BPASS_NUM_METALLICITIES
from unittest.mock import Mock, patch

data_path = pkg_resources.resource_filename('hoki', 'data')


class TestSpectraCompiler(object):
    columns = pd.MultiIndex.from_product([BPASS_NUM_METALLICITIES, np.linspace(1, 10, 10)],
                                         names=["Metallicicty", "Wavelength"])
    df = pd.DataFrame(index=np.linspace(6,11, 51), columns=columns, dtype=np.float64)
    data = model_output(f"{data_path}/spectra-bin-imf135_300.z002.dat").loc[:, slice("6.0", "11.0")]
    def model_output_slice(self, value):
        return self.data

    @patch("hoki.data_compilers.pd.DataFrame")
    @patch("hoki.data_compilers.model_output")
    def test_compiler(self, mock_data_compilers, mock_pd):
        print(self.data)
        mock_pd.return_value = self.df
        mock_data_compilers.side_effect = self.model_output_slice

        spec = SpectraCompiler(f"{data_path}",
                               f"{data_path}",
                               "imf135_300")
        npt.assert_allclose(spec.spectra.loc[:,(0.002,slice(0,10))], self.data.T)
