from hoki import load
import pkg_resources
from unittest.mock import patch
import numpy.testing as npt
import pytest
import pandas as pd
import numpy as np
from hoki.constants import *
from hoki.utils.exceptions import HokiFormatError, HokiKeyError, HokiTypeError


data_path = pkg_resources.resource_filename('hoki', 'data')

sn_file = data_path+'/supernova-bin-imf_chab100.z008.dat'
nmbr_file = data_path+'/numbers-bin-imf_chab100.z001.dat'
yields_file = data_path+'/yields-bin-imf_chab100.z001.dat'
masses_file_sin = data_path+'/starmass-sin-imf_chab100.z006.dat'
masses_file_bin = data_path+'/starmass-bin-imf_chab100.z014.dat'
hr_file = data_path+'/hrs-sin-imf_chab100.zem4.dat'
sed_file = data_path+'/spectra-bin-imf135_300.z002.dat'
ion_file = data_path+'/ionizing-bin-imf135_300.z002.dat'
colour_file = data_path+'/colours-bin-imf135_300.z002.dat'
cmd_path = data_path+'/cmd_bv_z002_bin_imf135_300'
optical_em_lines_path = data_path+'/Optical_data_bin_z006.dat'
UV_em_lines_path = data_path+'/UV_data_bin_z006.dat'


def test_unpickle():
    cmd = load.unpickle(cmd_path)
    assert cmd is not None, 'Unpickle returned None'


def test_model_output():
    data = load.model_output(sn_file)
    data = load.model_output(nmbr_file)
    data = load.model_output(yields_file)
    data = load.model_output(masses_file_bin)
    data = load.model_output(sed_file)
    data = load.model_output(ion_file)
    data = load.model_output(hr_file, hr_type='TL')
    data = load.model_output(hr_file, hr_type='Tg')
    data = load.model_output(hr_file, hr_type='TTG')
    del data


#def test_dummy_to_dataframe():
#    load.dummy_to_dataframe(data_path+"/NEWSINMODS/z020/sneplot-z020-11")


def test_load_sn_rates():
    data = load._sn_rates(sn_file)
    assert data.shape[0] > 0, "The DataFrame is empty"
    assert data.shape[1] == 18, "There should be 18 columns, instead there are "+str(data.shape[1])


def test_load_stellar_numbers():
    data = load._stellar_numbers(nmbr_file)
    assert data.shape[0] > 0, "The DataFrame is empty"
    assert data.shape[1] == 21, "There should be 21 columns, instead there are "+str(data.shape[1])


def test_load_yields():
    data = load._yields(yields_file)
    assert data.shape[0] > 0, "the dataframe is empty"
    assert data.shape[1] == 9, "there should be 9 columns, instead there are "+str(data.shape[1])


def test_load_stellar_masses_sin():
    data = load._stellar_masses(masses_file_sin)
    assert data.shape[0] > 0, "the dataframe is empty"
    assert data.shape[1] == 3, "there should be 3 columns, instead there are "+str(data.shape[1])


def test_load_stellar_masses_bin():
    data = load._stellar_masses(masses_file_bin)
    assert data.shape[0] > 0, "the dataframe is empty"
    assert data.shape[1] == 3, "there should be 3 columns, instead there are "+str(data.shape[1])


def test_load_hrTL():
    data = load._hrTL(hr_file)
    assert data.high_H.shape == (51, 100, 100), "Attribute high_H has the wrong shape"
    assert data.medium_H.shape == (51, 100, 100), "Attribute medium_H has the wrong shape"
    assert data.low_H.shape == (51, 100, 100), "Attribute low_H has the wrong shape"


def test_load_hrTg():
    data = load._hrTg(hr_file)
    assert data.high_H.shape == (51, 100, 100), "Attribute high_H has the wrong shape"
    assert data.medium_H.shape == (51, 100, 100), "Attribute medium_H has the wrong shape"
    assert data.low_H.shape == (51, 100, 100), "Attribute low_H has the wrong shape"


def test_load_hrTTG():
    data = load._hrTTG(hr_file)
    assert data.high_H.shape == (51, 100, 100), "Attribute high_H has the wrong shape"
    assert data.medium_H.shape == (51, 100, 100), "Attribute medium_H has the wrong shape"
    assert data.low_H.shape == (51, 100, 100), "Attribute low_H has the wrong shape"


def test_sed():
    data = load._sed(sed_file)
    assert data.shape[0] > 0, "the dataframe is empty"
    assert data.shape[1] == 52, "there should be 52 columns, instead there are "+str(data.shape[1])


def test_ion():
    data = load._ionizing_flux(ion_file)
    assert data.shape[0] > 0, "the dataframe is empty"
    assert data.shape[1] == 5, "there should be 5 columns, instead there are "+str(data.shape[1])


def test_colours():
    data = load._colours(colour_file)
    assert data.shape[0] > 0, "the dataframe is empty"
    assert data.shape[1] == 26, "there should be 26 columns, instead there are "+str(data.shape[1])


def test_nebular_emission_lines():
    data = load.nebular_lines(optical_em_lines_path)
    assert data.shape == (3087,24), "The DataFrame doesn't have the right shape something happened"
    data = load.nebular_lines(UV_em_lines_path)
    assert data.shape == (3087, 26)

#############################
#  Test BPASS File Loading  #
#############################


class TestLoadAllRates(object):

    # Setup files to load
    data = load.model_output(
        f"{data_path}/supernova-bin-imf135_300.zem5.dat")

    # Check if function loads rates
    @patch("hoki.load.model_output")
    def test_load_rates(self, mock_model_output):
        mock_model_output.return_value = self.data
        x = load.rates_all_z(f"{data_path}", "imf135_300"),\
            "The rates cannot be initialised."

    # Load rates
    with patch("hoki.load.model_output") as mock_model_output:
        mock_model_output.return_value = data
        x = load.rates_all_z(f"{data_path}", "imf135_300")

    # Test wrong inputs
    def test_file_not_present(self):
        with pytest.raises(AssertionError):
            _ = load.rates_all_z(f"{data_path}", "imf135_300"),\
                "The file is not present, but the load function runs."

    def test_wrong_imf(self):
        with pytest.raises(HokiKeyError):
            _ = load.rates_all_z(f"{data_path}", "i"),\
                "An unsupported IMF is taken as an input."

    # Test output
    def test_output_shape(self):
        assert type(self.x) == pd.DataFrame
        assert (self.x.columns.get_level_values(0).unique() ==
                np.array(BPASS_EVENT_TYPES)).all(),\
            "wrong headers read from the file."
        assert (self.x.columns.get_level_values(1).unique() ==
                np.array(BPASS_NUM_METALLICITIES)).all(),\
            "wrong metallicity header"

    def test_output(self):
        assert np.isclose(self.x.loc[:, ("Ia", 0.00001)],
                          self.data["Ia"]).all(),\
            "Models are not loaded correctly."


class TestLoadAllSpectra(object):

    # Initialise model_output DataFrame
    # This reduces I/O readings
    data = load.model_output(
        f"{data_path}/spectra-bin-imf135_300.z002.dat")

    # Patch the model_output function
    @patch("hoki.data_compilers.np.loadtxt")
    @patch("hoki.data_compilers.isfile")
    def test_compile_spectra(self, mock_isfile, mock_model_output):

        # Set the model_output to the DataFrame
        mock_model_output.return_value = self.data.to_numpy()
        mock_isfile.return_value = True
        spec = load.spectra_all_z(f"{data_path}", "imf135_300")

        # Check if compiled file is created
        assert os.path.isfile(f"{data_path}/all_spectra-bin-imf135_300.npy"),\
            "No compiled file is created."

        # Check output numpy array
        npt.assert_allclose(
            spec[3],
            self.data.loc[:, slice("6.0", "11.0")].T.to_numpy(),
            err_msg="Loading of files has failed."
        )

    def test_load_pickled_file(self):

        spec = load.spectra_all_z(f"{data_path}", "imf135_300")

        # Check output numpy array
        npt.assert_allclose(
            spec[3],
            self.data.loc[:, slice("6.0", "11.0")].T.to_numpy(),
            err_msg="Loading of compiled file has failed."
        )

        os.remove(f"{data_path}/all_spectra-bin-imf135_300.npy")


class TestLoadAllEmissivities(object):

    # Initialise model_output DataFrame
    # This reduces I/O readings
    data = load.model_output(
        f"{data_path}/ionizing-bin-imf135_300.z002.dat")

    # Patch the model_output function
    @patch("hoki.data_compilers.np.loadtxt")
    @patch("hoki.data_compilers.isfile")
    def test_compile_emissivities(self, mock_isfile, mock_model_output):

        # Set the model_output to the DataFrame
        mock_model_output.return_value = self.data.to_numpy()
        mock_isfile.return_value = True
        res = load.emissivities_all_z(f"{data_path}", "imf135_300")

        # Check if compiled file is created
        assert os.path.isfile(f"{data_path}/all_ionizing-bin-imf135_300.npy"),\
            "No compiled file is created."

        # Check output numpy array
        npt.assert_allclose(
            res[3],
            self.data.drop(columns='log_age').to_numpy(),
            err_msg="Loading of files has failed."
        )

    def test_load_pickled_file(self):

        res = load.emissivities_all_z(f"{data_path}", "imf135_300")

        # Check output numpy array
        npt.assert_allclose(
            res[3],
            self.data.drop(columns='log_age').to_numpy(),
            err_msg="Loading of compiled file has failed."
        )

        os.remove(f"{data_path}/all_ionizing-bin-imf135_300.npy")
