import pandas as pd
from hoki import spec, load
import numpy as np
import os
import pkg_resources
import pysynphot as psp
import pytest

data_path = pkg_resources.resource_filename('hoki', 'data')

#wavelengths and throughput/transmission manually read from the LICK.LICK.U file
wl = [3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100]
transmission = [0.015, 0.140,0.347,0.440,0.625,0.685,0.708,0.643,0.458,0.170,0.029]
data = load.model_output(f"{data_path}/spectra-bin-imf135_300.z002.dat")

def test_dopcor():
    ngc4993 = pd.read_csv(data_path+'/ngc4993_spec_tot.dat')
    wl0_init = ngc4993.wl[0]
    spec.dopcor(ngc4993,  0.009783)
    assert np.isclose((wl0_init-ngc4993.wl[0]), 46.4655, atol=1e-05), "The doppler correction isn't right"


class TestImportCustomFilter():

    def test_input_file_name(self):
        # Wrong file name
        with pytest.raises(FileNotFoundError):
            spec.import_custom_filter(f" ")

    bp = spec.import_custom_filter(f"{data_path}/LICK.LICK.U.xml")

    def test_type(self):
        assert type(self.bp) == psp.spectrum.ArraySpectralElement

    def test_name(self):
        assert self.bp.name == "LICK/LICK.U",\
               "The filter name is incorrectly imported"

    def test_wavelength(self):
        assert np.isclose(self.bp.wave,wl).all(),\
               "The wavelength bins are incorrectly imported"

    def test_throughput(self):
        assert np.isclose(self.bp.throughput,transmission).all(),\
                "The throughput is incorrectly imported"


class TestLoadBandpass():

    # test that environment variable for pysynphot has been set
    def test_PYSYN_CDBS_set(self):
        assert os.environ['PYSYN_CDBS'] != "", "'PYSYN_CDBS' not set"

    # Input-output type checks
    def test_multi_builtin_filter(self):
        bp = spec.load_bandpass("acs,hrc,f220w")
        assert type(bp) == psp.obsbandpass.ObsModeBandpass,\
               "Wrong output type for multi pysynphot filter"

    def test_single_builtin_filter(self):
        bp = spec.load_bandpass("johnson,v")
        assert type(bp) == psp.spectrum.TabularSpectralElement,\
               "Wrong output type for single pysynphot filter"

    def test_custom_filter(self):
        bp = spec.load_bandpass(f"{data_path}/LICK.LICK.U.xml")
        assert type(bp) == psp.spectrum.ArraySpectralElement,\
               "Wrong output type for custom filter"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            spec.load_bandpass("f")

    # Tests for checking correct output
    bp1 = spec.load_bandpass("johnson,v")
    bp2 = psp.ObsBandpass("johnson,v")

    def test_wave_builtin(self):
        assert np.isclose(self.bp1.wave, self.bp2.wave).all(),\
               "The bins are incorrectly created using `load_bandpass`"

    def test_througput_builtin(self):
        assert np.isclose(self.bp1.throughput, self.bp2.throughput).all(),\
               "The throughput is incorrect using `load_bandpass`"

    # Tests to check for correct custom filter input
    bp1 = spec.load_bandpass(f"{data_path}/LICK.LICK.U.xml")
    bp2 = spec.import_custom_filter(f"{data_path}/LICK.LICK.U.xml")

    def test_wave_custom(self):
        assert np.isclose(self.bp1.wave, self.bp2.wave).all(),\
               "The wavelength bins are incorrectly imported using `load_bandpass`"

    def test_throughput_custom(self):
        assert np.isclose(self.bp1.throughput, self.bp2.throughput).all(),\
               "The throughput bins are incorrectly imported using `load_bandpass`"


class TestBPASStoPSPSpectrum():

    def test_input_type(self):
        with pytest.raises(TypeError):
            spec.bpass_to_psp_spectrum(1,[10,10])

    def test_input_pandas(self):
        sp = spec.bpass_to_psp_spectrum(data.WL,data["6.0"])
        assert type(sp) == psp.spectrum.ArraySourceSpectrum,\
            "A panda.Series is not recognised as a valid input"

    def test_input_numpy(self):
        sp = spec.bpass_to_psp_spectrum(data.WL.values, data["6.0"])
        assert type(sp) == psp.spectrum.ArraySourceSpectrum,\
            "A numpy.array is not recognised as a valid input"

    sp = spec.bpass_to_psp_spectrum(data.WL.values, data["6.0"])

    def test_wave_output(self):
        assert np.isclose(self.sp.wave, data.WL.values).all(),\
            "The output wavelength bins are incorrect"

    def test_flux_output(self):
        assert np.isclose(self.sp.flux, data["6.0"].values*3.846e33).all(),\
            "The output flux is incorrecly transformed"

    def test_units_output(self):
        assert type(self.sp.fluxunits) == psp.units.Flam,\
            "The fluxunits are of the wrong type"


class TestApplyBandpass():

    sp = spec.bpass_to_psp_spectrum(data.WL.values, data["6.0"])
    bp = spec.load_bandpass("johnson,v")

    def test_input_type(self):
        with pytest.raises(TypeError):
            spec.apply_bandpass(np.arange(0,10,1), 1)
        with pytest.raises(TypeError):
            spec.apply_bandpass(self.bp, self.sp)

    obs = spec.apply_bandpass(sp, bp)

    def test_output_type(self):
        assert type(self.obs) == psp.observation.Observation,\
            "The apply bandpass output is of the wrong type"

    def test_wavebins(self):
        assert np.isclose(self.obs.binwave, data.WL.values).all(),\
            "The final wavelength bins do not line up with the original binning"


class TestFlux2VegaMag():

    def test_input_type(self):
        with pytest.raises(TypeError):
            spec.flux_to_vegamag([10,1], np.arange(0,10,1))

    def test_zeropoint(self):
        d2vega = 7.767 # pc (from doi:10.1088/0004-6256/136/1/452)
        vega = psp.Vega*4*np.pi*(d2vega*3.0857e18)**2
        bp = spec.load_bandpass("johnson,v")
        obs = spec.apply_bandpass(vega, bp)
        assert np.isclose(spec.flux_to_vegamag(bp, obs, d2vega),0),\
               "Vega normalisation is not applied correctly"
