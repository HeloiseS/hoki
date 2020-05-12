"""
Module to hold functions and utilities to be applied to spectra,
especially BPASS synthetic spectra
"""

import numpy as np
import pandas as pd
import astropy.io.votable

import pysynphot as psp


def dopcor(df, z, wl_col_index=0):
    """
    Basis doppler correction for hoki's dataframes

    Notes
    -----
    The correction is applied IN PLACE.

    """
    wl_dopcor = (df.iloc[:, wl_col_index].values) - (df.iloc[:, wl_col_index].values * z)
    df.iloc[:, wl_col_index] = wl_dopcor
    return

def load_bandpass(string, verbose=False):
    """Load a bandpass filter.

    Note
    -----
    `load_bandpass()` first checks if the filter is available in the pysynphot
    directory. If so, this filter is loaded.
    If no built-in filter is found, the function searches for a VOTable file
    at the given `string` parameter.

    If multiple throughputs are used, a `ObsModeBandpass` is returned.
    If a single througput is used a `TabularSpectralElement` is returned.
    And if a custom filter is used, a `ArraySpectralElement` is returned.

    Parameters
    ----------
    string : string
        The name of a built-in pysynphot filter or the path to a custom filter
        file.
    verbose : boolean
        Print logging output messages if True. Default=False.

    Returns
    -------
    `pysynphot.spectrum.ArraySpectralElement`
    or `pysynphot.obsbandpass.ObsModeBandpass`
    or `pysynphot.spectrum.TabularSpectralElement`
        A bandpass object
    """

    # Load the filter using pysynphot
    try:
        return psp.ObsBandpass(string)
    except ValueError as e:
        if verbose:
            print(f"{string} not found in build in filters. Searching for a file.")
        try:
            custom = import_custom_filter(string)
            if verbose:
                print("Custom Filter Found!")
            return custom
        except FileNotFoundError:
            raise

def apply_bandpass(spectrum, bandpass):
    """Applies a bandpass filter to a given spectrum.

    Note
    ----
    This function overrides the optmized wavelength binning of pysynphot.
    Instead it uses the wavelength binning of the given spectrum.

    Parameters
    ----------
    spectrum : pysynphot.spectrum.ArraySourceSpectrum
        The spectrum of a source
    bandpass : 'pysynphot.spectrum.TabularSpectralElement'
               or 'pysynphot.spectrum.TabularSpectralElement.ObsModeBandpass'
               or `pysynphot.spectrum.ArraySpectralElement`
        A bandpass to apply to the spectrum

    Returns
    -------
    pysynphot.observation.Observation
        The spectrum with the bandpass applied.
    """
    if not isinstance(spectrum,
                      (psp.spectrum.ArraySourceSpectrum,
                       psp.spectrum.CompositeSourceSpectrum)):
        raise TypeError("Spectrum is not of the correct type!")
    t = type(bandpass)
    if not isinstance(bandpass,
                      (psp.spectrum.TabularSpectralElement,
                       psp.obsbandpass.ObsModeBandpass,
                       psp.spectrum.ArraySpectralElement)):
        raise TypeError("Bandpass is not of the correct type!")

    return psp.Observation(spectrum, bandpass, binset=spectrum.wave)


def bpass_to_psp_spectrum(wavelength, flux):
    """Creats a pysynphot.ArraySpectrum object from a BPASS wavelength and flux.

    Parameters
    ----------
    wavelength : `numpy.array` or `pandas.Series`
        An array or pandas series of wavelength from BPASS
    flux : `numpy.array` or `pandas.Series`
        An array containing the flux at each wavelength in units
        Solar luminosity per Angstrom.

    Returns
    -------
    'pysynphot.spectrum.ArraySourceSpectrum'
        An pysynphot ArraySpectrum with the flux in erg/s/Anstrom.
    """
    # check if values are pandas series
    if type(wavelength) == pd.Series:
        wavelength = wavelength.values
    if type(flux) == pd.Series:
        flux = flux.values
    return psp.ArraySpectrum(wavelength, flux*3.846e33, fluxunits="Flam")


def flux_to_vegamag(bandpass, observation, distance=10):
    """Calculate a vega magnitude in vega system.

    Parameters
    ----------
    bandpass : 'pysynphot.spectrum.TabularSpectralElement'
               or 'pysynphot.spectrum.TabularSpectralElement.ObsModeBandpass'
               or `pysynphot.spectrum.ArraySpectralElement`
        A bandpass applied to the spectrum.
    observation : pysynphot.observation.Observation
        A spectrum with the bandpass applied to it.
    distance : float
        The distance to the object in pc. Default=10

    Returns
    -------
    float
        Vega Magnitude
    """
    if not isinstance(observation, psp.observation.Observation):
        raise TypeError("Spectrum is not of the correct type!")
    if not isinstance(bandpass,
                      (psp.spectrum.TabularSpectralElement,
                       psp.obsbandpass.ObsModeBandpass,
                       psp.spectrum.ArraySpectralElement)):
        raise TypeError("Bandpass is not of the correct type!")

    v = psp.Observation(psp.Vega, bandpass, binset=observation.binwave)
    if (np.diff(observation.binwave) == 1).all(): # check if sum is allowed
        zp = sum(v.binflux)
        luminosity = sum(observation.binflux)
    else:
        zp = np.trapz(v.binflux, v.binwave)
        luminosity = np.trapz(observation.binflux, observation.binwave)
    d = distance*3.0857e18
    flux = luminosity/(4*np.pi*d**2)
    return -2.5 * np.log10(flux/zp)


def import_custom_filter(filename):
    """Gives a filter profile class from a filter profile file containing
    a VOTable.

    Parameters
    ----------
    filename : str
        The filename conatining a VOTable with the filter profile.

    Returns
    -------
    'pysynphot.spectrum.ArraySpectralElement'
        The bandpass filter from the file.
    """
    votable = astropy.io.votable.parse(filename).get_first_table()
    return psp.ArrayBandpass(votable.array["Wavelength"].filled(),
                             votable.array["Transmission"].filled(),
                             name=votable.params[1].value.decode('utf-8')
                             )
