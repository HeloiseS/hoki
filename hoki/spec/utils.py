"""
Module to hold functions and utilities to be applied to spectra,
especially BPASS synthetic spectra
"""

import numpy as np
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiFormatWarning
import pandas as pd

def dopcor(df, z, wl_col_index=0):
    """
    Basis doppler correction for hoki'dc dataframes

    Notes
    -----
    The correction is applied IN PLACE.

    """
    wl_dopcor = (df.iloc[:, wl_col_index].values) - (df.iloc[:, wl_col_index].values * z)
    df.iloc[:, wl_col_index] = wl_dopcor
    return


def pseudo_continuum(wl, spectrum, lower_cont, upper_cont):
    """
    Traces a linear pseudo-continuum

    Notes
    -----
    This function takes two ranges of wavelength (typically below and above a line for which the EW will be calculated),
    and calculates a linear pseudo continuum between their respective middle point.

    Parameters
    ----------
    wl: 1D array
        Wavelength
    spectrum: 1D array
        Flux spectrum
    lower_cont: tuple, list or array of length==2
        Lower range of wavelength to probe the continuum
    upper_cont: tuple, list or array of length==2
        Upper range of wavelength to probe the continuum

    Returns
    -------
    pseudo continuum flux values in a 1D np.ndarray

    """
    # Calculating the mean Flux of the lower continuum region
    try:
        cont1 = np.mean(spectrum[(wl > lower_cont[0]) & (wl < lower_cont[1])])
    except TypeError as e:
        raise HokiFormatError(f'{e} DEBUGGING ASSISTANT: Check your input types!')

    # Finding the middle wavelength of the lower continuum region
    wl1 = np.mean(lower_cont)

    # Calculating the mean Flux of the upper continuum region
    try:
        cont2 = np.mean(spectrum[(wl > upper_cont[0]) & (wl < upper_cont[1])])
    except TypeError as e:
        raise HokiFormatError(f'{e} DEBUGGING ASSISTANT: Check your input types!')

    # Finding the middle wavelength of the upper continuum region
    wl2 = np.mean(upper_cont)

    # Calculates the gradient and intercept...
    m = (cont2 - cont1) / (wl2 - wl1)
    c = cont2 - wl2 * m

    # An then calculates the line of the continuum for the whole wavelength range.
    pseudo_cont = m * wl + c
    if isinstance(pseudo_cont, pd.Series): return pseudo_cont.values
    return pseudo_cont


def equivalent_width(wl, spectrum, lower_cont, upper_cont, line_bound=None):
    """
    Measure of the equivalent width

    Parameters
    ----------
    wl: 1D array
        Wavelength
    spectrum: 1D array
        Flux spectrum
    lower_cont: tuple, list or array of length==2
        Lower range of wavelength to probe the continuum
    upper_cont: tuple, list or array of length==2
        Upper range of wavelength to probe the continuum
    line_bound: tuple, list or array of length==2, optional
        The range within which to calculate the equivalent width. This is optional and if no range is given the
        equivalent width will be calculated in the between the lower and upper continuum ranges.

    Returns
    -------
    Equivalent width in a 1D np.ndarray

    """

    # Calculates the pseudo continuum
    cont = pseudo_continuum(wl, spectrum, upper_cont, lower_cont)

    # Masking areas outside of the line we care about
    if line_bound is None:
        mask = (wl > lower_cont[1]) & (wl < upper_cont[0])
    else:
        assert isinstance(line_bound, list), "not list"
        mask = (wl > line_bound[0]) & (wl < line_bound[1])

    return np.sum(1 - spectrum[mask] / cont[mask])