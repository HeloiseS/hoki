"""
Module to hold functions and utilities to be applied to spectra,
especially BPASS synthetic spectra
"""

import numpy as np
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiFormatWarning

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

#TODO: Unittests
def pseudo_continuum(wl, spectrum, lower, upper):
    """
    Traces a linear pseudo-continuum

    Parameters
    ----------
    wl
    spectrum
    lower
    upper

    Returns
    -------

    """
    # Calculating the mean Flux of the lower continuum region
    try:
        cont1 = np.mean(spectrum[(wl > lower[0]) & (wl < lower[1])])
    except TypeError as e:
        raise HokiFormatError(f'{e} DEBUGGING ASSISTANT: Check your input types!')

    # Finding the middle wavelength of the lower continuum region
    wl1 = np.mean(lower)

    # Calculating the mean Flux of the upper continuum region
    try:
        cont2 = np.mean(spectrum[(wl > upper[0]) & (wl < upper[1])])
    except TypeError as e:
        raise HokiFormatError(f'{e} DEBUGGING ASSISTANT: Check your input types!')

    # Finding the middle wavelength of the upper continuum region
    wl2 = np.mean(upper)

    # Calculates the gradient and intercept...
    m = (cont2 - cont1) / (wl2 - wl1)
    c = cont2 - wl2 * m

    # An then calculates the line of the continuum for the whole wavelength range.
    pseudo_cont = m * wl + c

    return pseudo_cont.values


def equivalent_width(wl, spectrum, lower_cont, upper_cont, line_bound=None):
    """
    Measure of the equivalent width

    Parameters
    ----------
    wl
    spectrum
    lower_cont
    upper_cont
    line_bound

    Returns
    -------

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