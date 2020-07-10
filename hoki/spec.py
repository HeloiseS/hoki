"""
Module to hold functions and utilities to be applied to spectra,
especially BPASS synthetic spectra
"""


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

