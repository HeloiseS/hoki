import numpy as np
import pandas as pd

#TODO: write the test!
def dopcor(df, z):
    """
    Basis doppler correction for hoki's dataframes
    """
    wl_dopcor = (df.WL.values) - (df.WL.values * z)
    df.WL = wl_dopcor
    return df
