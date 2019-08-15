"""
This module implements the tools to easily load BPASS data.
"""

import pandas as pd
#from specutils import Spectrum1D
#import numpy as np
import hoki.hrdiagrams as hr
from hoki.constants import *
import os

# TODO: Should I allow people to chose to load the data into a numpy arrays as well or is the
#       data frame good enough?


def model_output(path, hr_type=None):
    """
    Loads a BPASS output file

    Parameters
    ----------
    path : str
        Path to the file containing the target data.

    hr_type : str, optional
        Type of HR diagram to load: 'TL', 'Tg' or 'TTG'.

    Returns
    -------
    Output Data : pandas.DataFrame or hoki.hrdiagrams.HRDiagrams object

    """

    assert isinstance(path, str), "The location of the file is expected to be a string."
    assert os.path.isfile(path), "This file does not exist, or its path is incorrect."
    assert hr_type in [None,'TL', 'Tg', 'TTG'], "The HR diagram type is invalid. " \
                                                "Available options are: 'TL', 'Tg', 'TTG'. "

    if "supernova" in path:
        return _sn_rates(path)

    elif "numbers" in path:
        return _stellar_numbers(path)

    elif "yields" in path:
        return _yields(path)

    elif "starmass" in path:
        return _stellar_masses(path)

    elif "spectra" in path:
        return _sed(path)

    elif "ioniz" in path:
        return _ionizing_flux(path)

    elif "colour" in path:
        return _colours(path)

    elif "hrs" in path and hr_type == 'TL':
        return _hrTL(path)

    elif "hrs" in path and hr_type == 'Tg':
        return _hrTg(path)

    elif "hrs" in path and hr_type == 'TTG':
        return _hrTTG(path)

    else:
        print("Could not load the Stellar Population output. "
              "Trouble shooting:\n1) Is the filename correct?"
              "\n2) Trying to load an HR diagram? "
              "Make sure hr_type is set! Available options are: 'TL', 'Tg', 'TTG'. ")


def _sn_rates(path):
    """
    Loads One Supernova rate file into a dataframe
    """
    return pd.read_csv(path, sep=r"\s+",
                       names=['log_age', 'Ia', 'IIP', 'II', 'Ib', 'Ic', 'LGRB', 'PISNe', 'low_mass',
                       'e_Ia', 'e_IIP', 'e_II', 'e_Ib', 'e_Ic', 'e_LGRB', 'e_PISNe', 'e_low_mass',
                       'age_yrs'], engine='python')


def _stellar_numbers(path):
    """
    Load One stellar type number file into a dataframe
    """
    return pd.read_csv(path, sep=r"\s*",
                       names=['log_age', 'O_hL', 'Of_hL', 'B_hL', 'A_hL', 'YSG_hL',
                              'K', 'M_hL', 'WNH_hL', 'WN_hL', 'WC_hL',
                              'O_lL', 'Of_lL', 'B_lL', 'A_lL', 'YSG_lL',
                              'K', 'M_lL', 'WNH_lL', 'WN_lL', 'WC_lL',], engine='python')


def _yields(path):
    """
    Load One yields file into a dataframe
    """
    return pd.read_csv(path, sep=r"\s*",
                       names=['log_age', 'H_wind', 'He_wind', 'Z_wind', 'E_wind',
                              'E_sn', 'H_sn', 'He_sn', 'Z_sn'], engine='python')


def _stellar_masses(path):
    """
    Load One stellar masses file into a dataframe
    """
    return pd.read_csv(path, sep=r"\s*",
                       names=['log_age', 'stellar_mass', 'remnant_mass'], engine='python')


def _hrTL(path):
    """
    Load HR diagrams (TL type)
    """
    # 'a' is just a place order which contains the whole file in an array of shape (45900,100)
    a = np.loadtxt(path)
    return hr.HRDiagram(a[0:5100,:].reshape(51,100,100),
                        a[5100:10200,:].reshape(51,100,100),
                        a[10200:15300,:].reshape(51,100,100), hr_type='TL')


def _hrTg(path):
    """
    Load One HR diagrams (Tg type)
    """
    a = np.loadtxt(path)
    return hr.HRDiagram(a[15300:20400,:].reshape(51,100,100),
                        a[20400:25500,:].reshape(51,100,100),
                        a[25500:30600,:].reshape(51,100,100), hr_type='Tg')


def _hrTTG(path):
    """
    Load One HR diagrams (T/TG type)
    """
    a = np.loadtxt(path)
    return hr.HRDiagram(a[30600:35700,:].reshape(51,100,100),
                        a[35700:40800,:].reshape(51,100,100),
                        a[40800:,:].reshape(51,100,100), hr_type='TTG')


def _sed(path):
    """
    Load One SED file
    """
    return pd.read_csv(path, sep=r"\s+", engine='python',
                       names=['WL', '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8',
                              '6.9', '7', '7.1', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '7.8',
                              '7.9', '8.0', '8.1', '8.2', '8.3', '8.4', '8.5', '8.6', '8.7', '8.8',
                              '8.9', '9.0', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '9.7', '9.8',
                              '9.9', '10.0', '10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7',
                              '10.8', '10.9', '11'])


def _ionizing_flux(path):
    """
    Load One ionizing flux file
    """
    return pd.read_csv(path, sep=r'\s+', engine='python',
                       names=['log_age', 'prod_rate', 'halpha', 'FUV', 'NUV'])


def _colours(path):
    """
    Load One colour file
    """
    return pd.read_csv(path, sep=r'\s+', engine='python',
                       names=['log_age', 'V-I', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'u',
                              'g', 'r', 'i', 'z', 'f300w', 'f336w', 'f435w', 'f450w', 'f555w',
                              'f606w', 'f814w', 'prod_rate', 'halpha', 'FUV', 'NUV'])