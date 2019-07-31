"""
This module implements the tools to easily load BPASS data.
"""

import pandas as pd
import numpy as np
import hoki.hrdiagrams as hr


# TODO: Should I allow people to chose to load the data into a numpy arrays as well or is the
#       data frame good enough?

def population_output(path, hr_type=None):

    assert isinstance(path, str), "The location of the file is expected to be a string."
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

    elif "hrs" in path and hr_type == 'TL':
        return _hrTL(path)

    elif "hrs" in path and hr_type == 'Tg':
        return _hrTg(path)

    elif "hrs" in path and hr_type == 'TTG':
        return _hrTTG(path)

    else:
        print("Could not load the Stellar Population output. "
              "Trouble shooting:\n1) Is the filename correct?"
              "\n2) Trying to load an HR diagram? Make sure hr_type is set!")


def _sn_rates(path):
    """
    Loads One Supernova rate file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                       names=['age_log', 'Ia', 'IIP', 'II', 'Ib', 'Ic', 'LGRB', 'PISNe', 'low_mass',
                       'e_Ia', 'e_IIP', 'e_II', 'e_Ib', 'e_Ic', 'e_LGRB', 'e_PISNe', 'e_low_mass',
                       'age_yrs'], engine='python')
    return data


def _stellar_numbers(path):
    """
    Load One stellar type number file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                       names=['age_log', 'O_hL', 'Of_hL', 'B_hL', 'A_hL', 'YSG_hL',
                              'K', 'M_hL', 'WNH_hL', 'WN_hL', 'WC_hL',
                              'O_lL', 'Of_lL', 'B_lL', 'A_lL', 'YSG_lL',
                              'K', 'M_lL', 'WNH_lL', 'WN_lL', 'WC_lL',], engine='python')
    return data


def _yields(path):
    """
    Load One yields file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                       names=['age_log', 'H_wind', 'He_wind', 'Z_wind', 'E_wind',
                              'E_sn', 'H_sn', 'He_sn', 'Z_sn'], engine='python')
    return data


def _stellar_masses(path):
    """
    Load One stellar masses file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                       names=['age_log', 'stellar_mass', 'remnant_mass'], engine='python')
    return data


def _hrTL(path):
    """
    Load HR diagrams (TL type)
    """
    # 'a' is just a place order which contains the whole file in an array of shape (45900,100)
    a = np.loadtxt(path)
    hrTL_object = hr.HRDiagram(a[0:5100,:].reshape(51,100,100),
                                a[5100:10200,:].reshape(51,100,100),
                                a[10200:15300,:].reshape(51,100,100), hr_type='TL')

    return hrTL_object


def _hrTg(path):
    """
    Load One HR diagrams (Tg type)
    """
    a = np.loadtxt(path)
    hrTg_object = hr.HRDiagram(a[15300:20400,:].reshape(51,100,100),
                        a[20400:25500,:].reshape(51,100,100),
                        a[25500:30600,:].reshape(51,100,100), hr_type='Tg')

    return hrTg_object


def _hrTTG(path):
    """
    Load One HR diagrams (T/TG type)
    """
    a = np.loadtxt(path)
    hrTTG_object = hr.HRDiagram(a[30600:35700,:].reshape(51,100,100),
                        a[35700:40800,:].reshape(51,100,100),
                        a[40800:,:].reshape(51,100,100), hr_type='TTG')

    return hrTTG_object
