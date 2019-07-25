import pandas as pd

# TODO: Should I allow people to chose to load the data into a numpy arrays as well or is the
# data frame good enough?


def sn_rates(path):
    """
    Loads One Supernova rate file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                       names=['age_log', 'Ia', 'IIP', 'II', 'Ib', 'Ic', 'LGRB', 'PISNe', 'low_mass',
                       'e_Ia', 'e_IIP', 'e_II', 'e_Ib', 'e_Ic', 'e_LGRB', 'e_PISNe', 'e_low_mass',
                       'age_yrs'], engine='python')
    return data


def stellar_numbers(path):
    """
    Load One stellar type number file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                       names=['age_log', 'O_hL', 'Of_hL', 'B_hL', 'A_hL', 'YSG_hL',
                              'K', 'M_hL', 'WNH_hL', 'WN_hL', 'WC_hL',
                              'O_lL', 'Of_lL', 'B_lL', 'A_lL', 'YSG_lL',
                              'K', 'M_lL', 'WNH_lL', 'WN_lL', 'WC_lL',], engine='python')
    return data


def yields(path):
    """
    Load One yields file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                       names=['age_log', 'H_wind', 'He_wind', 'Z_wind', 'E_wind',
                              'E_sn', 'H_sn', 'He_sn', 'Z_sn'], engine='python')
    return data

