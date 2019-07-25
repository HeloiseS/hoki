import pandas as pd

# TODO: Should I allow people to chose to load the data into a numpy arrays as well or is the
# data frame good enough?


def sn_models(path):
    """
    Loads One Supernova rate file into a dataframe
    """
    data = pd.read_csv(path, sep=r"\s*",
                names=['age_log', 'ia', 'iip', 'ii', 'ib', 'ic', 'lgrb', 'pisne', 'low_mass',
                       'e_ia', 'e_iip', 'e_ii', 'e_ib', 'e_ic', 'e_lgrb', 'e_pisne', 'e_low_mass',
                       'age_yrs'], engine='python')
    return data


