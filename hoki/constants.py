"""
Just BPASS things
"""
import numpy as np
import yaml
import pkg_resources
import os
import io

#TODO: update documentation and add mentions of set_models path and set_default_bpass_verison in the constants

# module - it will change things in the CMD jupyter notebook I think.


data_path = pkg_resources.resource_filename('hoki', 'data')

path_to_settings = pkg_resources.resource_filename('hoki', 'data/settings.yaml')

with open(os.path.relpath(path_to_settings), 'rb') as stream:
    settings = yaml.safe_load(stream)

MODELS_PATH = settings['models_path']
OUTPUTS_PATH = settings['outputs_path'] # This constant for dev purposes.
BPASS_TIME_BINS = np.arange(6.0, 11.1, 0.1)
BPASS_TIME_INTERVALS = np.array([10**(t+0.05) - 10**(t-0.05) for t in BPASS_TIME_BINS])
BPASS_TIME_WEIGHT_GRID = np.array([np.zeros((100,100)) + dt for dt in BPASS_TIME_INTERVALS])

HOKI_NOW = 13.799e9

# Create a deprecation warning when using dummy_dict
dummy_dict = {'timestep': 0, 'age': 1, 'log(R1)': 2, 'log(T1)': 3, 'log(L1)': 4, 'M1': 5, 'He_core1': 6, 'CO_core1': 7,
              'ONe_core1': 8, 'X': 10, 'Y': 11, 'C': 12, 'N': 13, 'O': 14, 'Ne': 15, 'MH1': 16, 'MHe1': 17, 'MC1': 18,
              'MN1': 19, 'MO1': 20, 'MNe1': 21, 'MMg1': 22, 'MSi1': 23, 'MFe1': 24, 'envelope_binding_E': 25,
              'star_binding_E': 26, 'Mrem_weakSN': 27, 'Mej_weakSN': 28, 'Mrem_SN': 29, 'Mej_SN': 30,
              'Mrem_superSN': 31, 'Mej_superSN': 32, 'AM_bin': 33,
              'P_bin': 34, 'log(a)': 35, 'M2': 37, 'MTOT': 38, 'DM1W': 39, 'DM2W': 40, 'DM1A': 41, 'DM2A': 42,
              'DM1R': 43, 'DM2R': 44, 'DAM': 45, 'log(R2)': 46, 'log(T2)': 47, 'log(L2)': 48, '?': 49, 'modelimf': 50,
              'mixedimf': 51, 'V-I': 52, 'U': 53, 'B': 54, 'V': 55, 'R': 56, 'I': 57, 'J': 58, 'H': 59, 'K': 60,
              'u': 61, 'g': 62, 'r': 63, 'i': 64, 'z': 65, 'f300w': 66, 'f336w': 67, 'f435w': 68, 'f450w': 69,
              'f555w': 70, 'f606w': 71, 'f814w': 72, 'U2': 73, 'B2': 74, 'V2': 75, 'R2': 76, 'I2': 77, 'J2': 78,
              'H2': 79, 'K2': 80, 'u2': 81, 'g2': 82, 'r2': 83, 'i2': 84, 'z2': 85, 'f300w2': 86, 'f336w2': 87,
              'f435w2': 88, 'f450w2': 89, 'f555w2': 90, 'f606w2': 91, 'f814w2': 92, 'Halpha': 93, 'FUV': 94, 'NUV': 95}

DEFAULT_BPASS_VERSION = settings['default_bpass_version']

dummy_dicts = settings['dummy_dicts']


def set_models_path(path):
    """
    Changes the path to the stellar models in hoki's settings

    Parameters
    ----------
    path : str,
        Absolute path to the top level of the stellar models this could be a directory named something like
        bpass-v2.2-newmodels and the next level down should contain 'NEWBINMODS' and 'NEWSINMODS'.


    Notes
    -----
    You are going to have to reload hoki for your new path to take effect.

    """
    assert os.path.isdir(path), 'HOKI ERROR: The path provided does not correspond to a valid directory'

    path_to_settings = data_path+'/settings.yaml'
    with open(path_to_settings, 'r') as stream:
        settings = yaml.safe_load(stream)

    settings['models_path'] = path
    with io.open(path_to_settings, 'w', encoding='utf8') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    print('Looks like everything went well! You can check the path was correctly updated by looking at this file:'
          '\n'+path_to_settings)


def set_default_bpass_version(version):
    """
    Changes the path to the stellar models in hoki's settings

    Parameters
    ----------
    version : str, v221 or v222
        Version of BPASS to consider by default.

    Notes
    -----
    You are going to have to reload hoki for your new default version to be taken into account.

    """
    assert version in ['v221', 'v222'], 'HOKI ERROR: Invalid Version - your options are v221 or v222'

    path_to_settings = data_path+'/settings.yaml'
    with open(path_to_settings, 'r') as stream:
        settings = yaml.safe_load(stream)

    settings['default_bpass_version'] = version
    with io.open(path_to_settings, 'w', encoding='utf8') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    print('Looks like everything went well! You can check the path was correctly updated by looking at this file:'
          '\n'+path_to_settings)

