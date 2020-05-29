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
BPASS_LIN_TIME_EDGES = np.append([0.0], [10**(t+0.05) for t in BPASS_TIME_BINS])
BPASS_TIME_INTERVALS = np.diff(BPASS_LIN_TIME_EDGES)
BPASS_TIME_WEIGHT_GRID = np.array([np.zeros((100,100)) + dt for dt in BPASS_TIME_INTERVALS])

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
