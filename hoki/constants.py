"""
Just BPASS things
"""
import numpy as np
import yaml
import pkg_resources
import os
import io

#TODO: update documentation and add mentions of set_models path and set_default_bpass_verison in the constants
# TODO: put the constants in a dataclass!!
# module - it will change things in the CMD jupyter notebook I think.


data_path = pkg_resources.resource_filename('hoki', 'data')
path_to_settings = os.path.join(data_path, 'settings.yaml')

with open(os.path.relpath(path_to_settings), 'rb') as stream:
    settings = yaml.safe_load(stream)

MODELS_PATH = settings['models_path']
OUTPUTS_PATH = settings['outputs_path'] # This constant for dev purposes.
BPASS_TIME_BINS = np.arange(6.0, 11.1, 0.1)
BPASS_TIME_INTERVALS = np.array([10**(t+0.05) - 10**(t-0.05) for t in BPASS_TIME_BINS])
BPASS_TIME_WEIGHT_GRID = np.array([np.zeros((100, 100)) + dt for dt in BPASS_TIME_INTERVALS])

BPASS_LINEAR_TIME_EDGES = np.append([0.0], 10**np.arange(6.05, 11.15, 0.1))
BPASS_LINEAR_TIME_INTERVALS = np.diff(BPASS_LINEAR_TIME_EDGES)

BPASS_METALLICITIES = ["zem5", "zem4", "z001","z002", "z003", "z004", "z006", "z008", "z010", "z014", "z020", "z030", "z040"]
BPASS_NUM_METALLICITIES = np.array([0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010,
                                    0.014, 0.020, 0.030, 0.040])
BPASS_METALLICITY_MID_POINTS = (BPASS_NUM_METALLICITIES[1:] + BPASS_NUM_METALLICITIES[:-1])/2

BPASS_EVENT_TYPES = ["Ia", "IIP", "II", "Ib", "Ic", "LGRB", "PISNe", "low_mass"]
HOKI_NOW = 13.799e9

BPASS_IMFS = ["imf_chab100", "imf_chab300", "imf100_100", "imf100_300",
              "imf135_100", "imf135_300", "imf135all_100", "imf170_100",
              "imf170_300"]

# wavelengths at which spectra are given [angstrom]
BPASS_WAVELENGTHS = np.arange(1, 100001)

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

dummy_manual = {'timestep': 'Time Step Number',
                'age': 'Age in years',
                'log(R1)': 'Log10(R/Rsun) of the primary',
                'log(T1)': 'Effective log10 temperature of the primary',
                'log(L1)': 'Log10(L/Lsun) of the primary',
                'M1': 'M/Msun of the primary',
                'He_core1': 'He core mass of the primary /Msun',
                'CO_core1': 'CO core mass of the primary /Msun',
                'ONe_core1': 'ONe core mass of the primary /Msun',
                'X': 'Surface mass fractions for X (hydrogen)',
                'Y': 'Surface mass fractions for Y (helium)',
                'C': 'Surface mass fractions for C (carbon)',
                'N': 'Surface mass fractions for N (nitrogem)',
                'O': 'Surface mass fractions for O (oxygen)',
                'Ne': 'Surface mass fractions for Ne (neon)',
                'MH1': 'Mass of Hydrogen in primary /Msun',
                'MHe1': 'Mass of Helium in primary /Msun',
                'MC1': 'Mass of Carbon in primary /Msun',
                'MN1': 'Mass of Nitrogen in primary /Msun',
                'MO1': 'Mass of Oxygen in primary /Msun',
                'MNe1': 'Mass of Neon in primary /Msun',
                'MMg1': 'Mass of Hydrogen in primary /Msun',
                'MSi1': 'Mass of Silicon in primary /Msun',
                'MFe1': 'Mass of Iron in primary /Msun',
                'envelope_binding_E': 'Binding Energy of the envelope in Joules',
                'star_binding_E': 'Total star binding energy in Joules',
                'Mrem_weakSN': 'Remnant Mass /Msun for a weakSN (1e43 J)',
                'Mej_weakSN': 'Ejecta Mass /Msun for a weakSN (1e43 J)',
                'Mrem_SN': 'Remnant Mass /Msun for a normal SN (1e44 J)',
                'Mej_SN': 'Ejecta Mass /Msun for a normal SN (1e44 J)',
                'Mrem_superSN': 'Remnant Mass /Msun for a high energy SN (1e45 J)',
                'Mej_superSN': 'Ejecta Mass /Msun for a high energy SN (1e45 J)',
                'AM_bin': 'Angular momentum of the binary',
                'P_bin': 'Period of the binary in years',
                'log(a)': 'Log10 of the binary separation/Rsun ',
                'M2': 'Mass of the secondary /Msun',
                'MTOT': 'Total Mass of the binary /Msun',
                'DM1W': 'Wind mass loss rate of the primary (Msun/(1.989*s))',
                'DM2W': 'Wind mass loss rate of the secondary (Msun/(1.989*s))',
                'DM1A': 'Accretion onto primary (Msun/(1.989*s))',
                'DM2A': 'Accretion onto secondary (Msun/(1.989*s))',
                'DM1R': 'Roche Lobe overflow of the primary (Msun/(1.989*s))',
                'DM2R': 'Roche Lobe overflow of the secondary (Msun/(1.989*s))',
                'DAM': 'Angular momentum change',
                'log(R2)': 'Log10 of estimated radius of the secondary /Rsun',
                'log(T2)': 'Log10 of estimated effective Temperature of the secondary',
                'log(L2)': 'Log10 of estimated Luminosity of the secondary /Lsun',
                '?': 'Roche Lobe overflux of star 2',
                'modelimf': 'Total IMF probability of stars',
                'mixedimf': 'IMF probability of rejuvinated stars',
                'V-I': 'V-I (absolute mag both stars)', 'U': 'U (absolute mag both stars)',
                'B': 'B (absolute mag both stars)', 'V': 'V (absolute mag both stars)',
                'R': 'R (absolute mag both stars)', 'I': 'I (absolute mag both stars)',
                'J': 'J (absolute mag both stars)', 'H': 'H (absolute mag both stars)',
                'K': 'K (absolute mag both stars)', 'u': 'u (absolute mag both stars)',
                'g': 'g (absolute mag both stars)', 'r': 'r (absolute mag both stars)',
                'i': 'i (absolute mag both stars)', 'z': 'z (absolute mag both stars)',
                'f300w': ' f300w (absolute mag both stars)', 'f336w': 'f336w (absolute mag both stars)',
                'f435w': 'f435w (absolute mag both stars)', 'f450w': 'f450w (absolute mag both stars)',
                'f555w': 'f555w (absolute mag both stars)', 'f606w': 'f606w (absolute mag both stars)',
                'f814w': 'f814w (absolute mag both stars)',
                'U2': 'U (absolute mag secondary star)', 'B2': 'B (absolute mag secondary star)',
                'V2': 'V (absolute mag secondary star)', 'R2': 'R (absolute mag secondary star)',
                'I2': 'I (absolute mag secondary star)', 'J2': 'J (absolute mag secondary star)',
                'H2': 'H (absolute mag secondary star)', 'K2': 'K (absolute mag secondary star)',
                'u2': 'u (absolute mag secondary star)', 'g2': 'g (absolute mag secondary star)',
                'r2': 'r (absolute mag secondary star)', 'i2': 'i (absolute mag secondary star)',
                'z2': 'z (absolute mag secondary star)', 'f300w2': 'f300w (absolute mag secondary star)',
                'f336w2': 'f336w (absolute mag secondary star)', 'f435w2': 'f435w (absolute mag secondary star)',
                'f450w2': 'f450w (absolute mag secondary star)', 'f555w2': 'f555w (absolute mag secondary star)',
                'f606w2': 'f606w (absolute mag secondary star)', 'f814w2': 'f814w (absolute mag secondary star)',
                'Halpha': 'H alpha flux (log L erg/s)', 'FUV': 'UV flux (log L erg/s/A)',
                'NUV': 'Near UV flux (log L erg/s/A)'}

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

    path_to_settings = os.path.join(data_path, 'settings.yaml')
    
    with open(path_to_settings, 'r') as stream:
        settings = yaml.safe_load(stream)

    settings['models_path'] = path
    with io.open(path_to_settings, 'w', encoding='utf8') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    print('Looks like everything went well! You can check the path was correctly updated by looking at this file:'
          '\n'+path_to_settings)


def set_outputs_path(path):
    """
    Changes the defaullt path to the BPASS outputs in hoki's settings

    Parameters
    ----------
    path : str,
        Absolute path to the folder containing the BPASS outputs.

    Notes
    -----
    You are going to have to reload hoki for your new path to take effect.

    """
    assert os.path.isdir(path), 'HOKI ERROR: The path provided does not correspond to a valid directory'

    path_to_settings = os.path.join(data_path, 'settings.yaml')
    
    with open(path_to_settings, 'r') as stream:
        settings = yaml.safe_load(stream)

    settings['outputs_path'] = path
    with io.open(path_to_settings, 'w', encoding='utf8') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    print('Looks like everything went well! You can check the path was correctly updated by looking at this file:'
          '\n'+path_to_settings)


def set_default_bpass_version(version):
    """
    Changes the path to the stellar models in hoki'dc settings

    Parameters
    ----------
    version : str, v221 or v222
        Version of BPASS to consider by default.

    Notes
    -----
    You are going to have to reload hoki for your new default version to be taken into account.

    """
    assert version in ['v221', 'v222'], 'HOKI ERROR: Invalid Version - your options are v221 or v222'

    path_to_settings = os.path.join(data_path, 'settings.yaml')
    with open(path_to_settings, 'r') as stream:
        settings = yaml.safe_load(stream)

    settings['default_bpass_version'] = version
    with io.open(path_to_settings, 'w', encoding='utf8') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    print('Looks like everything went well! You can check the path was correctly updated by looking at this file:'
          '\n'+path_to_settings)
