"""
This module implements the tools to easily load BPASS data.
"""

import pandas as pd
import hoki.hrdiagrams as hr
from hoki.constants import *
import os
import yaml
import io
import pickle
import pkg_resources
import hoki.data_compilers
import warnings
from hoki.utils.exceptions import HokiDeprecationWarning, HokiKeyError

# TODO: Should I allow people to chose to load the data into a numpy arrays as well or is the
#       data frame good enough?

__all__ = ['model_input', 'model_output', 'set_models_path', 'unpickle']

data_path = pkg_resources.resource_filename('hoki', 'data')


########################
# GENERAL LOAD HELPERS #
########################


def unpickle(path):
    """Extract pickle files"""
    assert os.path.isfile(path), 'File not found.'
    return pickle.load(open(path, 'rb'))


# TODO: Deprecation warning
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
    deprecation_msg = "set_models_path has been moved to the hoki.constants module -- In future versions of hoki" \
                      "calling set_models_path from hoki.load will fail"

    warnings.warn(deprecation_msg, HokiDeprecationWarning)

    assert os.path.isdir(path), 'HOKI ERROR: The path provided does not correspond to a valid directory'

    path_to_settings = data_path+'/settings.yaml'
    with open(path_to_settings, 'r') as stream:
        settings = yaml.safe_load(stream)

    settings['models_path'] = path
    with io.open(path_to_settings, 'w', encoding='utf8') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    print('Looks like everything went well! You can check the path was correctly updated by looking at this file:'
          '\n'+path_to_settings)


########################
#  LOAD DUMMY VARIABLE #
########################


def dummy_to_dataframe(filename, bpass_version=DEFAULT_BPASS_VERSION):
    """Reads in dummy to df from a filename"""
    inv_dict ={v: k for k, v in dummy_dicts[bpass_version].items()} #inverted bpass dummy dictionary
    cols = [inv_dict[key] if key in inv_dict.keys() else 'Nan'+str(key) for key in range(96)]
    dummy = pd.read_csv(filename, names=cols, sep=r"\s+", engine='python')
    return dummy

#########################
# MODEL INPUT FUNCTIONS #
#########################

def model_input(path):
    """
    Loads inputs from one file and put them in a dataframe

    Parameters
    ----------
    path : str
        Path to the file containing the input data.

    Returns
    -------

    """

    assert isinstance(path, str), "The location of the file is expected to be a string."
    assert os.path.isfile(path), f"File {path} does not exist, or its path is incorrect."

    lines = open(path).read().split("\n")

    # rows [a,b,c,d] in the BPASS manual
    row = [1, 0, 0, 0]

    # All potential input parameters and filename
    # If there is no corresponding value for a particular model, will append a NaN
    filenames = []
    modelimfs = []
    modeltypes = []
    mixedimf = []
    mixedage = []
    initialBH = []
    initialP = []

    # This goes through the file line by line. Using .split is not possible/convenient
    # The vector will tell use what we should do with this line.
    for l in lines[1:]:

        # This line contains the filename.
        if row[0]:
            filenames.append(l)
            # The next line will contain the imf probability and the type - we reset the vector..
            row = [0, 1, 0, 0]
            # ... and we skip the rest to read in the next line.
            continue

        # This line contains the imf probability and the type
        elif row[1]:
            elements = l.split() # we split the line into the 2 values
            modelimfs.append(elements[0]) # and append them
            modeltypes.append(elements[1])

            # The next step is decided according to the value of type
            # To know what each value means, consult the BPASS manual
            if int(elements[1]) < 2:
                # In this case all the other potential inputs are NaNs and we go back to the
                # beginning to read a new file name
                row = [1, 0, 0, 0]
                mixedimf.append(0.0)
                mixedage.append(0.0)
                initialBH.append(np.nan)
                initialP.append(np.nan)
                continue

            elif int(elements[1]) != 4:
                # If type is 2 or 3, we know the initial BH and initial P will be NaN
                # but we also need to read the next line so we set the vector accordingly.
                row = [0, 0, 1, 0]
                initialBH.append(np.nan)
                initialP.append(np.nan)
                continue

            elif int(elements[1]) == 4:
                # If the type is 4 we need all the other outputs, so we need the next 2 lines.
                # We set the vector accordingly.
                row = [0, 0, 1, 1]
                continue

        elif row[2]:
            # Splitting the two values and putting them in their respective lists
            elements = l.split()
            mixedimf.append(elements[0])
            mixedage.append(elements[1])

            # Then depending on whether we need the next line for more inputs
            # we set the vector to either go back to reading a filename or to probe those inputs.
            if not row[3]:
                row = [1, 0, 0, 0]
            if row[3]:
                row[2] = 0

            continue

        elif row[3]:
            # This reads the last possible pair of inputs and puts them in their rightful lists.
            elements = l.split()
            initialBH.append(elements[0])
            initialP.append(elements[0])
            # We then reset the vector to be reading in filenames because no more inputs are coming
            row = [1, 0, 0, 0]

    # Once we've goe through the whole file and filled our lists, we can put them in a dataframe
    # with some named columns and set the datatypes to strings and numbers.
    input_df = pd.DataFrame.from_dict({'filenames': filenames[:-1], 
                                       'model_imf': modelimfs,
                                       'types': modeltypes, 
                                       'mixed_imf': mixedimf,
                                       'mixed_age': mixedage, 
                                       'initial_BH': initialBH,
                                       'initial_P': initialP}).astype({'filenames': str,
                                                                       'model_imf': float,
                                                                       'types': int,
                                                                       'mixed_imf': float,
                                                                       'mixed_age': float,
                                                                       'initial_BH': float,
                                                                       'initial_P': float})
    return input_df


##########################
# MODEL OUTPUT FUNCTIONS #
##########################

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

    assert isinstance(path, str), "HOKI ERROR: The location of the file is expected to be a string."
    assert os.path.isfile(path), "HOKI ERROR: This file does not exist, or its path is incorrect."
    assert hr_type in [None,'TL', 'Tg', 'TTG'], "HOKI ERROR: The HR diagram type is invalid. " \
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
        print("HOKI ERROR -- Could not load the Stellar Population output. "
              "\nDEBUGGING ASSISTANT:\n1) Is the filename correct?"
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
    return pd.read_csv(path, sep=r"\s+",
                       names=['log_age', 'O_hL', 'Of_hL', 'B_hL', 'A_hL', 'YSG_hL',
                              'K_hL', 'M_hL', 'WNH_hL', 'WN_hL', 'WC_hL',
                              'O_lL', 'Of_lL', 'B_lL', 'A_lL', 'YSG_lL',
                              'K_lL', 'M_lL', 'WNH_lL', 'WN_lL', 'WC_lL'], engine='python')


def _yields(path):
    """
    Load One yields file into a dataframe
    """
    return pd.read_csv(path, sep=r"\s+",
                       names=['log_age', 'H_wind', 'He_wind', 'Z_wind', 'E_wind',
                              'E_sn', 'H_sn', 'He_sn', 'Z_sn'], engine='python')


def _stellar_masses(path):
    """
    Load One stellar masses file into a dataframe
    """
    return pd.read_csv(path, sep=r"\s+",
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
    df = pd.read_csv(path, sep=r"\s+", engine='python', header=None)
    if df.columns.shape[0] == 26:
        # this to take into account the case where I'm using the files with half time res
        cols = ['WL', '6.0', '6.2', '6.4', '6.6', '6.8', '7.0', '7.2', '7.4', '7.6',
                '7.8', '8.0', '8.2', '8.4', '8.6', '8.8', '9.0', '9.2', '9.4', '9.6',
                '9.8', '10.0', '10.2', '10.4', '10.6', '10.8']
        
    if df.columns.shape[0] == 52:
        cols = ['WL', '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8',
                '6.9', '7.0', '7.1', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '7.8',
                '7.9', '8.0', '8.1', '8.2', '8.3', '8.4', '8.5', '8.6', '8.7', '8.8',
                '8.9', '9.0', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '9.7', '9.8',
                '9.9', '10.0', '10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7',
                '10.8', '10.9', '11.0']
    df.columns = cols
    return df


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


##########################
# NEBULAR EMISSION LINES #
##########################


def nebular_lines(path):
    """
    Load the nebular line output information
    Parameters
    ----------
    path

    Returns
    -------

    """
    assert isinstance(path, str), "HOKI ERROR: The location of the file is expected to be a string."
    assert os.path.isfile(path), "HOKI: ERROR This file does not exist, or its path is incorrect."

    if 'UV' in path:
        return _UV_nebular_lines(path)
    elif 'Optical' in path:
        return _optical_nebular_lines(path)


def _optical_nebular_lines(path):
    column_opt_em_lines=['model_num', 'logU', 'log_nH', 'log_age',
                         'NII6548_F', 'NII6548_EW', 'NII6584_F', 'NII6584_EW',
                         'SiII6716_F', 'SiII6716_EW', 'SiII6731_F', 'SiII6731_EW',
                         'OI6300_F', 'OI6300_EW',
                         'OIII4959_F','OIII4959_EW','OIII5007_F','OIII5007_EW',
                         'Halpha_F', 'Halpha_EW', 'Hbeta_F', 'Hbeta_EW',
                         'HeI4686_F', 'HeI4686_EW']

    return pd.read_csv(path, skiprows=1, sep=r'\s+', engine='python', names=column_opt_em_lines)


def _UV_nebular_lines(path):
    column_UV_em_lines = ['model_num', 'logU', 'log_nH', 'log_age',
                          'HeII1640_F', 'HeII1640_EW',
                          'CIII1907_F', 'CIII1907_EW', 'CIII1910_F', 'CIII1910_EW',
                          'CIV1548_F', 'CIV1548_EW', 'CIV1551_F', 'CIV1551_EW',
                          'OI1357_F', 'OI1357_EW',
                          'OIII1661_F', 'OIII1661_EW', 'OIII1666_F', 'OIII1666_EW',
                          'SiII1263_F', 'SiII1263_EW', 'SiIII1308_F', 'SiIII1308_EW', 'SiII1531_F', 'SiII1531_EW']

    return pd.read_csv(path, skiprows=1, sep=r'\s+', engine='python', names=column_UV_em_lines)



#####################################
#  BPASS Load over all metallicity  #
#####################################

def rates_all_z(data_path, imf, binary=True):
    """
    Loads the BPASS supernova event files.

    Notes
    -----
    The rates are just read from file and not normalised.

    Input
    -----
    data_path : `str`
        The filepath to the folder containing the BPASS data

    binary : `bool`
        Use the binary files or just the single stars. Default=True

    imf : `str`
        BPASS Identifier of the IMF to be used.
        The accepted IMF identifiers are:
        - `"imf_chab100"`
        - `"imf_chab300"`
        - `"imf100_100"`
        - `"imf100_300"`
        - `"imf135_100"`
        - `"imf135_300"`
        - `"imfall_300"`
        - `"imf170_100"`
        - `"imf170_300"`

    Returns
    -------
    `pandas.DataFrame` (51, (8,13)) (log_age, (event_types, metallicity)
        A pandas MultiIndex dataframe containing the BPASS number of events
        per metallicity per type.
        Usage:   rates.loc[log_age, (type, metallicity)]
        Example: rates.loc[6.5, ("Ia", 0.02)]


        Notes
        -----
        This dataframe has the following structure.
        The index is the log_age as a float.
        The column is a `pandas.MultiIndex` with the event types
        (level=0, `float`) and the metallicity (level=1, `float`)

        |Event Type | Ia      | IIP      |  ... | PISNe | low_mass |
        |Metallicity| 0.00001 | 0.00001  |  ... |  0.04 |    0.04  |
        | log_age   |---------|----------|------|-------|----------|
        |    6.0    |
        |    ...    |                  Event Rate values
        |    11.0   |

    """

    # Check population type
    star = "bin" if binary else "sin"

    # Check if the given IMF is in the accepted IMFs
    if imf not in BPASS_IMFS:
        raise HokiKeyError(
            f"{imf} is not a BPASS IMF. Please select a correct IMF.\n"\
            "These can be found in the documentation of this function.")

    # Create the output DataFrame
    arrays = [BPASS_NUM_METALLICITIES, BPASS_EVENT_TYPES]
    columns = pd.MultiIndex.from_product(
        arrays, names=["Metallicicty", "Event Type"])

    rates = pd.DataFrame(index=BPASS_TIME_BINS,
                         columns=columns,
                         dtype=np.float64)
    rates.index.name = "log_age"

    # load supernova count files
    for num, metallicity in enumerate(BPASS_METALLICITIES):
        data = model_output(
            f"{data_path}/supernova-{star}-{imf}.{metallicity}.dat"
        )
        data = data.loc[:, slice(BPASS_EVENT_TYPES[0], BPASS_EVENT_TYPES[-1])]

        rates.loc[:, (BPASS_NUM_METALLICITIES[num], slice(None))] = data.to_numpy()

    # swap metallicity and event type
    return rates.swaplevel(0, 1, axis=1)


def spectra_all_z(data_path, imf, binary=True):
    """
    Load all BPASS spectra from files.

    Notes
    -----
    The first time this function is ran on a folder it will generate a pickle
    file containing all the BPASS spectra per metallicity for faster loading
    in the future. It stores the file in the same folder with the name:
    `all_spectra-[bin/sin]-[imf].pkl`

    The spectra are just read from file and not normalised.


    Input
    -----
    data_path : `str`
        The path to the folder containing the BPASS spectra.

    binary : `bool`
        Use the binary files or just the single stars. Default=True

    imf : `str`
        BPASS Identifier of the IMF to be used.
        The accepted IMF identifiers are:
        - `"imf_chab100"`
        - `"imf_chab300"`
        - `"imf100_100"`
        - `"imf100_300"`
        - `"imf135_100"`
        - `"imf135_300"`
        - `"imfall_300"`
        - `"imf170_100"`
        - `"imf170_300"`


    Returns
    -------
    spectra : `numpy.ndarray` (13, 51, 100000) [(metallicity, log_age, wavelength)]
        A 3D numpy array containing all the BPASS spectra for a specific imf
        and binary or single star population.
        Usage: spectra[1][2][1000]
                (gives L_\\odot for Z=0.0001 and log_age=6.2 at 999 Angstrom)

    """
    # Check population type
    star = "bin" if binary else "sin"

    # check IMF key
    if imf not in BPASS_IMFS:
        raise HokiKeyError(
            f"{imf} is not a BPASS IMF. Please select a correct IMF.")

    # check if data_path is a string
    if not isinstance(data_path, str):
         raise HokiTypeError("The folder location is expected to be a string.")

    # check if compiled file exists
    if os.path.isfile(f"{data_path}/all_spectra-{star}-{imf}.npy"):
        print("Loading precompiled file.")
        spectra = np.load(f"{data_path}/all_spectra-{star}-{imf}.npy")
        print("Done Loading.")
    # Otherwise compile
    else:
        print("Compiled file not found. Data will be compiled")
        spec = hoki.data_compilers.SpectraCompiler(
            data_path, data_path, imf, binary=binary
        )
        spectra = spec.output
    return spectra


def emissivities_all_z(data_path, imf, binary=True):
    """
    Load all BPASS emissivities from files.

    Notes
    -----
    The first time this function is run on a folder it will generate an npy
    file containing all the BPASS emissivities for faster loading in the
    future. It stores the file in the same folder with the name:
    `all_ionizing-[bin/sin]-[imf].npy`.

    The emissivities are just read from file and not normalised.


    Input
    -----
    data_path : `str`
        The path to the folder containing the BPASS files.

    binary : `bool`
        Use the binary files or just the single stars. Default=True

    imf : `str`
        BPASS Identifier of the IMF to be used.
        The accepted IMF identifiers are:
        - `"imf_chab100"`
        - `"imf_chab300"`
        - `"imf100_100"`
        - `"imf100_300"`
        - `"imf135_100"`
        - `"imf135_300"`
        - `"imfall_300"`
        - `"imf170_100"`
        - `"imf170_300"`

    Returns
    -------
    emissivities : `numpy.ndarray` (N_Z, N_age, 4) [(metallicity, log_age, band)]
        A 3D numpy array containing all the BPASS emissivities (Nion [1/s],
        L_Halpha [ergs/s], L_FUV [ergs/s/A], L_NUV [ergs/s/A]) for a specific
        imf and binary or single star population.
        Usage: spectra[1][2][0]
                (gives Nion for Z=0.0001 and log_age=6.2)
    """
    # Check population type
    star = "bin" if binary else "sin"

    # check IMF key
    if imf not in BPASS_IMFS:
        raise HokiKeyError(
            f"{imf} is not a BPASS IMF. Please select a correct IMF.")

    # check if data_path is a string
    if not isinstance(data_path, str):
         raise HokiTypeError("The folder location is expected to be a string.")

    # Check if compiled spectra are already present in data folder
    if os.path.isfile(f"{data_path}/all_ionizing-{star}-{imf}.npy"):
        print("Load precompiled file.")
        emissivities = np.load(f"{data_path}/all_ionizing-{star}-{imf}.npy")
        print("Done Loading.")

    # Compile the spectra for faster reading next time otherwise
    else:
        print("Compiled file not found. Data will be compiled.")
        res = hoki.data_compilers.EmissivityCompiler(
            data_path, data_path, imf, binary=binary
        )
        emissivities = res.output
    return emissivities


#################
#               #
#################

def _do_not_use():
    import webbrowser
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    webbrowser.open_new_tab(url)
