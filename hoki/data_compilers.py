"""
Author: Max Briel & Heloise Stevance

Objects and pipelines that compile BPASS data files into more convenient, more pythonic data types
"""

import numpy as np

from hoki.constants import (DEFAULT_BPASS_VERSION, MODELS_PATH,
                            OUTPUTS_PATH, dummy_dicts,
                            BPASS_IMFS, BPASS_METALLICITIES)
import hoki.load
import pandas as pd
import re
from hoki.utils.progressbar import print_progress_bar
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiKeyError
from hoki.utils.hoki_object import HokiObject


bpass_input_z_list = ["z"+i for i in BPASS_METALLICITIES]

class ModelDataCompiler(HokiObject):
    """
    Given a list of metalicities, a list of valid BPASS model attributes (in the dummy array), chosen types of model
    (binary,s ingle or both) and correct paths, will compile the corresponding BPASS stellar models into a DataFrame
    """
    def __init__(self, z_list, columns=['V'], single=False, binary=True,
                 models_path=MODELS_PATH, input_files_path=OUTPUTS_PATH,
                 verbose=True, bpass_version=DEFAULT_BPASS_VERSION):
        """

        Parameters
        ----------
        z_list: list
            List of metallicities to include.
        columns: list
            Columns to include from the dummy array
        single: bool, optional
            Whether to include the BPASS models that only contain single star models. Default is False.
        binary: bool, optional
            Whether to include BPASS models that contain binary systems. Default is True.
        models_path: str, optional
            Location of the BPASS models (the 50GB folder). Default=MODELS_PATH
        input_files_path: str, optional
            Location of the input files (which are BPASS model outputs - I know. Focus.).
            Default=hopki.constants.OUTPUTS_PATH
        verbose: bool, optional
            Whether to print the welcome and goodbye messages. Default=True
        bpass_version: str, optional
            BPASS version. Default=hoki.constants.DEFAULT_BPASS_VERSION
        """

        assert isinstance(z_list, list), "z_list should be a list of strings"

        ####### CHECKING INPUTS ######

        # Metalicity
        wrong_z_keyword = set(z_list) - set(bpass_input_z_list)
        if len(wrong_z_keyword) != 0:
            raise HokiFormatError(
                f"Unknown metallicity keyword(dc): {wrong_z_keyword}\n\nDEBBUGGING ASSISTANT: "
                f"Here is a list of valid metallicity keywords\n{bpass_input_z_list}")

        # Columns
        assert isinstance(columns, list), "columns should be a list of strings"
        self.dummy_dict_cols = list(dummy_dicts[bpass_version].keys())
        wrong_column_names = set(columns) - set(self.dummy_dict_cols)
        if len(wrong_column_names) != 0:
            raise HokiFormatError(
                f"Unknown column name(dc): {wrong_column_names}\n\nDEBBUGGING ASSISTANT: "
                f"Here is a list of valid column names\n{self.dummy_dict_cols}")

        # Saying hi to the user and giving them advice
        if verbose: _print_welcome()

        # Basic attributes
        self.z_list = z_list
        self.columns = columns
        self.single = single
        self.binary = binary

        # Creating the list of input file names...
        self.input_file_list = _select_input_files(self.z_list, directory=input_files_path,
                                                   single=self.single, binary=self.binary)

        # ...then turning them into dataframes ...
        self.inputs_dataframe = _compile_input_files_to_dataframe(self.input_file_list)

        # ... and finally compiling the model data corresponding to the contents of
        # our inputs dataframe.
        self.data, self.not_found = _compile_model_data(self.inputs_dataframe, columns=self.columns,
                                                        models_path=models_path, bpass_version=bpass_version)

        # Telling the user everything went well with the compilation
        if verbose: _print_exit_message()

    def __getitem__(self, item):
        return self.data[item]


def _compile_model_data(inputs_df, columns, models_path=MODELS_PATH, bpass_version=DEFAULT_BPASS_VERSION):
    """
    Compile dataframe of models contained in the input_dataframe provided

    Parameters
    ----------
    inputs_df: pandas.DataFrame from compile_input_files_to_dataframe
        DataFrame containing the selected inputs we care about
    columns: list
        Columns to include from the dummy array
    models_path:
        Location of the BPASS models (the 50GB folder). Default=MODELS_PATH

    Returns
    -------
    pandas.DataFrame containing the columns requested for the models mentioned in the given input dataframe
    """
    dataframes = []
    not_found = []

    for i in range(inputs_df.shape[0]):
        print_progress_bar(i, inputs_df.shape[0])
        model_path = inputs_df.iloc[i, 0]
        if len(model_path) > 46:
            try:
                dummy_i = hoki.load.dummy_to_dataframe(models_path + model_path, bpass_version)
                dummy_i = dummy_i[columns]
            except FileNotFoundError as e:
                not_found.append(model_path)
                continue

        else:
            try:
                dummy_i = hoki.load.dummy_to_dataframe(models_path + model_path, bpass_version)
                dummy_i = dummy_i[columns]
            except FileNotFoundError as e:
                not_found.append(model_path)
                continue

        i_input_file = pd.DataFrame(inputs_df.iloc[i, :].values.reshape(-1, len(inputs_df.iloc[i, :].values)),
                                    columns=inputs_df.iloc[i, :].index.tolist())
        inputs_to_add = pd.concat([i_input_file] * dummy_i.shape[0], ignore_index=True)

        dataframes.append(pd.concat([dummy_i, inputs_to_add], axis=1))

    return pd.concat(dataframes).reset_index().drop('index', axis=1), not_found


def _compile_input_files_to_dataframe(input_file_list):
    """
    Puts together all inputs into one dataframe

    Parameters
    ----------
    input_file_list: list
        List of input files of interest

    Returns
    -------
    pandas.DataFrame containing information in the input files given
    """
    input_dfs = []
    for file in input_file_list:
        input_dfs.append(hoki.load.model_input(file))

    inputs_df = pd.concat(input_dfs)
    inputs_df['z'] = [re.search('-z(.*?)-', name).group()[2:-1] for name in inputs_df.filenames]
    return inputs_df.reset_index().drop('index', axis=1)


def _select_input_files(z_list, directory=OUTPUTS_PATH,
                        binary=True, single=False, imf='imf135_300'):
    """
    Creates list of relevant input file

    Parameters
    ----------
    z_list: list
        List of metallicities to include.
    directory: str, optional
        Location of the input files
    single: bool, optional
        Whether to include the BPASS models that only contain single star models. Default is False.
    binary: bool, optional
        Whether to include BPASS models that contain binary systems. Default is True.
    imf: str, optional
        IMF specification at the end of the filename. Default is 'imf135_300'. Consult the BPASS manual to check
        the format and valid IMF specifications (or just look at the last 10 characters of the input
        filenames that you care about and just use that).

    Returns
    -------
    List of filenames
    """
    base = 'input_bpass_'

    input_file_list = []
    if single:
        input_file_list += [directory + base + z + '_sin_' + imf for z in z_list]
    if binary:
        z_list = list(set(z_list) - set(['zem4hmg', 'zem5hmg', 'z001hmg', 'z002hmg', 'z003hmg', 'z004hmg']))
        input_file_list += [directory + base + z + '_bin_' + imf for z in z_list]

    return input_file_list

class SpectraCompiler():
    """
    Pipeline to load the BPASS spectra txt files and save them as a 3D
    `numpy.ndarray` binary file.

    Attributes
    ----------
    spectra : `numpy.ndarray` (13, 51, 100000) [(metallicity, log_age, wavelength)]
        A 3D numpy array containing all the BPASS spectra for a specific imf
        and binary or single star population.
        Usage: spectra[1][2][1000]
                (gives L_\\odot for Z=0.0001 and log_age=6.2 at 999 Angstrom)
    """

    def __init__(self, spectra_folder, output_folder, imf, binary=True, verbose=False):
        """
        Input
        -----
        spectra_folder : `str`
            Path to the folder containing the spectra of the given imf.

        output_folder : `str`
            Path to the folder, where to output the pickled pandas.DataFrame

        imf : `str`
            BPASS IMF Identifiers
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

        binary : `bool`
            If `True`, loads the binary files. Otherwise, just loads single stars.
            Default=True

        verbose : `bool`
            If `True` prints out extra information for the user.
            Default=False
        """
        if verbose:
            _print_welcome()

        # Check population type
        star = "bin" if binary else "sin"

        # check IMF key
        if imf not in BPASS_IMFS:
            raise HokiKeyError(
                f"{imf} is not a BPASS IMF. Please select a correct IMF.")

        # Setup the numpy output
        spectra = np.empty((13, 51, 100000), dtype=np.float64)

        # loop over all the metallicities and load all the spectra
        for num, metallicity in enumerate(BPASS_METALLICITIES):
            print_progress_bar(num, 12)
            spectra[num] = np.loadtxt(f"{spectra_folder}/spectra-{star}-{imf}.z{metallicity}.dat").T[1:, :]

        # pickle the datafile
        np.save(f"{output_folder}/all_spectra-{star}-{imf}", spectra)
        self.spectra = spectra
        print(
            f"Spectra file stored in {output_folder} as 'all_spectra-{star}-{imf}.npy'")
        if verbose:
            _print_exit_message()



def _print_welcome():
    print('*************************************************')
    print('*******    YOUR DATA IS BEING COMPILED     ******')
    print('*************************************************')
    print("\n\nThis may take a while ;)\nGo get yourself a cup of tea, sit back and relax\nI'm working for you boo!")

    print(
      "\nNOTE: The progress bar doesn't move smoothly - it might accelerate or slow down - it's perfectly normal :D")


def _print_exit_message():

    print('\n\n\n*************************************************')
    print('*******     JOB DONE! HAPPY SCIENCING!     ******')
    print('*************************************************')
