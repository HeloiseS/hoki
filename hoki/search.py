from hoki.constants import *
from hoki.load import model_input, dummy_to_dataframe
import pandas as pd
import re
from hoki.utils.progressbar import print_progress_bar
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiKeyError
from hoki.utils.hoki_object import HokiObject

#TODO: Write tests
#TODO: Write docstrings
#TODO: Write tutorial


class DataCompiler(HokiObject):
    def __init__(self, z_list, columns=['V'], single=False, binary=True,
                 models_path=MODELS_PATH, input_files_path=OUTPUTS_PATH, verbose=True):

        assert isinstance(z_list, list), "z_list should be a list of strings"

        # Checking if a keyword is wrong
        wrong_z_keyword = set(z_list) - set(bpass_input_z_list)
        if len(wrong_z_keyword) != 0:
            raise HokiFormatError(
                f"Unknown metallicity keyword(s): {wrong_z_keyword}\n\nDEBBUGGING ASSISTANT: "
                f"Here is a list of valid metallicity keywords\n{bpass_input_z_list}")

        assert isinstance(columns, list), "columns should be a list of strings"
        self.dummy_dict_cols = list(dummy_dicts[DEFAULT_BPASS_VERSION].keys())
        wrong_column_names = set(columns) - set(self.dummy_dict_cols)
        if len(wrong_column_names) != 0:
            raise HokiFormatError(
                f"Unknown column name(s): {wrong_column_names}\n\nDEBBUGGING ASSISTANT: "
                f"Here is a list of valid column names\n{self.dummy_dict_cols}")

        if verbose: _print_welcome()

        self.z_list = z_list
        self.columns = columns
        self.single = single
        self.binary = binary
        input_file_list = select_input_files(self.z_list, directory=input_files_path,
                                             single=self.single, binary=self.binary)
        inputs_dataframe = compile_inputs(input_file_list)
        self.data = compile_model_data(inputs_dataframe, columns=self.columns, source=models_path)

        if verbose: _print_exit_message()

    def __getitem__(self, item):
        return self.data[item]


def compile_model_data(inputs_df, columns, source=MODELS_PATH):
    dataframes = []
    not_found = []

    for i in range(inputs_df.shape[0]):
        print_progress_bar(i, inputs_df.shape[0])
        model_path = inputs_df.iloc[i, 0]
        if len(model_path) > 46:
            try:
                dummy_i = dummy_to_dataframe(source + 'NEWBINMODS/' + model_path)
                dummy_i = dummy_i[columns]
            except FileNotFoundError as e:
                not_found.append(model_path)
                continue

        else:
            try:
                dummy_i = dummy_to_dataframe(source + model_path)
                dummy_i = dummy_i[columns]
            except FileNotFoundError as e:
                not_found.append(model_path)
                continue

        i_input_file = pd.DataFrame(inputs_df.iloc[i, :].values.reshape(-1, len(inputs_df.iloc[i, :].values)),
                                    columns=inputs_df.iloc[i, :].index.tolist())
        inputs_to_add = pd.concat([i_input_file] * dummy_i.shape[0], ignore_index=True)

        dataframes.append(pd.concat([dummy_i, inputs_to_add], axis=1))

    return pd.concat(dataframes).reset_index().drop('index', axis=1)


def compile_inputs(input_file_list):
    """
    Puts together all inputs into one dataframe

    Parameters
    ----------
    input_file_list

    Returns
    -------

    """
    input_dfs = []
    for file in input_file_list:
        input_dfs.append(model_input(file))

    inputs_df = pd.concat(input_dfs)
    inputs_df['z'] = [re.search('-z(.*)-', name).group()[2:-1] for name in inputs_df.filenames]
    return inputs_df.reset_index().drop('index', axis=1)


def select_input_files(z_list, directory=OUTPUTS_PATH,
                       binary=True, single=False, imf='imf135_300'):
    """
    Creates list of relevant input file

    Parameters
    ----------
    z_list
    directory
    binary
    single
    imf

    Returns
    -------

    """
    base = 'input_bpass_'

    input_file_list = []
    if single:
        input_file_list += [directory + base + z + '_sin_' + imf for z in z_list]
    if binary:
        z_list = list(set(z_list) - set(['zem4hmg', 'zem5hmg', 'z001hmg', 'z002hmg', 'z003hmg', 'z004hmg']))
        input_file_list += [directory + base + z + '_bin_' + imf for z in z_list]

    return input_file_list


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


### should be in load??


