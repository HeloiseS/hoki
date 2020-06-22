"""
Objects and pipelines that compile BPASS data files into more convenient, more pythonic data types
"""

from hoki.constants import BPASS_NUM_METALLICITIES, BPASS_METALLICITIES
from hoki.load import model_output
import pandas as pd
import numpy as np
from hoki.utils.progressbar import print_progress_bar



class SpectraCompiler():
    """
    Pipeline to load the BPASS spectra txt files and save them as a pandas DataFrame.
    """
    def __init__(self, spectra_folder, output_folder, imf, binary=True, verbose=False):
        """
        Input
        -----
        spectra_folder : `str`
            Path to the folder containing the spectra of the given imf.
        output_folder : `str`
            Path to the folder, where to output the pickled pandas.DataFrame


        """
        if verbose: _print_welcome()

        # Set text for population type
        if binary:
            star = "bin"
        else:
            star = "sin"

        # Setup output pandas DataFrame with metallicities and wavelenths
        arrays = [BPASS_NUM_METALLICITIES, np.linspace(1, 100000, 100000)]
        columns = pd.MultiIndex.from_product(arrays, names=["Metallicicty", "Wavelength"])
        print("Allocating memory...", end="")
        spectra = pd.DataFrame(index=np.linspace(6,11, 51), columns=columns, dtype=np.float64)
        spectra.index.name = "log_age"

        # loop over all the metallicities and load all the specta
        for num, metallicity in enumerate(BPASS_METALLICITIES):
            print_progress_bar(num, 12)
            data = model_output(f"{spectra_folder}/spectra-{star}-{imf}.z{metallicity}.dat")
            data = data.loc[:, slice("6.0", "11.0")].T
            spectra.loc[:,(BPASS_NUM_METALLICITIES[num], slice(None))] = data.to_numpy()

        spectra.to_pickle(f"{output_folder}/all_spectra-{star}-{imf}.pkl")
        self.spectra = spectra
        print(f"Spectra file stored in {output_folder} as 'all_spectra-{star}-{imf}.pkl'")
        if verbose: _print_exit_message()

def _print_welcome():
    print('*************************************************')
    print('*******    YOUR DATA IS BEING COMPILED     ******')
    print('*************************************************')
    print("\n\nThis may take a while ;)\nGo get yourself a cup of tea, sit back and relax\nI'm working for you boo!")

    print(
        "\nNOTE: The progress bar doesn't move smoothly - it might accelerate or slow down - it'dc perfectly normal :D")


def _print_exit_message():

    print('\n\n\n*************************************************')
    print('*******     JOB DONE! HAPPY SCIENCING!     ******')
    print('*************************************************')
