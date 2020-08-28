"""
Author: Max Briel

Objects and pipelines that compile BPASS data files into more convenient, more pythonic data types
"""

import numpy as np

from hoki.constants import BPASS_IMFS, BPASS_METALLICITIES
from hoki.utils.progressbar import print_progress_bar


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
