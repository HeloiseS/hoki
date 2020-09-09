"""
Author: Max Briel

Objects and pipelines that compile BPASS data files into more convenient, more pythonic data types
"""
import abc
import numpy as np

from hoki.constants import (BPASS_IMFS, BPASS_METALLICITIES, BPASS_TIME_BINS,
                            BPASS_WAVELENGTHS)
from hoki.utils.progressbar import print_progress_bar


class _CompilerBase(abc.ABC):
    def __init__(self, input_folder, output_folder, imf, binary=True,
                 verbose=False):
        if verbose:
            _print_welcome()

        # Check population type
        star = "bin" if binary else "sin"

        # check IMF key
        if imf not in BPASS_IMFS:
            raise HokiKeyError(
                f"{imf} is not a BPASS IMF. Please select a correct IMF.")

        # Setup the numpy output
        output = np.empty(self._shape(), dtype=np.float64)

        # loop over all the metallicities and load all the spectra
        for num, metallicity in enumerate(BPASS_METALLICITIES):
            print_progress_bar(num, 12)
            output[num] = self._load_single(
                f"{input_folder}/{self._input_name()}-{star}-{imf}.z{metallicity}.dat"
            )

        # pickle the datafile
        np.save(f"{output_folder}/{self._output_name()}-{star}-{imf}", output)
        self.output = output
        print(
            f"Compiled data stored in {output_folder} as '{self._output_name()}-{star}-{imf}.npy'")
        if verbose:
            _print_exit_message()
        return

    @abc.abstractmethod
    def _input_name(self):
        return

    @abc.abstractmethod
    def _output_name(self):
        return

    @abc.abstractmethod
    def _shape(self):
        return

    @abc.abstractmethod
    def _load_single(self, path):
        return


class SpectraCompiler(_CompilerBase):
    """
    Pipeline to load the BPASS spectra txt files and save them as a 3D
    `numpy.ndarray` binary file.

    Attributes
    ----------
    output : `numpy.ndarray` (N_Z, N_age, N_lam) [(metallicity, log_age, wavelength)]
        A 3D numpy array containing all the BPASS spectra for a specific imf
        and binary or single star population.
        Usage: spectra[1][2][1000]
                (gives L_\\odot for Z=0.0001 and log_age=6.2 at 999 Angstrom)
    """

    def _input_name(self):
        return "spectra"

    def _output_name(self):
        return "all_spectra"

    def _shape(self):
        return (
            len(BPASS_METALLICITIES),
            len(BPASS_TIME_BINS),
            len(BPASS_WAVELENGTHS)
        )

    def _load_single(self, path):
        return np.loadtxt(path).T[1:, :]


class EmissivityCompiler(_CompilerBase):
    """
    Pipeline to load the BPASS ionizing txt files and save them as a 3D
    `numpy.ndarray` binary file.

    Attributes
    ----------
    output : `numpy.ndarray` (N_Z, N_age, 4) [(metallicity, log_age, band)]
        A 3D numpy array containing all the BPASS emissivities (Nion [1/s],
        L_Halpha [ergs/s], L_FUV [ergs/s/A], L_NUV [ergs/s/A]) for a specific
        imf and binary or single star population.
        Usage: spectra[1][2][0]
                (gives Nion for Z=0.0001 and log_age=6.2)
    """
    def _input_name(self):
        return "ionizing"

    def _output_name(self):
        return "all_ionizing"

    def _shape(self):
        return (
            len(BPASS_METALLICITIES),
            len(BPASS_TIME_BINS),
            4
        )

    def _load_single(self, path):
        return np.loadtxt(path)[:, 1:]


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
