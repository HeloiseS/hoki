"""
Object to calculate the spectra at a certain
lookback time or over a binned lookback time
"""
import numpy as np


import hoki.csp.utils as utils
from hoki.csp.csp import CSP
from hoki.constants import *
from hoki.utils.hoki_object import HokiObject


class CSPSpectra(HokiObject, CSP):
    """
    Object to calculate synthetic spectra with complex stellar formation histories.

    Parameters
    ----------
    data_path : `str`
        folder containing the BPASS data files

    binary : boolean
        If `True`, loads the binary files. Otherwise, just loads single stars.
        Default=True

    Attributes
    ----------
    bpass_spectra : `numpy.ndarray` (13, 51, 100000) [(metallicity, log_age, wavelength)]
        A 3D numpy array containing all the BPASS spectra for a specific imf
        and binary or single star population.
        Usage: spectra[1][2][1000]
                (gives L_\\odot for Z=0.0001 and log_age=6.2 at 999 Angstrom)
    """

    def __init__(self, data_path, imf, binary=True):
        self.bpass_spectra = utils._normalise_spectrum(
            utils.load_spectra(data_path, imf, binary=binary))

    def calculate_spec_over_time(self,
                                 SFH,
                                 ZEH,
                                 nr_time_bins,
                                 return_time_edges=False
                                 ):
        """
        Calculates spectra over lookback time.

        Parameters
        ----------
        SFH : `python callable`,
              `hoki.csp.sfh.SFH`,
              `list(hoki.csp.sfh.SFH, )`,
              `list(callable, )`
            SFH can be the following things:
            - A python callable (function) which takes the lookback time and
              returns the stellar formation rate in units M_\\odot per yr at
              the given time.
            - A list of python callables with the above requirement.
            - A `hoki.csp.sfh.SFH` object.
            - A list of `hoki.csp.sfh.SFH` objects.

        ZEH : callable, `list(callable, )`
            ZEH can be the following thins:
            - A python callable (function) which takes the lookback time and
              returns the metallicity at the given time.
            - A list of python callables with the above requirement.

        nr_time_bins : `int`
            The number of bins to split the lookback time into.

        return_time_edges : `bool`
            If `True`, also returns the edges of the lookback time bins.
            Default=False

        Returns
        -------
        `numpy.ndarray`
            If `return_time_edges=False`, returns a `numpy.ndarray` containing a
            spectrum per bin.
            Shape: [len(sfh), nr_time_bins, 100000]
            Usage: spectra[0][10][99]
                    (gives the L_\\odot at 100 Angstrom for the 11th bin
                    for the first sfh and metallicity history pair)

            If `return_time_edges=True`, returns a `numpy.ndarray` containing a
            spectrum per bin and the bin edges, eg. `out[0]=spectra`
            `out[1]=time_edges`.
        """

        # Check if in correct input format and return list objects
        SFH, ZEH = self._type_check_histories(SFH, ZEH)

        # Number of Stellar Formation Histories to loop through
        nr_sfh = len(SFH)

        # Initialise binning
        time_edges = np.linspace(0, self.now, nr_time_bins + 1)
        # Calculate mass and average metallicity per bin
        mass_per_bin_list = np.array(
            [utils.mass_per_bin(i, time_edges) for i in SFH])
        metallicity_per_bin_list = np.array(
            [utils.metallicity_per_bin(i, time_edges) for i in ZEH])

        output_spectra = np.zeros(
            (nr_sfh, nr_time_bins, 100000), dtype=np.float64)

        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list,  metallicity_per_bin_list)):
            spec = utils._over_time_spectrum(Z_per_bin,
                                             mass_per_bin,
                                             time_edges,
                                             self.bpass_spectra)

            output_spectra[counter] = spec / np.diff(time_edges)[:, None]
        if return_time_edges:
            return np.array([output_spectra, time_edges])
        else:
            return output_spectra

    def calculate_spec_at_time(self,
                               SFH,
                               ZEH,
                               t0,
                               sample_rate=None
                               ):
        """
        Calculates the spectrum at lookback time `t0`.

        Parameters
        ----------
        SFH : `python callable`,
              `hoki.csp.sfh.SFH`,
              `list(hoki.csp.sfh.SFH, )`,
              `list(callable, )`
            SFH can be the following things:
            - A python callable (function) which takes the lookback time and
              returns the stellar formation rate in units M_\\odot per yr at
              the given time.
            - A list of python callables with the above requirement.
            - A `hoki.csp.sfh.SFH` object.
            - A list of `hoki.csp.sfh.SFH` objects.

        ZEH : callable, `list(callable, )`
            ZEH can be the following thins:
            - A python callable (function) which takes the lookback time and
              returns the metallicity at the given time.
            - A list of python callables with the above requirement.

        t0 : `float`
            The lookback time, where to calculate the event rate.

        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = None.

            The default setting uses BPASS binning to calculate the event rates.

        Returns
        -------
        `numpy.ndarray`
            Returns a `numpy.ndarray` containing the spectra for each sfh
            and metallicity pair at the requested lookback time.
            Shape: [len(sfh), 100000]
            Usage: event_rates[1][999]
                (Gives the L_\\odot at 1000 Angstrom for the second
                sfh and metallicity history pair)
        """

        # check if in correct input format and return list objects
        SFH, ZEH = self._type_check_histories(SFH, ZEH)

        # The setup of these elements could almost all be combined into a function
        # with code that's repeated above. Similarly, with the event rate calculation.
        nr_sfh = len(SFH)

        output_spectrum = np.zeros((nr_sfh, 100000), dtype=np.float64)

        if sample_rate is None:
            time_edges = BPASS_LINEAR_TIME_EDGES
        else:
            time_edges = np.linspace(0, 13.8e9, sample_rate + 1)

        mass_per_bin_list = np.array(
            [utils.mass_per_bin(i, t0 + time_edges) for i in SFH])
        metallicity_per_bin_list = np.array([utils.metallicity_per_bin(i, t0 + time_edges)
                                             for i in ZEH])

        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list, metallicity_per_bin_list)):
            output_spectrum[counter] = utils._at_time_spectrum(Z_per_bin,
                                                               mass_per_bin,
                                                               time_edges,
                                                               self.bpass_spectra
                                                               )

        return output_spectrum
