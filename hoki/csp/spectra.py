"""
Object to calculate the spectra at a certain
lookback time or over a binned lookback time
"""
import numpy as np
import numba

import hoki.csp.utils as utils
from hoki.csp.csp import CSP
from hoki.constants import (BPASS_LINEAR_TIME_EDGES,
            HOKI_NOW, BPASS_NUM_METALLICITIES)
from hoki.utils.hoki_object import HokiObject
from hoki import load
from hoki.utils.progressbar import print_progress_bar


class CSPSpectra(HokiObject, CSP):
    """
    Object to calculate synthetic spectra with complex stellar formation histories.

    Parameters
    ----------
    data_path : `str`
        Folder containing the BPASS data files

    imf : `str`
        The BPASS identifier for the IMF of the BPASS event rate files.

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
            load.spectra_all_z(data_path, imf, binary=binary))

    ###################
    # Function inputs #
    ###################
    # Public functions that take SFH and ZEH functions as input

    def at_time(self, SFH, ZEH, t0, sample_rate=1000):
        """
        Calculates the spectrum at lookback time `t0` for functions as input.

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
            Default = 1000.
            If a negative value is given, the BPASS binning to calculate
            the spectra.

        Returns
        -------
        `numpy.ndarray`
            Returns a `numpy.ndarray` containing the spectra for each sfh
            and metallicity pair at the requested lookback time.
            Shape: [len(sfh), 100000]
            Usage: event_spectra[1][999]
                (Gives the L_\\odot at 1000 Angstrom for the second
                sfh and metallicity history pair)
        """

        # check if in correct input format and return list objects
        SFH, ZEH = self._type_check_histories(SFH, ZEH)

        nr_sfh = len(SFH)

        output_spectra = np.empty((nr_sfh, 100000), dtype=np.float64)

        time_edges = BPASS_LINEAR_TIME_EDGES if sample_rate < 0 \
                                else np.linspace(0, self.now, sample_rate+1)

        mass_per_bin_list = np.array(
            [utils.mass_per_bin(i, t0 + time_edges) for i in SFH])
        metallicity_per_bin_list = np.array(
            [utils.metallicity_per_bin(i, t0 + time_edges) for i in ZEH])
        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list, metallicity_per_bin_list)):
            output_spectra[counter] = utils._at_time(Z_per_bin,
                                                     mass_per_bin,
                                                     time_edges,
                                                     self.bpass_spectra)

        return output_spectra

    def over_time(self, SFH, ZEH, nr_time_bins, return_time_edges=False):
        """
        Calculates spectra over lookback time with functions as input.

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
        time_edges = np.linspace(0, self.now, nr_time_bins+1)

        # Calculate mass and average metallicity per bin
        mass_per_bin_list = np.array(
            [utils.mass_per_bin(i, time_edges) for i in SFH])
        metallicity_per_bin_list = np.array(
            [utils.metallicity_per_bin(i, time_edges) for i in ZEH])

        output_spectra = np.empty(
            (nr_sfh, nr_time_bins, 100000), dtype=np.float64)

        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list,  metallicity_per_bin_list)):
            spec = utils._over_time_spectrum(Z_per_bin,
                                             mass_per_bin,
                                             time_edges,
                                             self.bpass_spectra)
            output_spectra[counter] = spec / np.diff(time_edges)[:, None]

        if return_time_edges:
            return np.array([output_spectra, time_edges], dtype=object)
        else:
            return output_spectra

    ####################
    # Grid calculators #
    ####################
    # Public functions that take a 2D SFH split per metallicity (13, nr_time_points).

    def grid_at_time(self, SFH_list, time_points, t0, sample_rate=1000):
        """
        Calculates spectra for the given SFH 2D grids
        (over metallicity and per time_points) at `t0`


        Parameters
        ----------
        SFH_list : `numpy.ndarray` (N, 13, M) [nr_sfh, metalllicities, time_points]
            A list of N stellar formation histories divided into BPASS
            metallicity bins, over lookback time points with length M.

        time_points : `numpy.ndarray` (M)
            An array of the lookback time points of length N2 at which
            the SFH is given in the SFH_list.

        t0 : `float`
            The moment in lookback time, where to calculate the the event rate

        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = 1000.

        Returns
        ------
        `numpy.ndarray` (N, 13, 100000)
            A numpy array containing the spectra per SFH (N),
            per metallicity (13), per wavelength (100000) at t0.
        """

        nr_sfh = SFH_list.shape[0]

        output_spectra = np.empty((nr_sfh, 13, 100000), dtype=np.float64)

        for i in range(nr_sfh):
            print_progress_bar(i, nr_sfh)
            output_spectra[i] = self._grid_rate_calculator_at_time(self.bpass_spectra,
                                                         SFH_list[i],
                                                         time_points,
                                                         t0,
                                                         sample_rate)
        return output_spectra

    def grid_over_time(self, SFH_list, time_points, nr_time_bins, return_time_edges=False):
        """
        Calculates spectra for the given 2D Stellar Formation Histories
        (over metallicity and per time_points) over lookback time.

        Parameters
        ----------
        SFH_list : `numpy.ndarray` (N, 13, M) [nr_sfh, metalllicities, time_points]
            A list of N stellar formation histories divided into BPASS metallicity bins,
            over lookback time points with length M.

        time_points : `numpy.ndarray` (M)
            An array of the lookback time points of length N2 at which the SFH is given in the SFH_list.

        nr_time_bins : `int`
            The number of time bins in which to divide the lookback time

        return_time_edges : `bool`
            If `True`, also returns the edges of the lookback time bins.
            Default=False

        Returns
        ------
        `numpy.ndarray` (N, 13, nr_time_bins, 100000)
            A numpy array containing the spectra per SFH (N),
            per metallicity (13), per wavelength (100000) and per time bins (nr_time_bins).

            If `return_time_edges=True`, returns a numpy array containing the spectra
            and the time edges, eg. `out[0]=output_spectra` `out[1]=time_edges`.
        """
        nr_sfh = SFH_list.shape[0]

        output_spectra = np.zeros((nr_sfh, 13, nr_time_bins, 100000), dtype=np.float64)
        time_edges = np.linspace(0, HOKI_NOW, nr_time_bins+1)

        for i in range(nr_sfh):
            print_progress_bar(i, nr_sfh)
            output_spectra[i] = self._grid_rate_calculator_over_time(self.bpass_spectra,
                                                         SFH_list[i],
                                                         time_points,
                                                         nr_time_bins)
        if return_time_edges:
            return np.array([output_spectra, time_edges], dtype=object)
        else:
            return output_spectra


    #########################
    # Grid rate calculators #
    #########################
    # Private functions to calculate the rate using a 2D SFH split per metallicity (13, nr_time_points).

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _grid_rate_calculator_at_time(bpass_spectra, SFH, time_points, t0, sample_rate=1000):
        """
        Calculates the spectrum for the given 2D SFH at a specific time
        split up per metallicity

        Parameters
        ----------
        bpass_spectra : `numpy.ndarray` (13, 51, 100000) (metallicity, time_bin, wavelength)
            Numpy array containing the BPASS spectra
            metallicity (13), BPASS time bins (51), and wavelength (100000)

        SFH : `numpy.ndarray` (13, N) [metallicity, time_points]
            Gives the SFH for each metallicity at the time_points.

        time_points : `numpy.ndarray` (N)
            The time points at which the SFH is sampled (N)

        t0 : 'float'
            The lookback time at which to calculate the spectra

        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = 1000.

        Returns
        -------
        `numpy.ndarray` (13, 100000)
            Numpy array containing the spectrum per metallicity (13)
            per wavelength (100000) at the given time.
        """

        spectra = np.empty((13, 100000), dtype=np.float64)


        time_edges = np.linspace(0, HOKI_NOW, sample_rate+1)
        mass_per_bin_list = np.empty((13, sample_rate), dtype=np.float64)

        # # Calculate the mass per bin for each metallicity
        for i in numba.prange(13):
            # The input parameter here (sample_rate) is between the time sampled points
            mass_per_bin_list[i] = utils._optimised_mass_per_bin(time_points, SFH[i], t0+time_edges, sample_rate=25)

        # Loop over the metallicities
        for counter in numba.prange(13):
            spectra[counter] = utils._at_time(
                np.ones(sample_rate)*BPASS_NUM_METALLICITIES[counter],
                                           mass_per_bin_list[counter],
                                           time_edges,
                                           bpass_spectra)

        return spectra

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _grid_rate_calculator_over_time(bpass_spectra, SFH, time_points, nr_time_bins):
        """
        Calculates the spectrum per metallicity for a 2D grid SFH over time

        Parameters
        ----------
        bpass_spectra : `numpy.ndarray` (13, 51, 100000) [metallicity, time_bin, wavelength]
            Numpy array containing the BPASS spectra per metallicity,
            BPASS time bin, and wavelength.

        SFH : `numpy.ndarray` (13, N) [metallicity, SFH_time_sampling_points]
            Gives the SFH for each metallicity at the time_points

        time_points : `numpy.ndarray`
            The time points at which the SFH is sampled (N)

        nr_time_bins : `int`
            Bins of the final lookback time

        Returns
        -------
        `numpy.ndarray` (13, nr_time_bins, 100000)
            Numpy array containing the spectra per metallicity (13)
            and per time bins.

        """
        spectra = np.empty((13, nr_time_bins, 100000), dtype=np.float64)
        time_edges = np.linspace(0, HOKI_NOW, nr_time_bins+1)
        mass_per_bin_list = np.empty((13, nr_time_bins), dtype=np.float64)

        # Calculate the mass per bin for each metallicity
        for i in numba.prange(13):
            mass_per_bin_list[i] = utils._optimised_mass_per_bin(time_points, SFH[i], time_edges, sample_rate=25)

        # Loop over the metallicities
        for counter in numba.prange(13):
            spectrum = utils._over_time_spectrum(np.ones(nr_time_bins)*BPASS_NUM_METALLICITIES[counter],
                                          mass_per_bin_list[counter],
                                          time_edges,
                                          bpass_spectra)
            for count, i in enumerate(np.diff(time_edges)):
                spectra[counter, count] = spectrum[count]/i

        return spectra
