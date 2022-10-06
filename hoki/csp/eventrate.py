"""
Author: Max Briel

Object to calculate the event rate at a certain
lookback time or over binned lookback time
"""

import numpy as np
import numba

from hoki.csp.csp import CSP
import hoki.csp.utils as utils
from hoki.constants import (BPASS_LINEAR_TIME_EDGES,
        BPASS_NUM_METALLICITIES, BPASS_EVENT_TYPES, HOKI_NOW)
from hoki.utils.exceptions import HokiFatalError
from hoki.utils.hoki_object import HokiObject
from hoki import load
from hoki.utils.progressbar import print_progress_bar


class CSPEventRate(HokiObject, CSP):
    """
    Object to calculate event rates using complex stellar formation histories.

    Parameters
    ----------
    data_path : `str`
        Folder containing the BPASS data files (in units #events per bin)
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
    binary : `bool`
        If `True`, loads the binary files. Otherwise, just loads single stars.
        Default=True

    Attributes
    ----------
    bpass_rates : `pandas.DataFrame` (51, (8, 13))
        The BPASS delay time distributions in #events/yr/M_\\odot per metallicity.
        Usage: rates.loc[log_age, (type, metallicity)]

        Note
        -----
        This dataframe has the following structure.
        The index is the log_age as a float.
        The column is a `pandas.MultiIndex` with the event types
        (level=0, `str`) and the metallicity (level=1, `float`)

        |Event Type | Ia      | IIP      |  ... | PISNe | low_mass |
        |Metallicity| 0.00001 | 0.00001  |  ... |  0.04 |    0.04  |
        | log_age   |---------|----------|------|-------|----------|
        |    6.0    |
        |    ...    |                  Event Rate values
        |    11.0   |

    """

    def __init__(self, data_path, imf, binary=True):
        self.bpass_rates = utils._normalise_rates(
            load.rates_all_z(data_path, imf, binary=binary))

        # Has the shape (8, 13, 51) [event_type, metallicity, time_bin]
        self._numpy_bpass_rates = self.bpass_rates[BPASS_EVENT_TYPES].T.to_numpy().reshape((len(BPASS_EVENT_TYPES),13, 51))

    ###################
    # Function inputs #
    ###################
    # Public functions that take SFH and ZEH functions as input

    def at_time(self, SFH, ZEH, event_type_list, t0, sample_rate = 1000):
        """
        Calculates the event rates at lookback time `t0` for functions as input.

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

            ZEH can be the following things:
            - A python callable (function) which takes the lookback time and
              returns the metallicity at the given time.
            - A list of python callables with the above requirement.

        event_type_list : `list(str, )`
            A list of BPASS event types.
            The available types are:
            - Ia
            - IIP
            - II
            - Ib
            - Ic
            - LGRB
            - PISNe
            - low_mass

        t0 : `float`
            The moment in lookback time, where to calculate the the event rate

        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = 1000.
            If a negative value is given, the BPASS binning to calculate
            the event rates.

        Returns
        -------
        `numpy.ndarray` (N, M) [nr_sfh, nr_event_types]
            Returns a `numpy.ndarray` containing the event rates for each sfh
            and metallicity pair (N) and event type (M) at the requested lookback time.
            Usage: event_rates[1]["Ia"]
            (Gives the Ia event rates for the second sfh and metallicity history pair)
        """

        SFH, ZEH = self._type_check_histories(SFH, ZEH)

        if isinstance(event_type_list, type(list)):
            raise HokiFatalError(
                "event_type_list is not a list. Only a list is taken as input.")

        nr_events = len(event_type_list)
        nr_sfh = len(SFH)

        output_dtype = np.dtype([(i, np.float64) for i in event_type_list])
        event_rates = np.empty(nr_sfh, dtype=output_dtype)

        # Define time edges
        time_edges = BPASS_LINEAR_TIME_EDGES if sample_rate < 0 else np.linspace(0, self.now, sample_rate+1)

        bpass_rates = self._numpy_bpass_rates[[BPASS_EVENT_TYPES.index(i) for i in event_type_list]]

        mass_per_bin_list = np.array(
            [utils.mass_per_bin(i, t0+time_edges) for i in SFH])
        metallicity_per_bin_list = np.array(
            [utils.metallicity_per_bin(i, t0+time_edges) for i in ZEH])

        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list, metallicity_per_bin_list)):
            for count in range(nr_events):

                event_rates[counter][count] = utils._at_time(
                    Z_per_bin, mass_per_bin, time_edges, bpass_rates[count])

        return event_rates

    def over_time(self, SFH, ZEH, event_type_list, nr_time_bins, return_time_edges=False):
        """
        Calculates the event rates over lookback time for functions as input.

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

        event_type_list : `list(str, )`
            A list of BPASS event types.

            The available types are:
            - Ia
            - IIP
            - II
            - Ib
            - Ic
            - LGRB
            - PISNe
            - low_mass

        nr_time_bins : `int`
            The number of bins to split the lookback time into.

        return_time_edges : `bool`
            If `True`, also returns the edges of the lookback time bins.
            Default=False

        Returns
        -------
        `numpy.ndarray` (nr_sfh, nr_event_types, nr_time_bins),
                        ((nr_sfh, nr_event_types, nr_time_bins), nr_time_bins)

            The event rates in a 3D matrix with sides, the number of SFH-Z pairs,
            the number of events selected, and the number of time bins choosen.

            If `return_time_edges=False`, returns a `numpy.ndarray` containing the event
            rates.
            Usage: event_rates[1]["Ia"][10]
                (Gives the Ia event rates in bin number 11 for the second
                sfh and metallicity history pair)

            If `return_time_edges=True`, returns a numpy array containing the event
            rates and the edges, eg. `out[0]=event_rates` `out[1]=time_edges`.
        """

        # check and transform the input to the righ type
        SFH, ZEH = self._type_check_histories(SFH, ZEH)

        if isinstance(event_type_list, type(list)):
            raise HokiFatalError(
                "event_type_list is not a list. Only a list is taken as input.")

        nr_events = len(event_type_list)
        nr_sfh = len(SFH)

        output_dtype = np.dtype([(i, np.float64, nr_time_bins)
                                 for i in event_type_list])
        event_rates = np.empty(nr_sfh, dtype=output_dtype)

        time_edges = np.linspace(0, self.now, nr_time_bins+1)

        bpass_rates = self._numpy_bpass_rates[[BPASS_EVENT_TYPES.index(i) for i in event_type_list]]

        mass_per_bin_list = np.array(
            [utils.mass_per_bin(i, time_edges) for i in SFH])
        metallicity_per_bin_list = np.array(
            [utils.metallicity_per_bin(i, time_edges) for i in ZEH])


        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list, metallicity_per_bin_list)):
            for count in range(nr_events):
                event_rate = utils._over_time(Z_per_bin,
                                              mass_per_bin,
                                              time_edges,
                                              bpass_rates[count])
                event_rates[counter][count] = event_rate/np.diff(time_edges)

        if return_time_edges:
            return np.array([event_rates, time_edges], dtype=object)
        else:
            return event_rates

    ####################
    # Grid calculators #
    ####################
    # Public functions that take a 2D SFH split per metallicity (13, nr_time_points).

    def grid_at_time(self, SFH_list, time_points, event_type_list, t0, sample_rate=1000):
        """
        Calculates event rates for the given BPASS event types for the
        given SFH in a 2D grid (over BPASS metallicity and time_points)


        Parameters
        ----------
        SFH_list : `numpy.ndarray` (N, 13, M) [nr_sfh, metalllicities, time_points]
            A list of N stellar formation histories divided into BPASS metallicity bins,
            over lookback time points with length M.

        time_points : `numpy.ndarray` (M)
            An array of the lookback time points of length N2 at which the SFH is given in the SFH_list.

        event_type_list : `list(str, )`
            A list of BPASS event types.

            The available types are:
            - Ia
            - IIP
            - II
            - Ib
            - Ic
            - LGRB
            - PISNe
            - low_mass

        t0 : `float`
            The moment in lookback time, where to calculate the the event rate

        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = 1000.
            If a negative value is given, the BPASS binning to calculate
            the event rates.

        Returns
        ------
        event_rate_list : `numpy.ndarray` (N, 13, nr_events)
            A numpy array containing the event rates per SFH (N),
            per metallicity (13), per event type (nr_events) at t0.

        """
        nr_sfh = SFH_list.shape[0]

        nr_events = len(event_type_list)
        bpass_rates =  self._numpy_bpass_rates[[BPASS_EVENT_TYPES.index(i) for i in event_type_list]]

        event_rate_list = np.empty((nr_sfh, 13, nr_events), dtype=np.float64)

        for i in range(nr_sfh):
            print_progress_bar(i, nr_sfh)
            event_rate_list[i] = self._grid_rate_calculator_at_time(bpass_rates,
                                                         SFH_list[i],
                                                         time_points,
                                                         t0,
                                                         sample_rate)
        return event_rate_list

    def grid_over_time(self, SFH_list, time_points, event_type_list, nr_time_bins, return_time_edges=False):
        """
        Calculates event rates for the given BPASS event types for the
        given Stellar Formation Histories


        Parameters
        ----------
        SFH_list : `numpy.ndarray` (N, 13, M) [nr_sfh, metalllicities, time_points]
            A list of N stellar formation histories divided into BPASS metallicity bins,
            over lookback time points with length M.

        time_points : `numpy.ndarray` (M)
            An array of the lookback time points of length N2 at which the SFH is given in the SFH_list.

        event_type_list : `list(str, )`
            A list of BPASS event types.

            The available types are:
            - Ia
            - IIP
            - II
            - Ib
            - Ic
            - LGRB
            - PISNe
            - low_mass

        nr_time_bins : `int`
            The number of time bins in which to divide the lookback time

        return_time_edges : `bool`
            If `True`, also returns the edges of the lookback time bins.
            Default=False

        Returns
        ------
        event_rate_list : `numpy.ndarray` (N, 13, nr_events, nr_time_bins)
                                          ((N, 13, nr_events, nr_time_bins), time_edges)
            A numpy array containing the event rates per galaxy (N),
            per metallicity (13), per event type (nr_events) and per time bins (nr_time_bins).

            If `return_time_edges=True`, returns a numpy array containing the event
            rates and the time edges, eg. `out[0]=event_rates_list` `out[1]=time_edges`.

        """
        nr_sfh = SFH_list.shape[0]

        nr_events = len(event_type_list)
        time_edges = np.linspace(0,self.now, nr_time_bins+1)
        bpass_rates =  self._numpy_bpass_rates[[BPASS_EVENT_TYPES.index(i) for i in event_type_list]]

        event_rate_list = np.zeros((nr_sfh, 13, nr_events, nr_time_bins), dtype=np.float64)

        for i in range(nr_sfh):
            print_progress_bar(i, nr_sfh)
            event_rate_list[i] = self._grid_rate_calculator_over_time(bpass_rates,
                                                         SFH_list[i],
                                                         time_points,
                                                         nr_time_bins)

        if return_time_edges:
            return np.array([event_rate_list, time_edges], dtype=object)
        else:
            return event_rate_list

    #########################
    # Grid rate calculators #
    #########################
    # Private functions to calculate the rate using a 2D SFH split per metallicity
    # (13, nr_time_points). They use numba in a parallised manner to speed up
    # the calculation.

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _grid_rate_calculator_at_time(bpass_rates, SFH, time_points, t0, sample_rate=1000):
        """
        Calculates the event rates for the given rates and 2D SFH at a time.

        Note
        ----
        This function checks the `bpass_rates` for how many event rates have to
        be calculated, but does not know what the event types are.
        Furthermore, the SFH is a 2D matrix of size (13,time_points).

        Parameters
        ----------
        bpass_rates : `numpy.ndarray` (M, 13, 51) [event_type, metallicity, time_bin]
            Numpy array containing the BPASS event rates per event type (M),
            metallicity (13) and BPASS time bins (51).

        SFH : `numpy.ndarray` (13, N) [metallicity, time_points]
            Gives the SFH for each metallicity at the time_points.

        time_points : `numpy.ndarray` (N)
            The time points at which the SFH is sampled (N)

        t0 : 'float'
            The lookback time at which to calculate the event rates

        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = 1000.

        Returns
        -------
        event_rates : `numpy.ndarray` (13, M)
            A numpy array containing the event rates per metallicity (13) per event type (M)
            at the given time
        """

        nr_event_type = bpass_rates.shape[0]
        event_rates = np.empty((13, nr_event_type), dtype=np.float64)

        time_edges = np.linspace(0, HOKI_NOW, sample_rate+1)

        mass_per_bin_list = np.empty((13, sample_rate), dtype=np.float64)

        # Calculate the mass per bin for each metallicity
        for i in numba.prange(13):
            # The input parameter here (sample_rate) is between the time sampled points
            mass_per_bin_list[i] = utils._optimised_mass_per_bin(time_points, SFH[i], t0+time_edges, sample_rate=25)

        # Loop over the metallicities
        for counter in numba.prange(13):

            # Loop over the event types
            for count in numba.prange(nr_event_type):
                event_rates[counter][count] = utils._at_time(np.ones(sample_rate)*BPASS_NUM_METALLICITIES[counter],
                                                mass_per_bin_list[counter],
                                                time_edges,
                                                bpass_rates[count])[0]
        return event_rates

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _grid_rate_calculator_over_time(bpass_rates, SFH, time_points, nr_time_bins):
        """
        Calculates the event rates for specific BPASS rates over time

        Note
        ----
        This function checks the bpass_rates for how many event rates have to
        be calculated, but does not know what the event types are.
        Furthermore, the SFH is a 2D matrix of size (13, time_points).

        Parameters
        ----------
        bpass_rates : `numpy.ndarray` (M, 13, 51) [event_type, metallicity, time_bin]
            Numpy array containing the BPASS event rates per event type (M), metallicity and BPASS time bin.

        SFH : `numpy.ndarray` (13, N) [metallicity, SFH_time_sampling_points]
            Gives the SFH for each metallicity at the time_points

        time_points : `numpy.ndarray`
            The time points at which the SFH is sampled (N)

        nr_time_bins : `int`
            The number of time points in which to split the lookback time (final binning)

        Returns
        -------
        event_rates : `numpy.ndarray` (13, M, nr_time_bins)
            A numpy array containing the event rates per metallicity (13)
            per event type (M) per time bins.

        """
        nr_event_type = bpass_rates.shape[0]
        event_rates = np.empty((13, nr_event_type, nr_time_bins), dtype=np.float64)
        time_edges = np.linspace(0, HOKI_NOW, nr_time_bins+1)
        mass_per_bin_list = np.empty((13, nr_time_bins), dtype=np.float64)

        # Calculate the mass per bin for each metallicity
        for i in numba.prange(13):
            mass_per_bin_list[i] = utils._optimised_mass_per_bin(time_points, SFH[i], time_edges, sample_rate=25)

        # Loop over the metallcities
        for counter in numba.prange(13):

            # Loop over the event types
            for count in numba.prange(nr_event_type):

                event_rate = utils._over_time(np.ones(nr_time_bins)*BPASS_NUM_METALLICITIES[counter],
                                              mass_per_bin_list[counter],
                                              time_edges,
                                              bpass_rates[count])

                event_rates[counter][count] = event_rate/np.diff(time_edges)
        return event_rates
