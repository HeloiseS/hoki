"""
Author: Max Briel

Object to calculate the event rate at a certain
lookback time or over binned lookback time
"""

import numpy as np

from hoki.csp.csp import CSP
import hoki.csp.utils as utils
from hoki.constants import *
from hoki.utils.exceptions import HokiFatalError
from hoki.utils.hoki_object import HokiObject
from hoki import load

class CSPEventRate(HokiObject, CSP):
    """
    Object to calculate event rates using complex stellar formation histories.

    Notes
    -----

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


    Parameters
    ----------
    data_path : `str`
        Folder containing the BPASS data files (in units #events per bin)
    imf : `str`
        The BPASS identifier for the IMF of the BPASS event rate files.
    binary : `bool`
        If `True`, loads the binary files. Otherwise, just loads single stars.
        Default=True

    Attributes
    ----------
    bpass_rates : `pandas.DataFrame`
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
            load.all_rates(data_path, imf, binary=binary))

    def calculate_rate_over_time(self,
                                 SFH,
                                 ZEH,
                                 event_type_list,
                                 nr_time_bins,
                                 return_time_edges=False
                                 ):
        """
        Calculates the event rates over lookback time.

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
        `numpy.ndarray`
            If `return_time_edges=False`, returns a `numpy.ndarray` containing the event
            rates.
            Shape: [len(sfh), len(event_type_list), nr_time_bins)]
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
        event_rates = np.zeros(nr_sfh, dtype=output_dtype)

        time_edges = np.linspace(0, self.now, nr_time_bins+1)

        mass_per_bin_list = np.array([utils.mass_per_bin(np.vectorize(i), time_edges)
                                      for i in SFH])
        metallicity_per_bin_list = np.array([utils.metallicity_per_bin(np.vectorize(i), time_edges)
                                             for i in ZEH])
        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list,  metallicity_per_bin_list)):
            for count, event_type in enumerate(event_type_list):

                event_rate = utils._over_time(Z_per_bin,
                                              mass_per_bin,
                                              time_edges,
                                              self.bpass_rates[event_type].T.to_numpy())

                event_rates[counter][count] = event_rate/np.diff(time_edges)

        if return_time_edges:
            return np.array([event_rates, time_edges])
        else:
            return event_rates

    def calculate_rate_at_time(self,
                               SFH,
                               ZEH,
                               event_type_list,
                               t0,
                               sample_rate=1000
                               ):
        """
        Calculates the event rates at lookback time `t0`.

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

        t0 : `float`
            The moment in lookback time, where to calculate the the event rate

        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = 1000.
            If a negative value is given, the BPASS binning to calculate
            the event rates.

        Returns
        -------
        `numpy.ndarray`
            Returns a `numpy.ndarray` containing the event rates for each sfh
            and metallicity pair and event type at the requested lookback time.
            Shape: [len(sfh), len(event_type_list))]
            Usage: event_rates[1]["Ia"]
                (Gives the Ia event rates for the second
                sfh and metallicity history pair)
        """

        SFH, ZEH = self._type_check_histories(SFH, ZEH)
        if isinstance(event_type_list, type(list)):
            raise HokiFatalError(
                "event_type_list is not a list. Only a list is taken as input.")

        # The setup of these elements could almost all be combined into a function
        # with code that's repeated above.
        nr_sfh = len(SFH)
        output_dtype = np.dtype([(i, np.float64) for i in event_type_list])
        event_rates = np.zeros(nr_sfh, dtype=output_dtype)

        if sample_rate < 0:
            time_edges = BPASS_LINEAR_TIME_EDGES
        else:
            time_edges = np.linspace(0, self.now, sample_rate+1)

        mass_per_bin_list = np.array([utils.mass_per_bin(i, t0+time_edges)
                                      for i in SFH])
        metallicity_per_bin_list = np.array([utils.metallicity_per_bin(i, t0+time_edges)
                                             for i in ZEH])
        for counter, (mass_per_bin, Z_per_bin) in enumerate(zip(mass_per_bin_list, metallicity_per_bin_list)):
            for count, event_type in enumerate(event_type_list):

                event_rates[counter][count] = utils._at_time(
                    Z_per_bin, mass_per_bin, time_edges, self.bpass_rates[event_type].T.to_numpy())

        return event_rates
