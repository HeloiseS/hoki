#!/usr/local/bin/python3
"""
Object to calculate the event rate at a certain
lookback time or over binned lookback time
"""

import hoki.csp.utils as utils
from hoki.utils.hoki_object import HokiObject
from hoki.utils.exceptions import HokiFatalError
import numpy as np
from hoki.constants import *

# TODO Add IMF input option

class CSPEventRate(HokiObject, utils.CSP):
    """
    Object to calculate event rates with complex stellar formation histories.

    Parameters
    ----------
    data_folder : `str`
        folder containing the BPASS data files (in units #events per bin)
    binary : boolean
        If `True`, loads the binary files. Otherwise, just loads single stars.
        Default=True

    Attributes
    ----------
    now : float
        The age of the universe.
    bpass_rates : pandas.DataFrame
        The BPASS delay time distributions in #events/yr/M_\odot per metallicity.
    """

    def __init__(self, data_folder, imf, binary=True):
        self.bpass_rates = utils._normalise_rates(utils.load_rates(data_folder, imf, binary=binary))

    def calculate_rate_over_time(self,
                                 sfh_functions,
                                 Z_functions,
                                 event_types,
                                 nr_bins,
                                 return_edges = False
                                 ):
        """
        Calculates the event rates over lookback time.

        Parameters
        ----------
        sfh_functions : `list(function, )`
            An array containing the stellar formation histories functions
            in units M_\\odot per yr over lookback time.
        Z_functions : `list(function, )`
            An array containing functions describing the metallicity over
            lookback time.
        event_types : `list(str, )`
            A list of BPASS event types.
        nr_bins : `int`
            The number of bins to split the lookback time into.
        return_edges : `bool `
            If `True`, also returns the edges of the lookback time bins.
            Default=False

        Returns
        -------
        `numpy.ndarray`
            If `return_edges=False`, returns a numpy array containing the event
            rates.
            If `return_edges=True`, returns a numpy array containing the event
            rates and the edges, eg. `out[0]=event_rates` `out[1]=time_edges`.
        """

        # input sfr objects?
        self._type_check_histories(sfh_functions, Z_functions)
        if isinstance(event_types, type(list)):
            raise HokiFatalError("event_types is not a list. Only a list is taken as input.")

        nr_events = len(event_types)
        nr_sfh = len(sfh_functions)
        output_dtype = np.dtype([(i, np.float64, nr_bins) for i in event_types])
        event_rates = np.zeros(nr_sfh, dtype=output_dtype)

        time_edges = np.linspace(0, self.now, nr_bins+1)


        mass_per_bin = np.array([utils.mass_per_bin(i, time_edges)
                                        for i in sfh_functions])
        metallicity_per_bin = np.array([utils.metallicity_per_bin(i, time_edges)
                                                for i in Z_functions])
        for counter, (mass, Z) in enumerate(zip(mass_per_bin,  metallicity_per_bin)):
            for count, t in enumerate(event_types):

                event_rate = utils._over_time(Z,
                                       mass,
                                       time_edges,
                                       self.bpass_rates[t].T.to_numpy())

                event_rates[counter][count] = event_rate/np.diff(time_edges)

        if return_edges:
            return np.array([event_rates, time_edges])
        else:
            return event_rates

    def calculate_rate_at_time(self,
                               sfh_functions,
                               Z_functions,
                               event_types,
                               t,
                               sample_rate = None
                               ):
        """
        Calculates the event rates at lookback time `t`.

        Parameters
        ----------
        sfh_functions : `list(function, )`
            An array containing the stellar formation histories functions
            in units M_\\odot per yr over lookback time.
        Z_functions : `list(function, )`
            An array containing functions describing the metallicity over
            lookback time.
        event_types : `list(str)`
            A list of BPASS event types.
        t : `float`
            The lookback time, where to calculate the event rate.
        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.

        Returns
        -------
        float
            Returns the event rate at the given lookback time `t`.
        """

        self._type_check_histories(sfh_functions, Z_functions)
        if isinstance(event_types, type(list)):
            raise HokiFatalError("event_types is not a list. Only a list is taken as input.")


        # The setup of these elements could almost all be combined into a function
        # with code that's repeated above.
        nr_sfh = len(sfh_functions)
        output_dtype = np.dtype([(i, np.float64) for i in event_types])
        event_rates = np.zeros(nr_sfh, dtype=output_dtype)

        if sample_rate == None:
            time_edges = BPASS_LINEAR_TIME_EDGES
        else:
            time_edges = np.linspace(0,13.8e9, sample_rate+1)

        mass_per_bin = np.array([utils.mass_per_bin(i, t+time_edges)
                                    for i in sfh_functions])
        metallicity_per_bin = np.array([utils.metallicity_per_bin(i, t+time_edges)
                                                for i in Z_functions])
        for counter, (mass, Z) in enumerate(zip(mass_per_bin, metallicity_per_bin)):
            for count, ty in enumerate(event_types):

                event_rates[counter][count] = utils._at_time(Z, mass, time_edges,self.bpass_rates[ty].T.to_numpy())


        return event_rates
