#!/usr/local/bin/python3
"""
Object to calculate the event rate at a certain
lookback time or over binned lookback time
"""

import hoki.csp.utils as utils
from hoki.utils.hoki_object import HokiObject
from scipy import interpolate
import numpy as np
from hoki.constants import *

class CSPEventRate(HokiObject, utils.CSP):


    pass

class CSPEventRateOverTime(CSPEventRate):

    def __init__(self, data_folder, nr_bins, binary=True, age_universe=HOKI_NOW):
        self.nr_bins = nr_bins
        self.time_edges = np.linspace(0, age_universe, nr_bins+1)
        self.bpass_rates = utils._normalise_rates(utils._load_files(data_folder, "supernova", binary=binary), BPASS_LINEAR_TIME_INTERVALS)

        self.sfh = None
        self.metallicity = None
        self.mass_per_bin = None
        self.metallicty_per_bin = None

    def calculate_rate(self, metallicity, SFH, event_types):
        # input sfr object
        # input 2 arrays of equal length
        # input 2 arrays of many arrays
        # currently both are scipy.interpolate.splrep (spline representations)
        # or arrays of them
        # TODO: ADD BETTER TYPE CHECK!
        if isinstance(metallicity, type(list)):
            raise Exception("metallicity is not a list. Only list are taken as input")
        if isinstance(SFH, type(list)):
            raise Exception("sfr is not a list. Only lists are taken as input.")
        if isinstance(event_types, type(list)):
            raise Exception("event_types is not a list. Only a list is taken as input.")
        if len(metallicity) != len(SFH):
            raise Expection("metallicity and sfr are not of equal length.")

        self.sfh = SFH
        self.metallicity = metallicity
        nr_sfh = len(SFH)

        nr_events = len(event_types)

        self.event_rates = np.zeros((nr_events, nr_sfh, self.nr_bins))

        self.mass_per_bin = np.array([utils.mass_per_bin(i, self.time_edges)
                                        for i in self.sfh])
        self.metallicity_per_bin = np.array([utils.metallicity_per_bin(i, self.time_edges)
                                                for i in self.metallicity])


        for mass, Z in zip(self.mass_per_bin, self.metallicity_per_bin):
            for count, t in enumerate(event_types):

                event_rate = utils._over_time(Z,
                                       mass,
                                       self.time_edges,
                                       self.bpass_rates[t].T.to_numpy(),
                                       BPASS_LINEAR_TIME_INTERVALS)

                self.event_rates[count] = event_rate/np.diff(self.time_edges)


    # Set a SFR
    # Set metallicity
    # Set a binning
    # Find BPASS metallicities
    # Open the correct files
    # Calculate the Mass per bin
    # Calculate the event rates over time
    # have the possibility to return the event rates

class CSPEventRateAtTime(CSPEventRate):
    pass
