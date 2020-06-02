"""
Object to contain the stellar formation history.
"""
from scipy import interpolate
import numpy as np


class SFH(object):
    """
    An object to contain a stellar formation history of a population.

    Attributes
    ----------
    time_bins : numpy.array
        An array containing the given time points of the stellar formation rate

    """
    def __init__(self, time_bins, sfh, model_type):
        """
        Input
        ------
        time_bins: numpy.array
            An array containing the time points of the SFH.
            Must be given in yr.
        sfh: numpy array
            An array containing the Stellar Formation History at the time points.
            Must be given in M_solar/yr
        model_type : str
            Determines which stellar model is used.
        """
        self.time_bins = time_bins
        if model_type == "custom":
            self.sfr = interpolate.splrep(time_bins, sfh, k=1)
        else:
            raise TypeError("model type not recognised.")

    def stellar_formation_rate(self, t):
        """
        Returns the stellar formation rate at a given time.

        Input
        -----
        t : float
            A lookback time
        """
        return interpolate.splev(t, self.sfr)

    def mass_per_bin(self, time_edges):
        """
        Gives the mass per bin for the given edges in time.

        Input
        -----
        time_edges : numpy array
            The edges of the bins in which the mass per bin is wanted in yrs.

        Output
        ------
        numpy.array
            The mass per time bin given the time edges.
        """

        return np.array([interpolate.splint(t1, t2, self.sfr)
                for t1, t2 in zip(time_edges[:-1], time_edges[1:])])



def sfr_parameterisations():
    pass
