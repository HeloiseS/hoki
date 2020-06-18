"""
Object to contain the stellar formation history.
"""
import numpy as np
import hoki.csp.utils as utils
from hoki.utils.exceptions import HokiKeyError


class SFH(object):
    """
    An object to contain a stellar formation history of a population.

    Attributes
    ----------
    time_points : `numpy.ndarray`
        An array containing the given time points of the stellar formation rate
    sfh_points : `numpy.ndarray`
        The stellar formation history at the time points.
    """
    def __init__(self, time_points, sfh_points, parametric_sfh):
        """
        Input
        ------
        time_points: `numpy.ndarray`
            An array containing the time points of the SFH.
            Must be given in yr.
        sfh_points: `numpy.ndarray`
            An array containing the Stellar Formation History at the time points.
            Must be given in M_solar/yr
        parametric_sfh : `str` #[Change sfh_type to "parametric_sfh"]
            blaaaaa [HELOISE FIX]
        """
        if parametric_sfh == "custom":
            self.time_points = time_points
            self.sfh_points = sfh_points
        else:
            raise HokiKeyError(f"{parametric_sfh} is not a valid option.")

    def stellar_formation_rate(self, t):
        """
        Returns the stellar formation rate at a given time.

        Input
        -----
        t : `float`
            A lookback time
        """
        return np.interp(t, self.time_points, self.sfh_points)

    def mass_per_bin(self, time_edges):
        """
        Gives the mass per bin for the given edges in time.

        Input
        -----
        time_edges : `numpy.ndarray`
            The edges of the bins in which the mass per bin is wanted in yrs.

        Output
        ------
        `numpy.ndarray`
            The mass per time bin given the time edges.
        """

        return utils.mass_per_bin(self.sfh, time_edges)



def sfr_parameterisations():
    pass
