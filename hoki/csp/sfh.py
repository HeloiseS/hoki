"""
Object to contain the stellar formation history.
"""
import numpy as np
import hoki.csp.utils as utils
from hoki.utils.exceptions import HokiKeyError, HokiTypeError

# [M]: Should metallicity be part of this?

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
    def __init__(self, parametric_sfh, time_points, sfh_points):
        """
        Input
        ------
        parametric_sfh : `str` #[Change sfh_type to "parametric_sfh"]
            blaaaaa [HELOISE FIX]
        time_points: `numpy.ndarray`
            An array containing the time points of the SFH.
            Must be given in yr.
        sfh_points: `numpy.ndarray`
            An array containing the Stellar Formation History at the time points.
            Must be given in M_solar/yr

        """
        if parametric_sfh == "custom":
            if len(time_points) != len(sfh_points):
                raise HokiTypeError("time_points and sfh_points do not have the same length.")
            self.time_points = time_points
            self.sfh_points = sfh_points
            self._sfh = lambda x : np.interp(x, self.time_points, self.sfh_points)
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
        return self._sfh(t)

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

        return utils.mass_per_bin(self._sfh, time_edges)



def sfr_parameterisations():
    pass
