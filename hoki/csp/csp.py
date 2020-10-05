"""
Author: Max Briel

Parent class of a complex stellar population
"""

from hoki.csp.sfh import SFH
from hoki.utils.exceptions import HokiTypeError, HokiFormatError
from hoki.constants import HOKI_NOW


############################################
# Complex Stellar Populations Parent Class #
############################################


class CSP(object):
    """
    Complex Stellar Population class

    Notes
    -----
    Parent class for `CSPEventRate` and `CSPSpectra`

    Attributes
    ----------
    now : `float`
        The age of the universe.
    """
    now = HOKI_NOW

    def __init__(self):
        pass

    def _type_check_histories(self, sfh, zeh):
        """
        Function to check sfh and zeh are the correct type and transform them
        into a consistent format.

        Notes
        -----
        sfh and zeh can either be a callable python function or a list of
        callables.

        Input
        -----
        sfh
            Stellar formation history
        zeh
            Z evolution history

        Returns
        -------
        `tuple` ([sfh callables,], [zeh callables,])
            A tuple containing the sfh callables and zeh callables as arrays.
        """
        # Check sfh list
        if isinstance(sfh, list):

            # check if zeh is also a list
            if isinstance(zeh, list):
                # have to be equal lengths
                if len(sfh) == len(zeh):
                    # have to be all callables
                    if (all(callable(val) for val in sfh) and all(callable(val) for val in zeh)):
                        return (sfh, zeh)
                    # A non-callable is present
                    else:
                        raise HokiTypeError(
                            "SFH or ZEH contains an object that's not a SFH object or function.")
                # sfh and zeh are not equal length
                else:
                    raise HokiFormatError(
                        "sfh_functions and Z_functions must have the same length.")
            # zeh is not a list
            else:
                # sfh has to be 1 length and a callable, zeh has to be a callable
                if len(sfh) == 1:
                    if (callable(sfh[0]) and callable(zeh)):

                        return (sfh, [zeh])
                    else:
                        raise HokiTypeError(
                            "SFH or ZEH contains an object that's not a SFH object or python callable.")
                else:
                    raise HokiFormatError(
                        "SFH must have length 1, be a python callable, or a SFH object.")
        # sfh is a callable
        elif callable(sfh):
            # zeh is a list
            if isinstance(zeh, list):
                # list has to be 1 long, because sfh is callable
                if len(zeh) == 1:
                    return ([sfh], zeh)
                # list it too long
                else:
                    raise HokiFormatError(
                        "ZEH must be either length 1 or a python callable")

            # zeh is also a callable return
            elif callable(zeh):
                return ([sfh], [zeh])
            else:
                raise HokiTypeError(
                    "ZEH is not a python callable or a list of callables.")
        # sfh cannot be identified
        else:
            raise HokiTypeError(
                "SFH type is not a python callable or a SFH object."
            )
