"""
Authors: Max Briel and Heloise Stevance

Object to contain the stellar formation history.
"""
import matplotlib.pyplot as plt
import numpy as np

import hoki.csp.utils as utils
from hoki.utils.exceptions import (HokiAttributeError, HokiFormatError,
                                   HokiKeyError, HokiTypeError)


def sfherrormessage(func, *args, **kwargs):
    """
    A decorator to automate a repeated error message.
    """
    def wrapper(*args, **kwargs):
        e_message = "DEBUGGING ASSISTANT: make sure the parameters_dict contains " \
            "all the necessary parameters spelt correctly. " \
            "Accepted parameters are: 'tau', 'T0', 'constant', 'alpha', 'beta'"
        try:
            func(*args, **kwargs)
        except KeyError as e:
            raise HokiKeyError(
                f"{e} has not been defined and I need it. "+e_message)
    return wrapper


class SFH(object):
    """
    An object to contain a stellar formation history of a population.

    Attributes
    ----------
    time_axis : numpy.array
        An array containing the given time points of the stellar formation rate

    """

    def __init__(self, time_axis, sfh_type, parameters_dict=None, sfh_arr=None):
        """
        Input
        ------
        time_axis: numpy.ndarray
            An array containing the time axis of the star formation history (SFH).
            Must be given in yr.
        sfh_type : str
            Parametric star formation history type: 'b' for burst, 'c' for constant, 'e' for exponential,
            'de' for delayed exponential, 'dpl' for double power law, 'ln' for lognormal, 'custom' for custom.
        parameters_dict: dict, optional
            Dictionary containing the parameters for the sfh type selected. The list of valid keywords is as follows:
            'tau', 'T0', 'constant', 'alpha', 'beta'. You may only fill your dictionary with the keywords relevant to
            the SFH type of your choice. See the tutorial for more details on the parameterisation.
        sfh_arr: numpy.ndarray, optional
            An array containing the SFH at the time bins - Required if you selected the 'custom' SFH type.
            Must be given in M_solar/yr.
        """

        # Time Axis and Star formation rate initialistation
        self.time_axis = time_axis
        self.sfh = sfh_arr

        # Parameters for the parametric Star Formation Histories
        self.params = parameters_dict
        self._valid_parameters = ['tau', 'T0', 'constant', 'alpha', 'beta']

        self.parametric_sfh_dic = {'b': '_burst_sfh',
                                   'c': '_constant_sfh',
                                   'e': '_exp_sfh',
                                   'de': '_delayed_exp_sfh',
                                   'dpl': '_dble_pwr_law_sfh',
                                   'ln': '_lognormal_sfh',
                                   'custom': None}

        #### Star Formation History Types and Calculations ###
        if sfh_type == "custom" and sfh_arr is not None:
            if len(self.time_axis) != len(sfh_arr):
                raise HokiFormatError(
                    "time_axis and sfh_arr do not have the same length.")

            self._sfh_calculator = lambda x: np.interp(
                x, self.time_axis, self.sfh)

        elif set([sfh_type]) - set(self.parametric_sfh_dic.keys()) == set():
            if sfh_type == 'custom':
                raise HokiFormatError('Something went wrong with your custom SFH. DEBUGGING ASSISTANT: Check that the '
                                      'sfh_arr parameter is given an array with same length as the time_axis')
            try:
                invalid_params = set(self.params.keys()) - \
                    set(self._valid_parameters)
                if invalid_params != set():
                    raise HokiFormatError(
                        f"You submitted invalid parameters: {invalid_params}")
            except AttributeError:
                raise HokiAttributeError("Did you provide a parameters_dict?")

            getattr(self, self.parametric_sfh_dic[sfh_type])()  # (self.params)
            self._sfh_calculator = lambda x: np.interp(
                x, self.time_axis, self.sfh)

        else:
            raise HokiTypeError(f"SFH type not recognised\nDEBUGGING ASSISTANT: Valid options are"
                                f" {list(self.parametric_sfh_dic)}")

    def __call__(self, t):
        """
        Return the stellar formation rate at a given time.

        Input
        -----
        t : `float` or `int`
            A lookback time
        """
        return self._sfh_calculator(t)

    def mass_per_bin(self, time_edges, sample_rate=100):
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

        return utils.mass_per_bin(self._sfh_calculator, time_edges, sample_rate=sample_rate)

    @sfherrormessage
    def _constant_sfh(self):
        self.sfh = np.array([self.params['constant']] * len(self.time_axis))

    @sfherrormessage
    def _burst_sfh(self):
        burst_index = np.argmin(np.abs(self.params['T0'] - self.time_axis))
        sfh = np.array([0.] * len(self.time_axis))
        sfh[burst_index] += self.params['constant']
        self.sfh = sfh

    @sfherrormessage
    def _exp_sfh(self):
        sfh = [np.exp(-(t - self.params['T0']) / self.params['tau']) if t > self.params['T0'] else 0 for t in
               self.time_axis]
        self.sfh = np.array(sfh) * self.params['constant']

    @sfherrormessage
    def _delayed_exp_sfh(self):
        sfh = [(t - self.params['T0']) * np.exp(-(t - self.params['T0']) / self.params['tau']) if t > self.params[
            'T0'] else 0 for t in self.time_axis]
        self.sfh = np.array(sfh) * self.params['constant']

    @sfherrormessage
    def _dble_pwr_law_sfh(self):
        self.sfh = self.params['constant'] / ((self.time_axis / self.params['tau']) ** self.params['alpha'] + (
            self.time_axis / self.params['tau']) ** (-self.params['beta']))

    @sfherrormessage
    def _lognormal_sfh(self):
        self.sfh = self.params['constant'] * (
            (1 / np.sqrt(2 * np.pi * self.params['tau'] ** 2)) * (1 / self.params['tau']) * np.exp(
                - ((np.log(self.time_axis) - self.params['T0']) ** 2) / (2 * self.params['tau'] ** 2)))

    def plot(self, loc=111, **kwargs):
        # return plot
        sfh_plot = plt.subplot(loc)
        sfh_plot.step(self.time_axis, self.sfh, **kwargs)
        return sfh_plot


#
# class SFH(object):
#     """
#     An object to contain a stellar formation history of a population.
#
#     Attributes
#     ----------
#     time_points : `numpy.ndarray`
#         An array containing the given time points of the stellar formation rate
#     sfh_points : `numpy.ndarray`
#         The stellar formation history at the time points.
#     """
#     def __init__(self, parametric_sfh, time_points, sfh_points):
#         """
#         Input
#         ------
#         parametric_sfh : `str` #[Change sfh_type to "parametric_sfh"]
#             blaaaaa [HELOISE FIX]
#         time_axis: `numpy.ndarray`
#             An array containing the time points of the SFH.
#             Must be given in yr.
#         sfh_points: `numpy.ndarray`
#             An array containing the Stellar Formation History at the time points.
#             Must be given in M_solar/yr
#
#         """
#         if parametric_sfh == "custom":
#             if len(time_points) != len(sfh_points):
#                 raise HokiTypeError("time_axis and sfh_points do not have the same length.")
#             self.time_points = time_points
#             self.sfh_points = sfh_points
#             self._sfh = lambda x : np.interp(x, self.time_points, self.sfh_points)
#         else:
#             raise HokiKeyError(f"{parametric_sfh} is not a valid option.")
#
#     def stellar_formation_rate(self, t):
#         """
#         Returns the stellar formation rate at a given time.
#
#         Input
#         -----
#         t : `float`
#             A lookback time
#         """
#         return self._sfh(t)
#
#     def mass_per_bin(self, time_edges):
#         """
#         Gives the mass per bin for the given edges in time.
#
#         Input
#         -----
#         time_edges : `numpy.ndarray`
#             The edges of the bins in which the mass per bin is wanted in yrs.
#
#         Output
#         ------
#         `numpy.ndarray`
#             The mass per time bin given the time edges.
#         """
#
#         return utils.mass_per_bin(self._sfh, time_edges)
#
#
#
# def sfr_parameterisations():
#     pass
