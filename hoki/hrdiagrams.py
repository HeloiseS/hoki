"""
This module implements the HR diagram infrastructure.
"""

import numpy as np
import matplotlib.pyplot as plt
from hoki.constants import *
from matplotlib import ticker
import matplotlib.cm as cm
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError
import warnings
from hoki.utils.hoki_object import HokiObject
import copy


class HRDiagram(HokiObject):
    """
    **A class containing the HR diagram data produced by BPASS.**

    This class is called by the functions hrTL(), hrTg() and hrTTG() in hoki.load and users should
    not need to create an HRDiagram object themselves.

    For more details on the BPASS outputs - and therefore why the data structure is as it is -
    please refer to the manual:
    https://bpass.auckland.ac.nz/8/files/bpassv2_1_manual_accessible_version.pdf

    Note
    -----
    - **HRDiagram supports indexing.** The indexed array is a 51x100x100 np.array that stacked the time weighted arrays
    corresponding to the 3 different abundances.

    - Initialisation from a text file is done through the hoki.load functions

    Parameters
    ----------
    high_H_input : np.ndarray with shape (51x100x100)
        This inputs the HR diagrams corresponding to a hydrogen abundance n1 > 0.4.

    medium_H_input : np.ndarray with shape (51x100x100)
        This inputs the HR diagrams corresponding to a hydrogen abundance E-3 < n1 < 0.4.

    low_H_input : np.ndarray with shape (51x100x100)
        This inputs the HR diagrams corresponding to a hydrogen abundance n1 < E-3.

    hr_type : str - Valid options are 'TL' , 'Tg', 'TTG'
        This tells the class what type of HR diagrams are being given. For more details on what
        the 3 options mean, consult the BPASS manual section on HR diagram isocontours.

    Attributes
    ----------
    self.high_H : np.ndarray (51x100x100)
        HR diagrams for 51 time bins with a hydrogen abundance n1 > 0.4. Time weighted.

    self.medium_H : np.ndarray (51x100x100)
        HR diagrams for 51 time bins with a hydrogen abundance E-3 < n1 < 0.4. Time weighted.

    self.low_H : np.ndarray (51x100x100)
        HR diagrams for 51 time bins with a hydrogen abundance n1 < E-3. Time weighted.

    self.type : str
        Type of HR diagram: TL, Tg or TTG

    self.high_H_not_weighted : np.ndarray (51x100x100)
        HR diagrams for 51 time bins with a hydrogen abundance n1 > 0.4.

    self.medium_H_not_weighted : np.ndarray (51x100x100)
        HR diagrams for 51 time bins with a hydrogen abundance E-3 < n1 < 0.4.

    self.low_H_not_weighted : np.ndarray (51x100x100)
        HR diagrams for 51 time bins with a hydrogen abundance n1 < E-3.

    self._all_H : np.ndarray (51x100x100)
        HR diagrams for 51 time bins - all hydrogen abundances stacked. This attribute is private
        because it can simply be called using the indexing capabilities of the class.

    self.high_H_stacked : np.ndarray (51x100x100)
        HR diagram stacked for a given age range - hydrogen abundance n1 > 0.4. None before calling
        self.stack()

    self.medium_H_stacked : np.ndarray (51x100x100)
        HR diagram stacked for a given age range - hydrogen abundance E-3 < n1 < 0.4. None before
        calling self.stack()

    self.low_H_stacked : np.ndarray (51x100x100)
        HR diagram stacked for a given age range - hydrogen abundance E-3 > n1. None before calling
        self.stack()

    self.all_stacked : np.ndarray (51x100x100)
        HR diagram stacked for a given age range - all abundances added up. None before calling
        self.stack()

    self.t : np.ndarray 1D
        **Class attribute** - The time bins in BPASS - note they are in LOG SPACE

    self.dt : np.ndarray 1D
        **Class attribute** - Time intervals between bins NOT in log space



    """

    # HRD coordinates
    T_coord = np.arange(0.1, 10.1, 0.1)
    L_coord = np.arange(-2.9, 7.1, 0.1)
    G_coord = np.arange(-2.9, 7.1, 0.1)
    TG_coord = np.arange(0.1, 10.1, 0.1)

    def __init__(self, high_H_input, medium_H_input, low_H_input, hr_type):
        """
        Initialisation of HRDiagrams.

        Note
        ----
        Initialisation from a text file is done through the hoki.load functions

        Parameters
        ----------
        high_H_input : np.ndarray with shape (51x100x100)
            This inputs the HR diagrams corresponding to a hydrogen abundance n1 > 0.4.

        medium_H_input : np.ndarray with shape (51x100x100)
            This inputs the HR diagrams corresponding to a hydrogen abundance E-3 < n1 < 0.4.

        low_H_input : np.ndarray with shape (51x100x100)
            This inputs the HR diagrams corresponding to a hydrogen abundance n1 < E-3.

        hr_type : str - Valid options are 'TL' , 'Tg', 'TTG'
            This tells the class what type of HR diagrams are being given. For more details on what
            the 3 options mean, consult the BPASS manual section on HR diagram isocontours.

        """

        # Initialise core attributes
        self.type = str(hr_type)
        self.high_H_not_weighted = high_H_input
        self.medium_H_not_weighted = medium_H_input
        self.low_H_not_weighted = low_H_input

        self._apply_time_weighting()
        self._all_H = self.low_H + self.medium_H + self.high_H

        # Initialise attributes for later
        self.reset_stack()

    def reset_stack(self):
        self.high_H_stacked, self.medium_H_stacked, self.low_H_stacked = np.zeros((100, 100)), \
                                                                         np.zeros((100, 100)), \
                                                                         np.zeros((100, 100))
        self.all_stacked = None
        print("Stack has been reset")

    def stack(self, log_age_min=None, log_age_max=None):
        """
        Creates a stack of HR diagrams within a range of ages

        Parameters
        ----------
        log_age_min : int or float, optional
            Minimum log(age) to stack
        log_age_max : int or float, optional
            Maximum log(age) to stack


        Returns
        -------
        None
            This method stores the stacked values in the class attributes self.high_H_stacked,
            self.medium_H_stacked, self.low_H_stacked and self.all_stacked.

        """
        self.reset_stack()

        if log_age_min is None and log_age_max is not None:
            log_age_min = self.t[0]
            if log_age_max > self.t[-1]:
                raise HokiFatalError("Age_max too large. Give the log age.")

        if log_age_max is None and log_age_min is not None:
            log_age_max = self.t[-1]
            if log_age_min < self.t[0]:
                raise HokiFatalError("Age_min too low")

        if log_age_min is not None and log_age_max is not None:
            if log_age_min > log_age_max:
                raise HokiFatalError("Age_max should be greater than age_min")

            if log_age_min < np.round(self.t[0], 3) or log_age_max > np.round(self.t[-1], 3):
                raise HokiFatalError("The age range requested is outside the valid range "
                                     "(6.0 to 11.0 inclusive). You requested: "
                                     + str(log_age_min) + " to " + str(log_age_max))

        # Now that we have time limits we calculate what bins they correspond to.
        bin_min, bin_max = int(np.round(10*(log_age_min-6))), int(np.round(10*(log_age_max-6)))

        # And now we slice!
        for hrd1, hrd2, hrd3 in zip(self.high_H[bin_min:bin_max],
                                    self.medium_H[bin_min:bin_max],
                                    self.low_H[bin_min:bin_max]):

            self.high_H_stacked += hrd1
            self.medium_H_stacked += hrd2
            self.low_H_stacked += hrd3

        self.all_stacked = self.high_H_stacked+self.medium_H_stacked+self.low_H_stacked
        print("The following attributes were updated: .all_stacked, .high_H_stacked, "
              ".medium_H_stacked, .low_H_stacked.")
        return

    def at_log_age(self, log_age):
        """
        Returns the HR diagrams at a specific age.

        Parameters
        ----------
        log_age : int or float
            The log(age) of choice.

        Returns
        -------
        Tuple of 4 np.ndarrays (100x100):
            - [0] : Stack of all the abundances
            - [1] : High hydrogen abundance n1>0.4
            - [2] : Medium hydrogen abundance (E-3 < n1 < 0.4)
            - [3] : Low hydrogen abundance (n1 < E-3)

        """

        if log_age < 6.0 or log_age >= 11.1:
            raise HokiFatalError("Valid values of log age should be between 6.0 and 11.1 (inclusive)")

        bin_i = int(np.round(10*(log_age-6)))

        return (self.high_H[bin_i]+self.medium_H[bin_i]+self.low_H[bin_i], self.high_H[bin_i],
                self.medium_H[bin_i], self.low_H[bin_i])

    def plot(self, log_age=None, age_range=None, abundances=(1,1,1), **kwargs):
        """
        Plots the HR Diagram - calls hoki.hrdiagrams.plot_hrdiagram()

        Parameters
        ----------
        log_age : int or float, optional
            Log(age) at which to plot the HRdiagram.

        age_range : tuple or list of 2 ints or floats, optional
            Age range within which you want to plot the HR diagram

        abundances : tuple or list of 3 ints, zeros or ones, optional
            This turns on or off the inclusion of the abundances. The corresponding abundances are:
            (n1 > 0.4, E-3 < n1 < 0.4, E-3>n1). A 1 means a particular abundance should be included,
            a 0 means it will be ignored. Default is (1,1,1), meaning all abundances are plotted.
            Note that (0,0,0) is not valid and will return and assertion error.

        **kwargs : matplotlib keyword arguments, optional

        Notes
        -----
        If you give both an age and an age range, the age range will take precedent and be plotted.
        You will get a warning if that happens though.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot :
            The plot created is returned, so you can add stuff to it, like text or extra data.


        """

        #assert abundances != (0, 0, 0), "HOKI ERROR: abundances cannot be (0, 0, 0) - You're plotting nothing."
        if abundances == (0, 0, 0):
            raise HokiFatalError("Abundances cannot be (0, 0, 0) - You're plotting nothing.")

        if not isinstance(abundances, tuple):
            error_message="abundances should be a tuple of 3 integers - consult the docstrings for further details "
            raise HokiFormatError(error_message)

        hr_plot = None

        # Case were no age or age range are given

        if log_age is None and age_range is None:
            self.stack(BPASS_TIME_BINS[0], BPASS_TIME_BINS[-1])
            all_hr, high_hr, medium_hr, low_hr = self.all_stacked, self.high_H_stacked, \
                                                 self.medium_H_stacked, self.low_H_stacked

            if abundances == (1, 1, 1):
                hr_plot = plot_hrdiagram(all_hr, kind=self.type, **kwargs)
            elif abundances == (1, 1 , 0):
                hr_data = high_hr + medium_hr
                hr_plot = plot_hrdiagram(hr_data, kind=self.type, **kwargs)
            elif abundances == (1, 0, 0):
                hr_plot = plot_hrdiagram(high_hr, kind=self.type, **kwargs)
            elif abundances == (0, 1, 1):
                hr_data = medium_hr+low_hr
                hr_plot = plot_hrdiagram(hr_data, kind=self.type, **kwargs)
            elif abundances == (0, 1, 0):
                hr_plot = plot_hrdiagram(medium_hr, kind=self.type, **kwargs)
            elif abundances == (0, 0, 1):
                hr_plot = plot_hrdiagram(low_hr, kind=self.type, **kwargs)
            elif abundances == (1, 0, 1):
                hr_data = high_hr+low_hr
                hr_plot = plot_hrdiagram(hr_data, kind=self.type, **kwargs)

            hr_plot.set_xlabel("log"+self.type[0])
            hr_plot.set_ylabel("log"+self.type[1:])

            return hr_plot

        elif age_range is not None and log_age is not None:
            error_message = "You provided an age range as well as an age. The former takes " \
                            "precedent. If you wanted to plot a single age, this will be WRONG."
            warnings.warn(error_message, HokiUserWarning)

        # Case where an age or age_range is given

        if age_range is not None:
            if not isinstance(age_range, list) and not isinstance(age_range, tuple):
                raise HokiFormatError("Age range should be a list or a tuple")

            self.stack(age_range[0], age_range[1])
            all_hr, high_hr, medium_hr, low_hr = self.all_stacked, self.high_H_stacked, \
                                                 self.medium_H_stacked, self.low_H_stacked

        elif log_age is not None:
            if not isinstance(log_age, int) and not isinstance(log_age, float):
                raise HokiFormatError("Age should be an int or float")

            all_hr, high_hr, medium_hr, low_hr = self.at_log_age(log_age)



        if abundances == (1, 1, 1):
            hr_plot = plot_hrdiagram(all_hr, kind=self.type, **kwargs)
        elif abundances == (1, 1 , 0):
            hr_data = high_hr + medium_hr
            hr_plot = plot_hrdiagram(hr_data, kind=self.type, **kwargs)
        elif abundances == (1, 0, 0):
            hr_plot = plot_hrdiagram(high_hr, kind=self.type, **kwargs)
        elif abundances == (0, 1, 1):
            hr_data = medium_hr+low_hr
            hr_plot = plot_hrdiagram(hr_data, kind=self.type, **kwargs)
        elif abundances == (0, 1, 0):
            hr_plot = plot_hrdiagram(medium_hr, kind=self.type, **kwargs)
        elif abundances == (0, 0, 1):
            hr_plot = plot_hrdiagram(low_hr, kind=self.type, **kwargs)
        elif abundances == (1, 0, 1):
            hr_data = high_hr+low_hr
            hr_plot = plot_hrdiagram(hr_data, kind=self.type, **kwargs)

        hr_plot.set_xlabel("log"+self.type[0])
        hr_plot.set_ylabel("log"+self.type[1:])

        return hr_plot

    def _apply_time_weighting(self):
        """ Weighs all 51 grids by the number of years in each bin."""

        self.high_H = self.high_H_not_weighted*self.time_weight_grid
        self.medium_H = self.medium_H_not_weighted*self.time_weight_grid
        self.low_H = self.low_H_not_weighted*self.time_weight_grid

    # Now we can index HR diagrams!
    def __getitem__(self, item):
        return self._all_H[item]
        #return self.low_H[item]
        #return self.high_H[item]+self.medium_H[item]


def plot_hrdiagram(single_hr_grid, kind='TL', loc=111, cmap='Greys', **kwargs):
    """
    Plots an HR diagram with a contour plot

    Parameters
    ----------
    single_hr_grid : np.ndarray (100x100)
        One HR diagram grid.

    kind : str, optional
        Type of HR diagram: 'TL', 'Tg', or 'TTG'. Default is 'TL'.

    loc : int - 3 digits, optional
        Location to parse plt.subplot(). The Default is 111, to make only one plot.

    cmap : str, optional
        The matplotlib colour map to use. Default is 'RdGy'.

    kwargs : matplotlib key word arguments to parse


    Note
    -----
    The default levels are defined such that they show the maximum value, then a 10th, then a 100th,
    etc... down to the minimum level. You can also use the "levels" keyword of the contour function
    to choose the number of levels you want (but them matplotlib will arbitrarily define where the
    levels fall).


    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot :
        The plot created is returned, so you can add stuff to it, like text or extra data.

    """

    assert kind == 'TL' or kind == 'Tg' or kind == 'TTG', "Need to write an error message for this"
    if kind == 'TL':
        X, Y = np.meshgrid(np.arange(0.1, 10.1, 0.1), np.arange(-2.9, 7.1, 0.1))
    elif kind == 'Tg':
        X, Y = np.meshgrid(np.arange(0.1, 10.1, 0.1), np.arange(-2.9, 7.1, 0.1))
    elif kind == 'TTG':
        X, Y = np.meshgrid(np.arange(0.1, 10.1, 0.1), np.arange(-2.9, 7.1, 0.1))

    hr_diagram = plt.subplot(loc)

    # MAKING CONTOUR PLOT IN LOG SPACE!
    # This requires some attention: I need to log10 my array, so zero values will be undefined
    # this messed with the contour and contourf matplotlib functions.
    # The work around is to replace the 0 values in the grid by the lowest, non-zero value in the
    # grid. I also chose a default colour map where low values are white, so it doesn't look
    # like I populated the grid.

    # Now we define our default levels
    top_level = single_hr_grid.max()
    min_level = single_hr_grid[single_hr_grid > 1].min()

    # we want our levels to be fractions of 10 of our maximum value
    # and yes it didn't need to be written this way, but isn't it gorgeous?
    possible_levels = [#top_level*0.00000000001,
                       #top_level*0.0000000001,
                       #top_level*0.000000001,
                       top_level*0.00000001, #
                       top_level*0.0000001,
                       top_level*0.000001,
                       top_level*0.00001,
                       top_level*0.0001,
                       top_level*0.001,
                       top_level*0.01,
                       top_level*0.1,
                       top_level]

    # to make sure the colourmap is sensible we want to ensure the minimum level == minimum value
    levels = [min_level] + [level for level in possible_levels if level > min_level]

    colMap = copy.copy(cm.get_cmap(cmap))

    colMap.set_under(color='white')

    # Take the grid and replace zeros by something non-zero but still smaller than lowest value
    single_hr_grid[single_hr_grid == 0] = min(single_hr_grid[single_hr_grid != 0])-\
                                          0.1*min(single_hr_grid[single_hr_grid != 0])

    # I then log the grid and transpose the array directly in the plotting function
    # The transpose is required so that my HR diagram is the right way around.
    CS = hr_diagram.contourf(X, Y, np.log10(single_hr_grid.T), np.log10(levels).tolist(),
                         cmap=cmap, **kwargs)

    # Temperature should be inverted
    hr_diagram.invert_xaxis()


    #plt.colorbar(CS)

    return hr_diagram

