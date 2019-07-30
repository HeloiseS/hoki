"""
Dr. H. F. Stevance - hfstevance@gmail.com
The University of Auckland



"""

import numpy as np
import matplotlib.pyplot as plt

BPASS_TIME_BINS = np.arange(6, 11.1, 0.1)
BPASS_TIME_INTERVALS = np.array([10**(t+0.05) - 10**(t-0.05) for t in BPASS_TIME_BINS])
BPASS_TIME_WEIGHT_GRID = np.array([np.zeros((100,100)) + dt for dt in BPASS_TIME_INTERVALS])


class HRDiagram(object):
    """
    Need a Doc String!
    """
    # TODO: write documentation
    # TODO: WRITE TESTS!

    t = BPASS_TIME_BINS
    dt = BPASS_TIME_INTERVALS
    _time_weights = BPASS_TIME_WEIGHT_GRID
    logg_bins = np.arange(-2.9, 7.1, 0.1)
    logTG_bins = np.arange(-2.9, 7.1, 0.1)

    def __init__(self, high_H_input, medium_H_input, low_H_input):

        # Initialise core attributes

        self.high_H_not_weighted = high_H_input
        self.medium_H_not_weighted = medium_H_input
        self.low_H_not_weighted = low_H_input

        self._apply_time_weighting()
        self._all_H = self.low_H + self.medium_H + self.high_H

        # Initialise attributes for later

        self.high_H_stacked, self.medium_H_stacked, self.low_H_stacked = np.zeros((100, 100)), \
                                                                         np.zeros((100, 100)),\
                                                                         np.zeros((100, 100))
        self.all_stacked = None

    def stack(self, age_min=None, age_max=None):
        # TODO: Finish the documentation (if we even keep this)
        """
        Creates a stack of HR diagrams within a range of ages

        Parameters
        ----------
        age_min
        age_max

        Returns
        -------

        """

        # Just making sure the limits given make sense
        if age_min is not None and age_max is None:
            assert age_min < self.t[-1], "FATAL ERROR: age_min should be smaller than maximum age"
        elif age_max is not None and age_min is None:
            assert age_max > self.t[0], "FATAL ERROR: age_max should be grater than the minimum age"

        # Detecting whether the limits were given in log space or in years
        if age_min is None:
            age_min_log = self.t[0]

        if age_max is None:
            age_max_log = self.t[-1]+0.1

        if age_min is not None and age_max is not None:
            assert age_min < age_max, "FATAL ERROR: age_max should be greater than age_min"

            if age_min >= BPASS_TIME_BINS[0] and age_max <= BPASS_TIME_BINS[-1]:
                age_min_log, age_max_log = age_min, age_max

            elif age_min > 999999 and age_max > 999999:
                print("WARNING: It looks like you gave me the time interval in years, "
                      "I'll convert to logs")
                age_min_log, age_max_log = np.log10(age_min), np.log10(age_max)

            else:
                assert age_min >= BPASS_TIME_BINS[0] and age_max <= BPASS_TIME_BINS[-1], \
                    "FATAL ERROR: The age range requested is outside the valid range " \
                    "(6.0 to 11.1 inclusive)"

        # Now that we have time limits we calculate what bins they correspond to.
        bin_min, bin_max = int(np.round(10*(age_min_log-6))), int(np.round(10*(age_max_log-6)))

        # And now we slice!
        for hrd1, hrd2, hrd3 in zip(self.high_H[bin_min:bin_max],
                                    self.medium_H[bin_min:bin_max],
                                    self.low_H[bin_min:bin_max]):

            self.high_H_stacked += hrd1
            self.medium_H_stacked += hrd2
            self.low_H_stacked += hrd3

        self.all_stacked = self.high_H_stacked+self.medium_H_stacked+self.low_H_stacked

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
            - [1] : High hydrogen abundance X>0.4
            - [2] : Medium hydrogen abundance (E-3 < X < 0.4)
            - [3] : Low hydrogen abundance (X < E-3)

        """
        assert log_age >= 6.0 and log_age <= 11.1, \
            "FATAL ERROR: Valid values of log age should be between 6.0 and 11.1 (inclusive)"

        bin_i = int(np.round(10*(log_age-6)))

        return (self.high_H[bin_i]+self.medium_H[bin_i]+self.low_H[bin_i], self.high_H[bin_i],
                self.medium_H[bin_i], self.low_H[bin_i])

    def plot(self, log_age=None, age_range=None, kind='TL', abundances=(1,1,1), **kwargs):
        """
        Plots the HR Diagram - calls hoki.hrdiagrams.plot_hrdiagram()

        Parameters
        ----------
        log_age
        age_range
        abundances
        kwargs

        Returns
        -------

        """
        assert abundances != (0, 0, 0), "abundances cannot be (0, 0, 0) - You're plotting nothing."
        #TODO TEST EVERYTHING
        hr_plot = None

        # Case were no age or age range are given

        if log_age is None and age_range is None:
            self.stack(BPASS_TIME_BINS[0], BPASS_TIME_BINS[-1])
            all_hr, high_hr, medium_hr, low_hr = self.all_stacked, self.high_H_stacked, \
                                                 self.medium_H_stacked, self.low_H_stacked

            if abundances == (1, 1, 1):
                hr_plot = plot_hrdiagram(all_hr, kind=kind, **kwargs)
            elif abundances == (1, 1 , 0):
                hr_data = high_hr + medium_hr
                hr_plot = plot_hrdiagram(hr_data, kind=kind, **kwargs)
            elif abundances == (1, 0, 0):
                hr_plot = plot_hrdiagram(high_hr, kind=kind, **kwargs)
            elif abundances == (0, 1, 1):
                hr_data = medium_hr+low_hr
                hr_plot = plot_hrdiagram(hr_data, kind=kind, **kwargs)
            elif abundances == (0, 1, 0):
                hr_plot = plot_hrdiagram(medium_hr, kind=kind, **kwargs)
            elif abundances == (0, 0, 1):
                hr_plot = plot_hrdiagram(low_hr, kind=kind, **kwargs)
            elif abundances == (1, 0, 1):
                hr_data = high_hr+low_hr
                hr_plot = plot_hrdiagram(hr_data, kind=kind, **kwargs)

            hr_plot.set_xlabel("log"+kind[0])
            hr_plot.set_ylabel("log"+kind[:-1])

            return hr_plot

        # Case where an age or age_range is given

        if log_age:
            assert isinstance(log_age, int) or isinstance(log_age, float), \
                "Age should be an int or float"

            all_hr, high_hr, medium_hr, low_hr = self.at_log_age(log_age)

        elif age_range:
            assert isinstance(age_range, list) or isinstance(age_range, tuple), \
                "Age range should be a list or a tuple"

            self.stack(age_range[0], age_range[1])
            all_hr, high_hr, medium_hr, low_hr = self.all_stacked, self.high_H_stacked, \
                                                 self.medium_H_stacked, self.low_H_stacked

        elif age_range and log_age:
            print("\nWARNING: you provided an age range as well as an age. The latter takes "
                  "precedent. If you wanted to plot a single age, this will be WRONG.")

        if abundances == (1, 1, 1):
            hr_plot = plot_hrdiagram(all_hr, kind='TL', **kwargs)
        elif abundances == (1, 1 , 0):
            hr_data = high_hr + medium_hr
            hr_plot = plot_hrdiagram(hr_data, kind='TL', **kwargs)
        elif abundances == (1, 0, 0):
            hr_plot = plot_hrdiagram(high_hr, kind='TL', **kwargs)
        elif abundances == (0, 1, 1):
            hr_data = medium_hr+low_hr
            hr_plot = plot_hrdiagram(hr_data, kind='TL', **kwargs)
        elif abundances == (0, 1, 0):
            hr_plot = plot_hrdiagram(medium_hr, kind='TL', **kwargs)
        elif abundances == (0, 0, 1):
            hr_plot = plot_hrdiagram(low_hr, kind='TL', **kwargs)
        elif abundances == (1, 0, 1):
            hr_data = high_hr+low_hr
            hr_plot = plot_hrdiagram(hr_data, kind='TL', **kwargs)

        hr_plot.set_xlabel("log"+kind[0])
        hr_plot.set_ylabel("log"+kind[1:])

        return hr_plot

    def _apply_time_weighting(self):
        """ Weighs all 51 grids by the number of years in each bin."""

        self.high_H = self.high_H_not_weighted*self._time_weights
        self.medium_H = self.medium_H_not_weighted*self._time_weights
        self.low_H = self.low_H_not_weighted*self._time_weights

    # Now we can index HR diagrams!
    def __getitem__(self, item):
        return self._all_H[item]


#  # The following functions find the index of the bin corresponding to a log value
#  def _T_index(self, log_T):
#      return int(np.round((log_T-self.logT_bins[0])/0.1))
#
# def _g_index(self, log_g):
#      return int(np.round((log_g-self.logg_bins[0])/0.1))
#
#  def _TG_index(self, log_TG):
#      return int(np.round((log_TG-self.logTG_bins[0])/0.1))
#  def _T_index(self, log_T):
#      return int(np.round((log_T-self.logT_bins[0])/0.1))
#
#  def _L_index(self, log_L):
#      return int(np.round((log_L-self.logL_bins[0])/0.1))


def plot_hrdiagram(single_hr_grid, kind='TL', levels=10,loc=111, cmap='RdGy', **kwargs):
    """

    Parameters
    ----------
    single_hr_grid
    kind
    levels
    loc
    cmap
    kwargs

    Returns
    -------

    """
    #TODO: Write this docstring!

    assert kind == 'TL' or kind == 'Tg' or kind == 'TTG', "Need to write an error message for this"
    if kind == 'TL':
        X, Y = np.meshgrid(np.arange(0.1, 10.1, 0.1), np.arange(-2.9, 7.1, 0.1))
    elif kind == 'Tg':
        X, Y = np.meshgrid(np.arange(-2.9, 7.1, 0.1), np.arange(-2.9, 7.1, 0.1))
    elif kind == 'TTG':
        X, Y = np.meshgrid(np.arange(-2.9, 7.1, 0.1), np.arange(-2.9, 7.1, 0.1))

    hr_diagram = plt.subplot(loc)
    hr_diagram.contour(X, Y, single_hr_grid.T, levels, cmap=cmap, **kwargs)

    return hr_diagram

