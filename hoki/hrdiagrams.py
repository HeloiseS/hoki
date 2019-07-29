import numpy as np

BPASS_TIME_BINS = np.arange(6, 11.1, 0.1)
BPASS_TIME_INTERVALS = np.array([10**(t+0.05) - 10**(t-0.05) for t in BPASS_TIME_BINS])
BPASS_TIME_WEIGHT_GRID = np.array([np.zeros((100,100)) + dt for dt in BPASS_TIME_INTERVALS])


class HRdiagram(object):
    t = BPASS_TIME_BINS
    dt = BPASS_TIME_INTERVALS
    _time_weights = BPASS_TIME_WEIGHT_GRID

    def __init__(self, high_H_input, medium_H_input, low_H_input):

        # Initialise core attributes

        self.high_H_not_weighted = high_H_input
        self.medium_H_not_weighted = medium_H_input
        self.low_H_not_weighted = low_H_input

        self._apply_time_weighting()
        self._all_H = self.low_H + self.medium_H + self.high_H

        # Initialise attributes for later

        self.high_H_stacked, self.medium_H_stacked, self.low_H_stacked = np.zeros((100,100)), \
                                                                         np.zeros((100,100)),\
                                                                         np.zeros((100,100))
        self.all_stacked = None

    def stack(self, age_min=None, age_max=None):

        # Just making sure the limits given make sense
        if age_min is not None and age_max is None:
            assert age_min < self.t[-1], "age_min should be smaller than maximum age"
        elif age_max is not None and age_min is None:
            assert age_max > self.t[0], "age_max should be grater than the minimum age"


        # Detecting whether the limits were given in log space or in years
        if age_min is None:
            age_min_log = self.t[0]

        if age_max is None:
            age_max_log = self.t[-1]+0.1

        if age_min is not None and age_max is not None:
            assert age_min < age_max, "age_max should be greater than age_min"

            if age_min >= 6 and age_max <= 11.1:
                # since we've already checked that age_max > age_min this is enough to ensure a correct range
                age_min_log, age_max_log = age_min, age_max

            elif age_min > 999999 and age_max > 999999:
                print("It looks like you gave me the time interval in years, I'll convert to logs")
                age_min_log, age_max_log = np.log10(age_min), np.log10(age_max)


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
        bin_i = int(np.round(10*(log_age-6)))

        return (self.high_H[bin_i]+self.medium_H[bin_i]+self.low_H[bin_i], self.high_H[bin_i],
                self.medium_H[bin_i], self.low_H[bin_i])

    def _apply_time_weighting(self):
        """ Weighs all 51 grids by the number of years in each bin."""

        self.high_H = self.high_H_not_weighted*self._time_weights
        self.medium_H = self.medium_H_not_weighted*self._time_weights
        self.low_H = self.low_H_not_weighted*self._time_weights

    def __getitem__(self, item):
        return self._all_H[item]
