"""
This module implements the CMD infrastructure
"""

from hoki import load
import matplotlib.pyplot as plt
from hoki.constants import *
import numpy as np
import matplotlib.cm as cm
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiKeyError
from hoki.utils.hoki_object import HokiObject

__all__ = ['CMD']


class CMD(HokiObject):
    """
    **Colour Magnitude Diagram Object**

    Parameters
    ----------
    file : str
        Location of the file containing the model inputs
    col_lim : list of 2 integers (positive or negative), optional
        Limits on the colour range of the CMD grid, Default is [-3,7].
    mag_lim : list of 2 integers (positive or negative), optional
        Limits on the magnitude range of the CMD grid. Default is [-14,10].
    res_el : float or int, optional
        Resolution element of the CMD grid. The resolution element is the same for colour and magnitude.
        Default is 0.1.
    bpass_version : str, optional (v221 or v222)
        The BPASS version to consider - this changes the "dummy_dictionary" that is loaded from the settings.yaml file
        Default = hoki.constants.DEFAULT_BPASS_VERSION

    Attributes
    ----------
    self.bpass_input : str
        Input file given by the `file` parameter
    self.col_range : numpy.ndarray (1D)
        Colour range spanned by the grid (with `res_el`-sized intervals)
    self.mag_range : numpy.ndarray (1D)
        Magnitude range spanned by the grid (with `res_el`-sized intervals)
    self.grid : numpy.ndarray (2D)
        Colour-Magnitude grid.
    self.path : str
        The absolute path to the stellar models. It is set to `hoki.cconstants.MODELS_PATH` which you can set to
        the right path by using `hoki.load.set_models.path()`.
    self.t : np.ndarray 1D
        **Class attribute** - The time bins in BPASS - note they are in LOG SPACE
    self.dt : np.ndarray 1D
        **Class attribute** - Time intervals between bins NOT in log space

    """
    # NOTE: dummy is the name of the big array returned by the BPASS models
    # in the hoki code I use it as a "proper noun" - not a random variable name

    # dummy_col_number=len(dummy_dict) I think this line is no longer useful

    def __init__(self, file,
                 col_lim=[-3, 7],
                 mag_lim=[-14, 10],
                 res_el=0.1,
                 bpass_version=DEFAULT_BPASS_VERSION):
        """
         Initialisation of the Colour Magnitude Diagram object

         Parameters
         ----------
         file : str
             Location of the file containing the model inputs
         col_lim : list of 2 integers (positive or negative), optional
             Limits on the colour range of the CMD grid, Default is [-3,7].
         mag_lim : list of 2 integers (positive or negative), optional
             Limits on the magnitude range of the CMD grid. Default is [-14,10].
         res_el : float or int, optional
             Resolution element of the CMD grid. The resolution element is the same for colour and magnitude.
             Default is 0.1.
         """
        self.bpass_input = load.model_input(file)
        self._file_does_not_exist = []
        self.dummy_dict=dummy_dicts[bpass_version]

        # Setting up the grid's resolution
        self.col_range = np.arange(col_lim[0], col_lim[1], res_el)
        self.mag_range = np.arange(mag_lim[0], mag_lim[1], res_el)
        self.grid = None
        self.path = MODELS_PATH
        self._my_data = None
        self._col_bins = None
        self._mag_bins = None
        self._time_bins = None
        self._log_ages = None
        self._ages = None
        self.mag_filter = None
        self.col_filter1 = None
        self.col_filter2 = None

    def make(self, mag_filter, col_filters):
        """
        Make the CMD - a.k.a fill the grid

        Notes
        ------
            - This may take a few seconds to a minute to run.
            - The colour will be mag_filter - filter2
            - If you later call CMD.plot() you will obtain a contour plot of mag_filter against mag_filter-filter2

        Parameters
        ----------
        mag_filter : str
            Magnitude Axis Filter
        col_filters : list of 2 str
            The two filters for the colour. The colours will be col_filter[0]-col_filter[1].

        Returns
        -------
        None
        """
        self.grid = np.zeros((len(BPASS_TIME_BINS), len(self.mag_range), len(self.col_range)))


        self.mag_filter = str(mag_filter)
        # self.filter2 = str(filter2)

        # check list and that list is len(2)
        try:
            self.col_filter1, self.col_filter2 = str(col_filters[0]), str(col_filters[1])
        except TypeError as e:
            err_m='Python said: '+str(e)+'\nDEBUGGING ASSISTANT: col_filter must be a list or tuple of 2 strings'
            raise HokiFormatError(err_m)


        # FIND THE KEYS TO THE COLUMNS OF INTEREST IN DUMMY

        col_keys = ['timestep', 'age', self.col_filter1, self.col_filter2, 'M1', 'log(R1)', 'log(L1)']

        try:
            cols = tuple([self.dummy_dict[key] for key in col_keys])
        except KeyError as e:
            err_m='Python said: '+str(e)+'\nDEBUGGING ASSISTANT: \nOne or both of the chosen filters do not correspond ' \
                                         'to a valid filter key. Here is a list of valid filters - ' \
                                         'input them as string:\n'+str(list(self.dummy_dict.keys())[49:-23])
            raise HokiKeyError(err_m)


        # LOOPING OVER EACH LINE IN THE INPUT FILE
        for filename,  model_imf, mixed_imf, mixed_age, model_type in zip(self.bpass_input.filenames,
                                                                          self.bpass_input.model_imf,
                                                                          self.bpass_input.mixed_imf,
                                                                          self.bpass_input.mixed_age,
                                                                          self.bpass_input.types):

            # PRE PROCESSING THE MODEL INPUTS

            # The model_imf and mixed_imf have different precisions because of the FORTRAN code
            # model_imf is double precision but mixed_imf is double precision. We round to take care of that.
            model_imf = round(model_imf, 6)
            mixed_imf = round(mixed_imf, 6)

            # some mixed ages come out negative - it should be taken into account by setting them to 0
            if mixed_age == np.inf:
                mixed_age = 0

            # LOADING THE DATA FILE
            # Making sure it exists - If not keep the name in a list
            try:
                self._my_data = np.loadtxt(self.path + filename, unpack=True, usecols=cols)
            except (FileNotFoundError, OSError):
                self._file_does_not_exist.append(filename)
                continue

            # MAKING THE COLOUR
            try:
                colours = [filt1 - filt2 for filt1, filt2 in zip(self._my_data[2], self._my_data[3])]
            except TypeError:
                # Sometimes there is only one row - i.e. the star did not evolve.
                # Then the zip will fail - These are stars that have not evolved and there is
                # very few of them so we are skipping them for now.
                continue

            # LIST WHICH BINS IN THE GRID EACH COLOUR AND MAGNITUDE BELONGS TO
            self._col_bins = [np.abs((self.col_range - c)).argmin()
                              if self.col_range[np.abs((self.col_range - c)).argmin()] <= c
                              else np.abs((self.col_range - c)).argmin() - 1
                              for c in colours]

            self._mag_bins = [np.abs((self.mag_range - mag)).argmin()
                              if self.mag_range[np.abs((self.mag_range - mag)).argmin()] <= mag
                              else np.abs((self.mag_range - mag)).argmin() - 1
                              for mag in self._my_data[3]]

            # MIXED AGE = 0.0 OR NAN CASE (i.e. no rejuvination)
            if np.isnan(mixed_age) or float(mixed_age) == 0.0:
                self._ages = self._my_data[1]
                # first line is always zero and will mess up the log so we take care of that
                self._log_ages = np.concatenate((np.array([0]), np.log10(self._my_data[1,1:])))
                self._log_ages = [age if age >= 6.0 else 6.0 for age in self._log_ages]
                self._fill_grid_with(model_imf, model_type)

            # MIXED AGE NON ZERO CASE (i.e. rejuvination has occured)
            else:
                # MODEL IMF = MIXED IMF (These models only occur after rejuvination)
                if np.isclose(model_imf,mixed_imf, atol=1e-05):
                    self._ages = self._my_data[1] + mixed_age
                    self._log_ages = np.log10(self._my_data[1] + mixed_age)
                    self._fill_grid_with(mixed_imf, model_type)

                #  MODEL INF != MIXED IMF (These can occur with or without rejuvination)
                else:
                    # NON REJUVINATED MODELS
                    self._ages = self._my_data[1]
                    self._log_ages = np.concatenate((np.array([0]), np.log10(self._my_data[1,1:])))
                    self._log_ages = [age if age >= 6.0 else 6.0 for age in self._log_ages]
                    self._fill_grid_with(model_imf-mixed_imf, model_type)

                    # REJUVINATED MODELS
                    self._ages = self._my_data[1] + mixed_age
                    self._log_ages = np.log10(self._my_data[1] + mixed_age)
                    self._fill_grid_with(mixed_imf, model_type)

    def _fill_grid_with(self, imf, model_type):

        for i, M, R, L in zip(range(len(self._ages)), self._my_data[4], self._my_data[5], self._my_data[6]):
            # Weird stuff happens to the primary models when they because WD and they can get counted twice
            # Here we helpout BPASS a tad by checking when a primary (type 1) has got the gravity, mass and luminosity
            # of a WD and we jsut remove it from this round so they don't linger on the MS.
            if round(model_type, 1) == 1:
                log_g = np.log10( (6.67259*10**(-8)) * (1.989*10**33) * M /
                                  (((10**R) *6.9598*(10**10))**2) )

                try:
                    if log_g > 6.9 and L < -1 and M < 1.5:
                        # In this case our primary has become a white dwarf and
                        # we need to take it out of the simulations
                        return
                except ValueError as e:
                    print(e)
                    return

            # NEED SPECIAL CASES FOR i = 0
            if i == 0:
                # First line isn't really a bin
                # self.grid[0, self._mag_bins[0], self._col_bins[0]] += imf * self._ages[0]
                continue

            try:
                N_i_m1 = np.abs(BPASS_TIME_BINS - self._log_ages[i-1]).argmin()
                N_i = np.abs(BPASS_TIME_BINS - self._log_ages[i]).argmin()
            except IndexError:
                print("This should not happen")

            # If the time step within one time bin
            if N_i_m1 == N_i:
                dt_i = self._ages[i] - self._ages[i-1]
                # Some time bins can be negative in BPASS and should be ignored
                if dt_i <0: continue

                self.grid[N_i, self._mag_bins[i], self._col_bins[i]] += imf * dt_i

            # If the time step spans multiple time bins
            else:
                N_list = np.arange(N_i_m1, N_i+1)

                # First bin
                weight = 10**(BPASS_TIME_BINS[N_list[0]]+0.05) - self._ages[i-1]
                # Negative time bins should be ignored
                if weight < 0: continue
                self.grid[N_list[0], self._mag_bins[i], self._col_bins[i]] += imf * weight

                # Last bin
                weight = self._ages[i] - 10**(BPASS_TIME_BINS[N_list[-1]]-0.05)
                # Negative time bins should be ignored.
                if weight <0: continue
                self.grid[N_list[-1], self._mag_bins[i], self._col_bins[i]] += imf * weight

                # Bins in between, if any
                if len(N_list)>2:
                    for N in N_list[1:-1]:
                        weight = BPASS_TIME_INTERVALS[N]
                        self.grid[N, self._mag_bins[i], self._col_bins[i]] += imf * weight

    def plot(self, log_age=6.8, loc=111, cmap='Greys', **kwargs):
        """
        Plots the CMD grid at a particular age

        Parameters
        ----------
        log_age : float
            Must be a valid BPASS time bin
        loc : 3 integers, optional
            Location of the subplot. Default is 111.
        cmap : str, optional
            Colour map for the contours. Default is 'Greys'
         **kwargs : matplotlib keyword arguments, optional

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot :
            The plot created is returned, so you can add stuff to it, like text or extra data.

        """
        cm_diagram = plt.subplot(loc)

        #  THIS IS VERY SIMILAR TO THE PLOTTING FUNCTION IN HOKI.HRDIAGRAMS.

        #  Now we define our default levels
        index = np.where(np.round(BPASS_TIME_BINS,1) == log_age)[0]

        if log_age < 6.0 or log_age > 11.0:
            raise HokiFatalError("Valid values of log age should be between 6.0 and 11.1 (inclusive)")

        single_cmd_grid = self.grid[int(index)]

        # This step has been approved by JJ :o)
        infinities = np.where(single_cmd_grid == np.inf)
        for i in infinities: single_cmd_grid[i] = 0.0

        np.nan_to_num(single_cmd_grid, copy=False)

        single_cmd_grid[single_cmd_grid == 0] = min(single_cmd_grid[single_cmd_grid != 0]) - \
                                                0.1*min(single_cmd_grid[single_cmd_grid != 0])

        top_level = single_cmd_grid.max()
        min_level = single_cmd_grid.min()

        # we want our levels to be fractions of 10 of our maximum value
        # and yes it didn't need to be written this way, but isn't it gorgeous?
        possible_levels = [top_level*0.0000000000001,
                           top_level*0.000000000001,
                           top_level*0.00000000001,
                           top_level*0.0000000001,
                           top_level*0.000000001,
                           top_level*0.00000001,
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

        colMap = cm.get_cmap(cmap)

        colMap.set_under(color='white')

        cm_diagram.contourf(self.col_range, self.mag_range, np.log10(single_cmd_grid), np.log10(levels).tolist(),
                            cmap=cmap, **kwargs)

        cm_diagram.invert_yaxis()

        cm_diagram.set_ylabel(self.mag_filter)
        cm_diagram.set_xlabel(self.col_filter1+ "-" + self.col_filter2)

        return cm_diagram

    def at_log_age(self, log_age):
        """
        Returns the HR diagrams at a specific age.

        Parameters
        ----------
        log_age : int or float
            The log(age) of choice.

        Returns
        -------
        The CMD grid : np.ndarray 240x100

        """

        if log_age < 6.0 or log_age > 11.0:
            raise HokiFatalError("Valid values of log age should be between 6.0 and 11.1 (inclusive)")

        bin_i = int(np.round(10*(log_age-6)))
        return self.grid[bin_i]

    def __getitem__(self, item):
        return self.grid[item]


