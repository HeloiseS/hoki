from hoki import load
import matplotlib.pyplot as plt
from hoki.constants import *
import numpy as np
import matplotlib.cm as cm


class CMD(object):
    # NOTE: dummy is the name of the big array returned by the BPASS models
    # in the hoki code I use it as a "proper noun" - not a random variable name

    # dummy_col_number=len(dummy_dict) I think this line is no longer useful

    # just for consistency with the HRDiagram
    t = BPASS_TIME_BINS
    dt = BPASS_TIME_INTERVALS

    def __init__(self, file,
                 col_lim=[-3, 7],
                 mag_lim=[-14, 10],
                 res_el=0.1,
                 path=MODELS_PATH):
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
         path : str, optional
             Path to the stellar models. Default is MODEL_PATH with is defined in the constants module.
         """
        self.bpass_input = load.model_input(file)
        self._file_does_not_exist = []

        # Setting up the grid's resolution
        self.col_range = np.arange(col_lim[0], col_lim[1], res_el)
        self.mag_range = np.arange(mag_lim[0], mag_lim[1], res_el)
        self.grid = np.zeros((len(BPASS_TIME_BINS), len(self.mag_range), len(self.col_range)))
        self.path = path
        self._col_bins = None
        self._mag_bins = None
        self._time_bins = None
        self._log_ages = None
        self._ages = None

    def make(self, filter1, filter2):
        """
        Make the CMD - a.k.a fill the grid

        Notes
        ------
            - This may take a few seconds to a minute to run.
            - The colour will be filter1 - filter2

        Parameters
        ----------
        filter1 : str
            First filter
        filter2 : str
            Seconds filter

        Returns
        -------
        None
        """

        # FIND THE KEYS TO THE COLUMNS OF INTEREST IN DUMMY

        col_keys = ['timestep', 'age', str(filter1), str(filter2)]

        try:
            cols = tuple([dummy_dict[key] for key in col_keys])
        except KeyError as e:
            print('Received the following error -- KeyError:', e,
                  '\n----- TROUBLESHOOTING ----- '
                  '\nOne or both of the chosen filters do not correspond to a valid filter key. '
                  'Here is a list of valid filters - input them as string:\n'+str(list(dummy_dict.keys())[49:-23]))
            return

        # LOOPING OVER EACH LINE IN THE INPUT FILE
        for filename,  model_imf, mixed_imf, mixed_age in zip(self.bpass_input.filenames,
                                                              self.bpass_input.model_imf,
                                                              self.bpass_input.mixed_imf,
                                                              self.bpass_input.mixed_age):

            # LOADING THE DATA FILE
            # Making sure it exists - If not keep the name in a list
            try:
                my_data = np.loadtxt(self.path + filename, unpack=True, usecols=cols)
            except (FileNotFoundError, OSError):
                self._file_does_not_exist.append(filename)
                continue

            # MAKING THE COLOUR
            try:
                colours = [filt1 - filt2 for filt1, filt2 in zip(my_data[2], my_data[3])]
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
                              for mag in my_data[3]]

            # MIXED AGE = 0.0 OR NAN CASE (i.e. no rejuvination)
            if np.isnan(mixed_age) or float(mixed_age) == 0.0:
                self._ages = my_data[1]
                self._log_ages = np.log10(my_data[1])
                self._log_ages = [age if age >= 6.0 else 6.0 for age in self._log_ages]
                self._fill_grid_with(model_imf)

            # MIXED AGE NON ZERO CASE (i.e. rejuvination has occured)
            else:
                # MODEL IMF = MIXED IMF (These models only occur after rejuvination)
                if np.isclose(model_imf,mixed_imf):
                    self._ages = my_data[1]+mixed_age
                    self._log_ages = np.log10(my_data[1]+mixed_age)
                    self._fill_grid_with(mixed_imf)

                #  MODEL INF != MIXED IMF (These can occur with or without rejuvination)
                else:
                    # NON REJUVINATED MODELS
                    self._ages = my_data[1]
                    self._log_ages = np.log10(my_data[1])
                    self._log_ages = [age if age >= 6.0 else 6.0 for age in self._log_ages]
                    self._fill_grid_with(model_imf-mixed_imf)

                    # REJUVINATED MODELS
                    self._ages = my_data[1]+mixed_age
                    self._log_ages = np.log10(my_data[1]+mixed_age)
                    self._fill_grid_with(mixed_imf)

    def _fill_grid_with(self, imf):

        for i in range(len(self._ages)):
            # NEED SPECIAL CASES FOR i = 0
            if i == 0:
                self.grid[0, self._mag_bins[0], self._col_bins[0]] += imf * self._ages[0]
                continue

            try:
                N_i_m1 = np.abs(BPASS_TIME_BINS - self._log_ages[i-1]).argmin()
                N_i = np.abs(BPASS_TIME_BINS - self._log_ages[i]).argmin()
            except IndexError:
                print("This should not happen")

            # If the time step within one time bin
            if N_i_m1 == N_i:
                dt_i = self._ages[i] - self._ages[i-1]
                self.grid[N_i, self._mag_bins[i], self._col_bins[i]] += imf * dt_i

            # If the time step spans multiple time bins
            else:
                N_list = np.arange(N_i_m1, N_i+1)

                # First bin
                weight = 10**(BPASS_TIME_BINS[N_list[0]]+0.05) - self._ages[i-1]
                self.grid[N_list[0], self._mag_bins[i], self._col_bins[i]] += imf * weight

                # Last bin
                weight = self._ages[i] - 10**(BPASS_TIME_BINS[N_list[-1]]-0.05)
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

        assert 6.0 <= log_age < 11.1,"FATAL ERROR: Valid values of log age should be between 6.0 and 11.1 (inclusive)"

        single_cmd_grid = self.grid[int(index)]
        single_cmd_grid[single_cmd_grid == 0] = min(single_cmd_grid[single_cmd_grid != 0]) - \
                                                0.1*min(single_cmd_grid[single_cmd_grid != 0])

        top_level = single_cmd_grid.max()
        min_level = single_cmd_grid.min()

        # we want our levels to be fractions of 10 of our maximum value
        # and yes it didn't need to be written this way, but isn't it gorgeous?
        possible_levels = [# top_level*0.0000000000001,
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

        return cm_diagram

