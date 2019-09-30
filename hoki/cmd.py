from hoki import load
import pandas as pd
import matplotlib.pyplot as plt
from hoki.constants import *
import numpy as np
import matplotlib.cm as cm


# TODO: 1) Review with JJ the probas imfs I'm putting into the grid
# TODO: 2) I logged the colour map which looks alright - Do I need to choose specific levels like with hrdiagrams?
# TODO: 3) Sometimes the stellar model files contain only one row - so only the first time step.
#          This breaks the code and contains unphysical quantities - I have not taken these cases into account in CMDs
# TODO: 4) Review with JJ the binning method - see page 62 in my log book
# TODO: 5) Do I need to create a dedicated function feature that can plot stacked CMD ages or is that a rare occurance?


# TODO: 6) REVIEW WITH JJ - Interpolation for time bins that are not given in stellar model files.



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

        # self.mask = [False]*self.dummy_col_number

        self.file_does_not_exist = []

        # Setting up the grid's resolution
        self.col_range = np.arange(col_lim[0], col_lim[1], res_el)
        self.mag_range = np.arange(mag_lim[0], mag_lim[1], res_el)
        self.grid = np.zeros((len(BPASS_TIME_BINS), len(self.mag_range), len(self.col_range)))
        self.path = path

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

        col_keys = ['timestep', 'age', str(filter1), str(filter2)]

        try:
            cols = tuple([dummy_dict[key] for key in col_keys])
        except KeyError as e:
            print('Received the following error -- KeyError:', e,
                  '\n----- TROUBLESHOOTING ----- '
                  '\nOne or both of the chosen filters do not correspond to a valid filter key. '
                  'Here is a list of valid filters - input them as string:\n'+str(list(dummy_dict.keys())[49:-23]))
            return

        for filename, modeltype, model_imf, mixed_imf in zip(self.bpass_input.filenames,
                                                             self.bpass_input.types,
                                                             self.bpass_input.imfs_proba,
                                                             self.bpass_input.mixed_imf):

            # Loading the file and making sure it exists - If not keep the name in a list
            try:
                my_data = np.loadtxt(self.path + filename, unpack=True, usecols=cols)
            except (FileNotFoundError, OSError):
                self.file_does_not_exist.append(filename)
                continue

            # Sometimes there is only one row - i.e. the star did not evolve.
            # Then the zip will fail - I skip it because I don't care about stars that do nothing?
            try:
                colours = [filt1 - filt2 for filt1, filt2 in zip(my_data[2], my_data[3])]
            except TypeError:
                continue

            # Once we know the file exists and there is data to use, we check the model type
            # This tells us what imf probas to put in our grid.
            if modeltype < 2:
                imf = model_imf
            else:
                imf = model_imf + mixed_imf

            # we turn the ages (given in years) into log_ages to compare to the BPASS_TIME_BINS
            log_ages = np.log10(my_data[1])
            log_ages = [age if age >= 6.0 else 6.0 for age in log_ages]
            # for all intended pruposes this is the age bing that lower ages will end up in

            # List comprehension hell to figure out the indices of the bins I need to fill
            time_index = [np.abs((BPASS_TIME_BINS - log_age)).argmin()
                          if BPASS_TIME_BINS[np.abs((BPASS_TIME_BINS - log_age)).argmin()] <= log_age
                          else np.abs((BPASS_TIME_BINS - log_age)).argmin() - 1
                          for log_age in log_ages]


            col_index = [np.abs((self.col_range - c)).argmin()
                         if self.col_range[np.abs((self.col_range - c)).argmin()] <= c
                         else np.abs((self.col_range - c)).argmin() - 1
                         for c in colours]

            mag_index = [np.abs((self.mag_range - mag)).argmin()
                         if self.mag_range[np.abs((self.mag_range - mag)).argmin()] <= mag
                         else np.abs((self.mag_range - mag)).argmin() - 1
                         for mag in my_data[3]]

            # And now I fill the right grid bin with the imf proba we calculated earlier
            for mag_i, col_i, t_i in zip(mag_index, col_index, time_index):
                self.grid[t_i, mag_i, col_i] += imf


            # Now some age bins (range form 0 to 51) are not populated - we have to do this to
            # 1) we find the empty time bins, and while we're at it the non-empty ones
            empty_time_bins = np.array([i for i in range(0,51) if i not in time_index])
            not_empty_time_bins = np.array([i for i in range(0,51) if i in time_index])


            # Then for each empty time bin we find the nearest non-empty one
            nearest_not_empty_bins = [np.abs(not_empty_time_bins - t).argmin()
                                      for t in empty_time_bins]

            # Then for each empty time bin and corresponding nearest time bin
            # we find the position of the LAST value in the time_index that == nearest time bin
            # this position is also the position of the indices in col_index and mag_index
            # that we are going to use to population the grid, alongside the empty_time_bin
            # (the missing time index).
            # My brain is metling out of my ears.

            # """

            for empty, nearest in zip(empty_time_bins, nearest_not_empty_bins):
                index_all_bins_at_nearest_age = np.where((time_index - nearest) == 0)[0]

                try:
                    # this break if there is only one value
                    index_last_bin_at_nearest_age = index_all_bins_at_nearest_age[-1]
                except IndexError:
                    self.grid[empty, mag_index[nearest], col_index[nearest]] += imf
                    #print(empty, nearest, time_index[nearest])
                    continue

                i = index_last_bin_at_nearest_age
                self.grid[empty, mag_index[i], col_index[i]] += imf
            # """

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

        colMap = cm.get_cmap(cmap)

        colMap.set_under(color='white')

        index = np.where(np.round(BPASS_TIME_BINS,1) == log_age)[0]

        assert 6.0 <= log_age <= 11.1, \
            "FATAL ERROR: Valid values of log age should be between 6.0 and 11.1 (inclusive)"

        toplot = self.grid[int(index)]
        toplot[toplot == 0] = min(toplot[toplot != 0])- 0.1*min(toplot[toplot != 0])

        cm_diagram.contourf(self.col_range, self.mag_range, np.log10(toplot), cmap=cmap, **kwargs)

        cm_diagram.invert_yaxis()

        return cm_diagram


