from hoki import load
import pandas as pd
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
                 col_lim = [-3, 7],
                 mag_lim = [-14, 10],
                 res_el = 0.05):

        self.bpass_input = load.model_input(file)

        # self.mask = [False]*self.dummy_col_number

        self.file_does_not_exist = []

        # Setting up the grid's resolution
        self.col_range = np.arange(col_lim[0], col_lim[1], res_el)
        self.mag_range = np.arange(mag_lim[0], mag_lim[1], res_el)
        self.grid = np.zeros((len(BPASS_TIME_BINS), len(self.mag_range), len(self.col_range)))

    def make(self, filter1, filter2):

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
                my_data = np.loadtxt(MODELS_PATH + filename, unpack=True, usecols=cols)
            except (FileNotFoundError, OSError):
                self.file_does_not_exist.append(filename)
                continue

            # Sometimes there is only one row - i.e. the star did not evolve.
            # Then the zip will fail - I skip it because I don't care about stars that do nothing
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
            log_ages[0] = 6.0  # because log10(0) fails

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

    def plot(self, log_age=6.8, loc=111, cmap='Greys', **kwargs):
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
