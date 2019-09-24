from hoki import load
import pandas as pd
import matplotlib.pyplot as plt
from hoki.constants import *
import numpy as np


class CMD(object):
    # dummy is the name of the big array returned by the BPASS models
    # in the hoki code I use it as a proper noun - not a random variable name
    dummy_col_number=96

    def __init__(self, file='/home/fste075/BPASS_hoki_dev/bpass_v2.2.1_imf135_300/input_bpass_z020_bin_imf135_300'):
        self.bpass_input = load.model_input(file)
        self.mask=[False]*self.dummy_col_number

        self.file_does_not_exist = []

        #coordinates
        #TODO: need to be able to change the resolution and the range on initialisation
        self.col_range = np.arange(-10,10, 0.1)
        self.mag_range = np.arange(-10,10, 0.1)

        # Make the grid we gonna fill
        self.stellar_pop_cmd = np.zeros((len(BPASS_TIME_BINS),  len(self.mag_range), len(self.col_range)))

    def make(self, filter1, filter2):
    # break up this function once it works

        try:
            col_keys = ['timestep','age', str(filter1), str(filter2)]
        except KeyError as e:
            print('Received the following error: ', e,
                  'TROUBLESHOOTING: \nOne or both of the chosen filters do not correspond to a valid filter key. '
                  'Tell people where to find the valid keys'
                  )

        for col_key in col_keys:
            self.mask[dummy_dict[col_key]]=True

        for filename, modeltype, not_mixed, mixed in zip(self.bpass_input.filenames,
                                                         self.bpass_input.types,
                                                         self.bpass_input.imfs_proba,
                                                         self.bpass_input.mixed_imf):

            if modeltype < 2:
                imf = not_mixed
            else:
                imf = not_mixed + mixed

            try:
                try:
                    data_np = np.genfromtxt(MODELS_PATH + filename, unpack = True)
                    my_data = data_np[np.arange(0, self.dummy_col_number, 1)[self.mask], :]
                except IndexError:
                    # if only one line file this will fail
                    # data_np = np.loadtxt(MODELS_PATH + filename, unpack =True, ndmin=1)
                    # my_data = data_np[np.arange(0, self.dummy_col_number, 1)[self.mask]]
                    # now i get an error because can't zip the 1d array fuck me. ignore this for noe
                    continue

            except (FileNotFoundError, OSError):
                self.file_does_not_exist.append(filename)
                continue

            print(data_np.shape, len(self.mask), filename)

            colours = [filt1 - filt2 for filt1, filt2 in zip(my_data[2], my_data[3])]

            log_ages = np.log10(my_data[1])
            log_ages[0] = 6.0

            # REFACTOR VARIABLE NAMES
            which_time_bin = [np.abs((BPASS_TIME_BINS - log_age)).argmin()
                              if BPASS_TIME_BINS[np.abs((BPASS_TIME_BINS - log_age)).argmin()] <= log_age
                              else np.abs((BPASS_TIME_BINS - log_age)).argmin()-1
                              for log_age in log_ages ]

            which_col_bin = [np.abs((self.col_range - c)).argmin()
                             if self.col_range[np.abs((self.col_range - c)).argmin()] <= c
                             else np.abs((self.col_range - c)).argmin()-1
                             for c in colours]

            which_mag_bin = [np.abs((self.mag_range - mag)).argmin()
                             if self.mag_range[np.abs((self.mag_range - mag)).argmin()] <= mag
                             else np.abs((self.mag_range - mag)).argmin() - 1
                             for mag in my_data[3]]

            for mag_i, col_i, t_i in zip(which_mag_bin, which_col_bin, which_time_bin):
                self.stellar_pop_cmd[t_i, mag_i, col_i] += imf

        #except FileNotFoundError:
        #    self.file_does_not_exist.append(filename)
        #continue


    def plot(self):
        plt.figure(figsize=(10,10))

        plt.imshow(np.log10(self.stellar_pop_cmd[9]), cmap='Greys', aspect='auto')
        plt.xlim([90, 120])
        plt.ylim([120,0])
        plt.show()