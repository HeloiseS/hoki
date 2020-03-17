import pandas as pd
import numpy as np
import hoki.load as load


def find_hrd_coordinates(obs_df, myhrd, stfu=False):
    # List if indeces that located the HRD location that most closely matches observations
    L_i = []
    T_i = []

    # How this works:
    # abs(myhrd.L_coord-L)==abs(myhrd.L_coord-L).min() *finds* the HRD location that most closely corresponds to obs.
    # np.where(....)[0] *finds* the index of that location (which was originally in L or T space)
    # int( ....) is juuust to make sure we get an integer because Python is a motherfucker and adds s.f. for no reason
    # Then we append that index to our list.

    for T, L in zip(obs_df.logT, obs_df.logL):

        try:
            T=float(T)
            T_i.append(int((np.where(abs(myhrd.T_coord - T) == abs(myhrd.T_coord - T).min()))[0]))
        except ValueError:
            print("T="+str(T)+" cannot be converted to a float")
            T_i.append(np.nan)


        try:
            L=float(L)
            L_i.append(int((np.where(abs(myhrd.L_coord - L) == abs(myhrd.L_coord - L).min()))[0]))
        except ValueError:
            print("L="+str(L)+" cannot be converted to a float")
            L_i.append(np.nan)





    return T_i, L_i
