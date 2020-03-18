import pandas as pd
import numpy as np
import hoki.hrdiagrams
import hoki.load as load


def find_hrd_coordinates(obs_df, myhrd, stfu=False):
    assert isinstance(obs_df, pd.DataFrame), "obs_df should be a pandas.DataFrame"
    assert isinstance(myhrd, hoki.hrdiagrams.HRDiagram), "myhrd should be an instance of hoki.hrdiagrams.HRDiagrams"

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


def normalise_1d(distribution):
    area = np.sum([bin_t for bin_t in distribution])
    return distribution/area


def calculate_pdfs(obs_df, myhrd):
    T_coord, L_coord = find_hrd_coordinates(obs_df, myhrd)

    try:
        source_names = obs_df.name
    except AttributeError:
        print("No source names given so I'll make my own")
        source_names = ["s" + str(i) for i in range(obs_df.shape[0])]

    pdfs = []

    for i, name in zip(range(obs_df.shape[0]), source_names):
        Ti, Li = T_coord[i], L_coord[i]

        if np.isnan(Ti) or np.isnan(Li):
            print("ERROR: NaN Value encountered in (T,L) coordinates for source: " + name)
            pdfs.append([np.nan] * 51)
            continue

        distrib_i = []
        for hrd in myhrd:
            distrib_i.append(hrd[Ti, Li])

        pdf_i = normalise_1d(distrib_i)
        pdfs.append(pdf_i.tolist())

    pdf_df = pd.DataFrame((np.array(pdfs)).T, columns=source_names)
    pdf_df['time_bins'] = hoki.constants.BPASS_TIME_BINS

    return pdf_df


def combine_pdfs(pdf_df):
    assert isinstance(pdf_df, pd.DataFrame)

    combined_pdf = [0] * pdf_df.shape[0]

    columns = [col for col in pdf_df.columns if "time_bins" not in col]

    for col in columns:  # pdf_df.columns[:-1]:
        combined_pdf += pdf_df[col].values

    combined_df = pd.DataFrame(normalise_1d(combined_pdf))
    combined_df.columns = ['pdf']

    return combined_df