import pandas as pd
import numpy as np
import hoki.hrdiagrams
import hoki.load as load
from hoki.constants import *
import warnings
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError


class AgeWizard(object):
    """

    """
#TODO: documentation giiirl!

    def __init__(self, obs_df, model):
        """
        Initialisation of the AgeWizard object

        Parameters
        ----------
        obs_df: pandas.DataFrame
            Observational data. MUST contain a logT and logL column.
        model: str or hoki.hrdiagrams.HRDiagrams()
            Location of the modeled HRD of interest: this can be a PATH (in a string) to the file containing the
            HRD of choice or an already loaded HRDiagram object.
        """
        # Checking what format they giving for the model:
        if isinstance(model, hoki.hrdiagrams.HRDiagram):
            self.myhrd = model
        elif isinstance(model, str):
            self.myhrd = load.model_output(model, hr_type='TL')
        else:
            print('-----------------')
            print('HOKI DEBUGGER:\nThe model param should be a path to \na BPASS HRDiagram output file or\n',
                  'a hoki.hrdiagrams.HRDiagram')
            print('-----------------')
            raise TypeError('model is ' + str(type(model)))

        # Making sure the osbervational properties are given in a format we can use.
        if not isinstance(obs_df, pd.DataFrame):
            raise HokiFormatError("Observations should be stored in a Data Frame")

        # This will need to be re-asessed when I put in a feature for colour magnitude diagrams.
        if 'logL' not in obs_df.columns or 'logT' not in obs_df.columns:
            raise HokiFormatError("obs_df needs to contain a logL and a logT column")

        if 'name' not in obs_df.columns:
            warnings.warn("We expect the name of sources to be given in the 'name' column. "
                          "If I can't find names I'll make my own ;)", HokiFormatWarning)

        self.obs_df = obs_df
        self.hrd_coordinates = find_hrd_coordinates(self.obs_df, self.myhrd)

        self.pdfs = calculate_pdfs(self.obs_df, self.myhrd)
        self.sources = self.pdfs.columns[:-1].to_list()
        self.multiplied_pdf = None
        self._most_likely_age = None
        self._most_likely_ages = None

    def multiply_pdfs(self, not_you=None, return_df=False):
        """
        Calls the multiply_pdfs function

        Parameters
        ----------
        not_you: list, optional
            List of the column names to ignore. Default is None so all the pdfs are multiplied
        return_df: bool, optional
            Whether or not the resulting DataFrame should be returned (it is automatically stored in the
            attribute multiplied_pdf). Default is False.

        Returns
        -------
            None or pandas.DataFrame containing the multiplied pdf.

        """
        # if self.pdfs is None:
        #    raise AttributeError('self.pdfs is not yet defined -- have you run AgeWizard.calculate_pdfs()?')
        # warnings.warn('self.pdfs not yet defined -- running AgeWizard.calc_pdfs()', UserWarning)
        # self.calc_pdfs()

        self.multiplied_pdf = multiply_pdfs(self.pdfs, not_you)

        if return_df: return self.multiplied_pdf

    @property
    def most_likely_age(self):
        """
        Finds  the most likely age by finding the max value in self.multiplied_pdf
        """
        if self._most_likely_age is not None: return self._most_likely_age
        if self.multiplied_pdf is None:
            warnings.warn('self.multiplied_pdf is not yet defined -- running AgeWizard.combined_pdfs()', HokiUserWarning)
            self.multiply_pdfs()

        index = self.multiplied_pdf.index[self.multiplied_pdf.pdf == max(self.multiplied_pdf.pdf)].tolist()
        return BPASS_TIME_BINS[index]

    @property
    def most_likely_ages(self):
        """
        Finds the most likely ages for all the sources given in the obs_df DataFrame.
        """
        if self._most_likely_ages is not None:
            return self._most_likely_ages

        index = self.pdfs.drop('time_bins', axis=1).idxmax(axis=0).tolist()
        return BPASS_TIME_BINS[index]

    def calculate_p_given_age_range(self, age_range=None):
        """
        Calculates the probability that each source has age within age_range

        Parameters
        ----------
        age_range: list or tuple of 2 values
            Minimum and Maximum age to consider (inclusive).

        Returns
        -------
        numpy.array containing the probabilities.

        """
        probability = self.pdfs.drop('time_bins', axis=1)[
            (round(self.pdfs.time_bins, 2) >= min(age_range)) & (round(self.pdfs.time_bins, 2) <= max(age_range))].sum()
        return probability.values


def find_hrd_coordinates(obs_df, myhrd):
    """
    Find the BPASS HRD coordinates that match the given observations

    Parameters
    ----------
    obs_df: pandas.DataFrame
        Observational data. MUST contain a logT and logL column.
    myhrd: hoki.hrdiagrams.HRDiagrams
        BPASS HRDiagram

    Returns
    -------
    Tuple of lists:(logT coordinates, logL coordinates)
    """
    if not isinstance(obs_df, pd.DataFrame):
        raise HokiFormatError("obs_df should be a pandas.DataFrame")
    if not isinstance(myhrd, hoki.hrdiagrams.HRDiagram):
        raise HokiFormatError("myhrd should be an instance of hoki.hrdiagrams.HRDiagrams")

    # List if indices that located the HRD location that most closely matches observations
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
            warnings.warn("T="+str(T)+" cannot be converted to a float", HokiUserWarning)
            T_i.append(np.nan)

        try:
            L=float(L)
            L_i.append(int((np.where(abs(myhrd.L_coord - L) == abs(myhrd.L_coord - L).min()))[0]))
        except ValueError:
            warnings.warn("L="+str(L)+" cannot be converted to a float", HokiUserWarning)
            L_i.append(np.nan)

    return T_i, L_i


def normalise_1d(distribution):
    """
    Simple function that devides by the sum of the 1D array or DataFrame given.
    """
    area = np.sum([bin_t for bin_t in distribution])
    return distribution/area


def calculate_pdfs(obs_df, myhrd):
    """
    Given observations and an HR Diagram, calculates the age probability distribution functions.

    Parameters
    ----------
    obs_df: pandas.DataFrame
        Observational data. MUST contain a logT and logL column.
    myhrd: hoki.hrdiagrams.HRDiagrams
        BPASS HRDiagram

    Returns
    -------
    Age Probability Distribution Functions in a pandas.DataFrame.

    """
    T_coord, L_coord = find_hrd_coordinates(obs_df, myhrd)

    try:
        source_names = obs_df.name
    except AttributeError:
        warnings.warn("No source names given so I'll make my own", HokiUserWarning)
        source_names = ["s" + str(i) for i in range(obs_df.shape[0])]

    pdfs = []

    for i, name in zip(range(obs_df.shape[0]), source_names):
        Ti, Li = T_coord[i], L_coord[i]

        if np.isnan(Ti) or np.isnan(Li):
            warnings.warn("NaN Value encountered in (T,L) coordinates for source: " + name, HokiUserWarning)
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


def multiply_pdfs(pdf_df, not_you=None):
    """
    Multiplies together all the columns in given in DataFrame apart from the "time_bins" column

    Parameters
    ----------
    pdf_df: pandas.DataFrame
        DataFrame containing probability distribution functions
    not_you: list, optional
        List of the column names to ignore. Default is None so all the pdfs are multiplied

    Returns
    -------
    Combined Probability Distribution Function in a pandas.DataFrame.
    """
    assert isinstance(pdf_df, pd.DataFrame)

    combined_pdf = [1] * pdf_df.shape[0]
    if not_you:
        try:
            pdf_df = pdf_df.drop(labels=not_you, axis=1)
            #print('Labels: '+str(not_you)+' succesfully excluded.')
        except KeyError as e:
            message = 'FEATURE DISABLED'+'\nKeyError'+str(e)+'\nHOKI DIALOGUE: Your labels could not be dropped -- ' \
                                                              'all pdfs will be combined \nDEBUGGING ASSISTANT: ' \
                                                              'Make sure the labels your listed are spelled correctly:)'
            warnings.warn(message, HokiUserWarning)

    columns = [col for col in pdf_df.columns if "time_bins" not in col]

    for col in columns:  # pdf_df.columns[:-1]:
        combined_pdf *= pdf_df[col].values

    combined_df = pd.DataFrame(normalise_1d(combined_pdf))
    combined_df.columns = ['pdf']

    return combined_df