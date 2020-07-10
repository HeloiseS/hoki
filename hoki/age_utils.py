import pandas as pd
import hoki.hrdiagrams
import hoki.cmd
import hoki.load as load
from hoki.constants import *
import warnings
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiFormatWarning
from hoki.utils.hoki_object import HokiObject


class AgeWizard(HokiObject):
    """
    AgeWizard object
    """

    def __init__(self, obs_df, model):
        """
        Initialisation of the AgeWizard object

        Parameters
        ----------
        obs_df: pandas.DataFrame
            Observational data. MUST contain a logT and logL column (for HRD comparison) or a col and mag column
            (for CMD comparison)
        model: str or hoki.hrdiagrams.HRDiagrams() hoki.cmd.CMD()
            Location of the modeled HRD or CMD. This can be an already instanciated HRDiagram or CMD() object, or a
            path to an HR Diagram file or a pickled CMD.
        """

        # Making sure the osbervational properties are given in a format we can use.
        if not isinstance(obs_df, pd.DataFrame):
            raise HokiFormatError("Observations should be stored in a Data Frame")

        if 'name' not in obs_df.columns:
            warnings.warn("We expect the name of sources to be given in the 'name' column. "
                          "If I can't find names I'll make my own ;)", HokiFormatWarning)

        # Checking what format they giving for the model:
        if isinstance(model, hoki.hrdiagrams.HRDiagram):
            self.model = model
        elif isinstance(model, hoki.cmd.CMD):
            self.model = model
        elif isinstance(model, str) and 'hrs' in model:
            self.model = load.model_output(model, hr_type='TL')
        elif isinstance(model, str):
            try:
                self.model = load.unpickle(path=model)
            except AssertionError:
                print('-----------------')
                print(
                    'HOKI DEBUGGER:\nThe model param should be a path to \na BPASS HRDiagram output file or pickled CMD,'
                    'or\na hoki.hrdiagrams.HRDiagram or a hoki.cmd.CMD')
                print('-----------------')
                raise HokiFatalError('model is ' + str(type(model)))

        else:
            print('-----------------')
            print('HOKI DEBUGGER:\nThe model param should be a path to \na BPASS HRDiagram output file or pickled CMD,'
                  'or\na hoki.hrdiagrams.HRDiagram or a hoki.cmd.CMD')
            print('-----------------')
            raise HokiFatalError('model is ' + str(type(model)))

        self.obs_df = obs_df
        self.coordinates = find_coordinates(self.obs_df, self.model)

        # This line is obsolete but might need revival if we ever want to add the not normalised distributions again
        # self._distributions = calculate_distributions_normalised(self.obs_df, self.model)

        self.pdfs = calculate_individual_pdfs(self.obs_df, self.model).fillna(0)
        self.sources = self.pdfs.columns.to_list()
        self.sample_pdf = None
        self._most_likely_age = None

    def calculate_sample_pdf(self, not_you=None, return_df=False):
        #self.sample_pdf = calculate_sample_pdf(self._distributions, not_you=not_you)
        self.sample_pdf = calculate_sample_pdf(self.pdfs, not_you=not_you)
        if return_df: return self.sample_pdf

    @property
    def most_likely_age(self):
        """
        Finds  the most likely age by finding the max value in self.calculate_sample_pdf
        """
        if self._most_likely_age is not None: return self._most_likely_age

        if self.sample_pdf is None:
            warnings.warn('self.multiplied_pdf is not yet defined -- running AgeWizard.combined_pdfs()',
                          HokiUserWarning)
            self.calculate_sample_pdf()

        index = self.sample_pdf.index[self.sample_pdf.pdf == max(self.sample_pdf.pdf)].tolist()
        return self.t[index]

    @property
    def most_likely_ages(self):
        """
        Finds the most likely ages for all the sources given in the obs_df DataFrame.
        """
        # index = self.pdfs.drop('time_bins', axis=1).idxmax(axis=0).tolist()
        index = self.pdfs.idxmax(axis=0).tolist()
        return self.t[index]

    def calculate_p_given_age_range(self, age_range):
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
        # Selects only the rows corresponding to the range age_range[0] to age_range[1] (inclusive)
        # and then we sum the probabilities up for each column.
        probability = calculate_p_given_age_range(self.pdfs, age_range)

        return probability


def find_coordinates(obs_df, model):
    """
    Finds the coordinates on a BPASS CMD or HRD that correspond to the given observations

    Parameters
    ----------
    obs_df: pandas.DataFrame
        Observational data. MUST contain a logT and logL column (for HRD comparison) or a col and mag column
        (for CMD comparison)

    model: str or hoki.hrdiagrams.HRDiagrams() hoki.cmd.CMD()
        Location of the modeled HRD or CMD. This can be an already instanciated HRDiagram or CMD() object, or a
        path to an HR Diagram file or a pickled CMD.

    Returns
    -------

    """

    if isinstance(model, hoki.hrdiagrams.HRDiagram):
        return _find_hrd_coordinates(obs_df, model)

    elif isinstance(model, hoki.cmd.CMD):
        return _find_cmd_coordinates(obs_df, model)

    else:
        raise HokiFormatError("The model should be an instance of hoki.hrdiagrams.HRDiagrams or hoki.cmd.CMD")


def _find_hrd_coordinates(obs_df, myhrd):
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
        raise HokiFormatError("model should be an instance of hoki.hrdiagrams.HRDiagrams")

    # List if indices that located the HRD location that most closely matches observations
    L_i = []
    T_i = []

    try:
        logT, logL = obs_df.logT, obs_df.logL
    except AttributeError:
        raise HokiFormatError("obs_df should have a logT and a logL column")

    # How this works:
    # abs(model.L_coord-L)==abs(model.L_coord-L).min() *finds* the HRD location that most closely corresponds to obs.
    # np.where(....)[0] *finds* the index of that location (which was originally in L or T space)
    # int( ....) is juuust to make sure we get an integer because Python is a motherfucker and adds s.f. for no reason
    # Then we append that index to our list.

    for T, L in zip(logT, logL):

        try:
            T = float(T)
            # Finds the index that is at the minimum distance in Temperature space and adds it to the list
            T_i.append(int((np.where(abs(myhrd.T_coord - T) == abs(myhrd.T_coord - T).min()))[0]))
        except ValueError:
            warnings.warn("T=" + str(T) + " cannot be converted to a float", HokiUserWarning)
            T_i.append(np.nan)

        try:
            L = float(L)
            # Finds the index that is at the minimum distance in Luminosity space and adds it to the list
            L_i.append(int((np.where(abs(myhrd.L_coord - L) == abs(myhrd.L_coord - L).min()))[0]))
        except ValueError:
            warnings.warn("L=" + str(L) + " cannot be converted to a float", HokiUserWarning)
            L_i.append(np.nan)

    return T_i, L_i


def _find_cmd_coordinates(obs_df, mycmd):
    """
    Find the BPASS HRD coordinates that match the given observations

    Parameters
    ----------
    obs_df: pandas.DataFrame
        Observational data. MUST contain a col and mag column.
    mycmd: hoki.cmd.CMD
        BPASS CMD

    Returns
    -------
    Tuple of lists:(colour coordinates, magnitude coordinates)
    """
    if not isinstance(obs_df, pd.DataFrame):
        raise HokiFormatError("obs_df should be a pandas.DataFrame")
    if not isinstance(mycmd, hoki.cmd.CMD):
        raise HokiFormatError("cmd should be an instance of hoki.cmd.CMD")

    # List if indices that located the HRD location that most closely matches observations
    col_i = []
    mag_i = []

    try:
        colours, magnitudes = obs_df.col, obs_df.mag
    except AttributeError:
        raise HokiFormatError("obs_df should have a logT and a logL column")

    # How this works:
    # abs(model.L_coord-L)==abs(model.L_coord-L).min() *finds* the HRD location that most closely corresponds to obs.
    # np.where(....)[0] *finds* the index
    # of that location (which was originally in L or T space)
    # int( ....) is juuust to make sure we get an integer because Python is a motherfucker and adds s.f. for no reason
    # Then we append that index to our list.

    for col, mag in zip(colours, magnitudes):

        try:
            col = float(col)
            # Finds the index that is at the minimum distance in Colour space and adds it to the list
            col_i.append(int((np.where(abs(mycmd.col_range - col) == abs(mycmd.col_range - col).min()))[0]))
        except ValueError:
            warnings.warn("Colour=" + str(col) + " cannot be converted to a float", HokiUserWarning)
            col_i.append(np.nan)

        try:
            mag = float(mag)
            # Finds the index that is at the minimum distance in Magnitude space and adds it to the list
            mag_i.append(int((np.where(abs(mycmd.mag_range - mag) == abs(mycmd.mag_range - mag).min()))[0]))
        except ValueError:
            warnings.warn("Magnitude=" + str(mag) + " cannot be converted to a float", HokiUserWarning)
            mag_i.append(np.nan)

    return col_i, mag_i


def normalise_1d(distribution, crop_the_future=False):
    """
    Simple function that devides by the sum of the 1D array or DataFrame given.
    """
    if crop_the_future:
        distribution = _crop_the_future(distribution)

    area = np.sum([bin_t for bin_t in distribution])
    return distribution / area


def _crop_the_future(distribution):
    # Anything about 10.1 is the future -  time bin 42 and above must have proba == 0
    array_that_erases_the_future = np.array([1]*42+[0]*9)
    return np.array(distribution)*array_that_erases_the_future


def calculate_individual_pdfs(obs_df, model):
    """
     Calculates the age pdfs of all the stars in the sample and returns them in a dataframe

     Parameters
     ----------
     obs_df: pandas.DataFrame
        Dataframe containing the observational data
     model: hoki.hrdiagrams.HRDiagrams or hoki.cmd.CMD
        BPASS HRDiagram or CMD

     Returns
     -------
     pandas.Dataframe containing the age pdfs of each star

    """
    likelihoods = calculate_distributions_normalised(obs_df, model)

    pdfs = []
    for col in likelihoods.columns:
        pdfs.append(normalise_1d(likelihoods[col].values))

    return pd.DataFrame(np.array(pdfs).T, columns=likelihoods.columns)


def calculate_distributions(obs_df, model):
    """
    Given observations and an HR Diagram, calculates the distribution across ages (not normalised)

    Parameters
    ----------
    obs_df: pandas.DataFrame
        Observational data. MUST contain a logT and logL column.
    model: hoki.hrdiagrams.HRDiagrams or hoki.cmd.CMD
        BPASS HRDiagram or CMD

    Returns
    -------
    Age Probability Distribution Functions in a pandas.DataFrame.

    """
    # Checking whether it;s HRD or CMD
    if isinstance(model, hoki.hrdiagrams.HRDiagram):
        x_coord, y_coord = find_coordinates(obs_df, model)
    if isinstance(model, hoki.cmd.CMD):
        y_coord, x_coord = find_coordinates(obs_df, model)  # yeah it's reversed... -_-

    # If source names not given we make our own
    try:
        source_names = obs_df.name
    except AttributeError:
        warnings.warn("No source names given so I'll make my own", HokiUserWarning)
        source_names = ["s" + str(i) for i in range(obs_df.shape[0])]

    likelihoods = []

    # Time to calcualte the pdfs
    for i, name in zip(range(obs_df.shape[0]), source_names):
        xi, yi = x_coord[i], y_coord[i]  # just saving space

        # Here we take care of the possibility that a coordinate is a NaN
        if np.isnan(xi) or np.isnan(yi):
            warnings.warn("NaN Value encountered in coordinates for source: " + name, HokiUserWarning)
            likelihoods.append([0] * 51)  # Probability is then 0 at all times - That star doesn't exist in our models
            continue

        # Here we fill our not-yet-nromalised distribution
        distrib_i = []
        for model_i in model:
            # For each time step i, we retrieve the proba in CMD_i or HRD_i and fill our distribution element distrib_i
            # with it. At the end of the for loop we have iterated over all 51 time bins
            distrib_i.append(model_i[xi, yi])

        # Then we normalise, so that we have proper probability distributions
        # pdf_i = normalise_1d(distrib_i)

        # finally our pdf is added to the list
        likelihoods.append(distrib_i)

    # Our list of pdfs (which is a list of lists) is turned into a PDF with the source names as column names
    likelihoods_df = pd.DataFrame((np.array(likelihoods)).T, columns=source_names)
    # We add the time bins in there because it can make plotting extra convenient.
    # distributions_df['time_bins'] = hoki.constants.BPASS_TIME_BINS

    return likelihoods_df


def calculate_distributions_normalised(obs_df, model):
    """
    Given observations and an HR Diagram, calculates the distribution across ages NORMALISED

    Parameters
    ----------
    obs_df: pandas.DataFrame
        Observational data. MUST contain a logT and logL column.
    model: hoki.hrdiagrams.HRDiagrams or hoki.cmd.CMD
        BPASS HRDiagram or CMD

    Returns
    -------
    Age Probability Distribution Functions in a pandas.DataFrame.

    """
    # Checking whether it;s HRD or CMD
    if isinstance(model, hoki.hrdiagrams.HRDiagram):
        x_coord, y_coord = find_coordinates(obs_df, model)
    if isinstance(model, hoki.cmd.CMD):
        y_coord, x_coord = find_coordinates(obs_df, model)  # yeah it's reversed... -_-

    # If source names not given we make our own
    try:
        source_names = obs_df.name
    except AttributeError:
        warnings.warn("No source names given so I'll make my own", HokiUserWarning)
        source_names = ["s" + str(i) for i in range(obs_df.shape[0])]

    likelihoods = []

    # Time to calcualte the pdfs
    for i, name in zip(range(obs_df.shape[0]), source_names):
        xi, yi = x_coord[i], y_coord[i]  # just saving space

        # Here we take care of the possibility that a coordinate is a NaN
        if np.isnan(xi) or np.isnan(yi):
            warnings.warn("NaN Value encountered in coordinates for source: " + name, HokiUserWarning)
            likelihoods.append([0] * 51)  # Probability is then 0 at all times - That star doesn't exist in our models
            continue

        # Here we fill our not-yet-nromalised distribution
        distrib_i = []
        for model_i in model:
            # For each time step i, we retrieve the proba in CMD_i or HRD_i and fill our distribution element distrib_i
            # with it. At the end of the for loop we have iterated over all 51 time bins
            distrib_i.append(model_i[xi, yi])

        # Then we normalise, so that we have proper probability distributions
        # pdf_i = normalise_1d(distrib_i)

        # finally our pdf is added to the list
        likelihoods.append(normalise_1d(distrib_i, crop_the_future=True))

    # Our list of pdfs (which is a list of lists) is turned into a PDF with the source names as column names
    likelihoods_df = pd.DataFrame((np.array(likelihoods)).T, columns=source_names)
    # We add the time bins in there because it can make plotting extra convenient.
    # distributions_df['time_bins'] = hoki.constants.BPASS_TIME_BINS

    return likelihoods_df


def calculate_sample_pdf(distributions_df, not_you=None):
    """
    Multiplies together all the columns in given in DataFrame apart from the "time_bins" column

    Parameters
    ----------
    distributions_df: pandas.DataFrame
        DataFrame containing probability distribution functions
    not_you: list, optional
        List of the column names to ignore. Default is None so all the pdfs are multiplied

    Returns
    -------
    Combined Probability Distribution Function in a pandas.DataFrame.
    """
    assert isinstance(distributions_df, pd.DataFrame)

    # We start our combined pdf with a list of 1s. We'll the multiply each pdf in sequence.

    combined_pdf = [0] * distributions_df.shape[0]

    # We want to allow the user to exclude certain columns -- we drop them here.
    if not_you:
        try:
            distributions_df = distributions_df.drop(labels=not_you, axis=1)
        except KeyError as e:
            message = 'FEATURE DISABLED' + '\nKeyError' + str(
                e) + '\nHOKI DIALOGUE: Your labels could not be dropped -- ' \
                     'all pdfs will be combined \nDEBUGGING ASSISTANT: ' \
                     'Make sure the labels your listed are spelled correctly:)'
            warnings.warn(message, HokiUserWarning)

    # We also must be careful not to multiply the time bin column in there so we have a list of the column names
    # that remain after the "not_you" exclusion minus the time_bins column.
    #columns = [col for col in distributions_df.columns if "time_bins" not in col]

    columns = []
    if "time_bins" not in distributions_df.columns:
        for col in distributions_df.columns:
            columns.append(col) 
            
    for col in columns:
        # for col in distributions_df.columns:
        combined_pdf += distributions_df[col].values

    combined_df = pd.DataFrame(normalise_1d(combined_pdf) )
    combined_df.columns = ['pdf']

    return combined_df


def calculate_p_given_age_range(pdfs, age_range=None):
    """
    Calculates the probability that each source has age within age_range

    Parameters
    ----------
    pdfs: pandas.DataFrame
        Age Probability Distributions Functions
    age_range: list or tuple of 2 values
        Minimum and Maximum age to consider (inclusive).

    Returns
    -------
    numpy.array containing the probabilities.

    """
    # Selects only the rows corresponding to the range age_range[0] to age_range[1] (inclusive)
    # and then we sum the probabilities up for each column.
    probability = pdfs[(np.round(BPASS_TIME_BINS, 2) >= min(age_range))
                       & (np.round(BPASS_TIME_BINS, 2) <= max(age_range))].sum()

    return probability
