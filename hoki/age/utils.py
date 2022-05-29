import pandas as pd
import hoki.hrdiagrams
import hoki.cmd
import hoki.load as load
from hoki.constants import BPASS_TIME_BINS
import warnings
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiFormatWarning
from hoki.utils.hoki_object import HokiObject
from hoki.utils.hoki_dialogue import HokiDialogue
import numpy as np
from scipy import stats, optimize

Dialogue = HokiDialogue()

#########
# MISC  #
#########


def normalise_1d(distribution, crop_the_future=False):
    """
    Simple function that devides by the sum of the 1D array or DataFrame given.
    """
    if crop_the_future:
        distribution = _crop_the_future(distribution)

    area = np.sum([bin_t for bin_t in distribution])
    return distribution / area


def _crop_the_future(distribution):
    # Anything about 10.1 is the future -  time bin 42 and above must have proba_bool == 0
    array_that_erases_the_future = np.array([1] * 42 + [0] * 9)
    return np.array(distribution) * array_that_erases_the_future


# TODO: Write this docstring
# TODO: Write test
def fit_lognorm_params(c, m, p, percentiles=np.array([0.16, 0.5, 0.84]), p0=1):
    """
    Fits the CDF of a log normal distribution to the percentiles and offset values of a parameter

    Parameters
    ----------
    c: numpy.array
        50th percentile, a.k.a median (c for center)
    m: numpy.array
        Lower error - 1 sigma (m for minus, because this value would be subtracted to the median to retrieve
        the Xth percentile). Please provide the absolute value.
    p: numpy.array
        Upper error - 1 sigma (p for plus, because these values would be added to the median to create
        the Yth percentile)
    percentiles: numpy.array

    p0: numpy.array

    Returns
    -------

    """
    ones = np.ones(c.shape[0])
    bs = c-ones
    xs = np.array([ones-m, ones, ones+p]).T

    ss, serrs= [], []
    for x in xs:
        s, serr = optimize.curve_fit(stats.lognorm.cdf, x, percentiles, p0=p0)
        ss.append(s)
        serrs.append(serr)

    # bizarre data format - fix
    ss = np.array(ss)
    ss = ss.reshape(ss.shape[0])
    serrs = np.array(serrs).astype(float)
    serrs = serrs.reshape(serrs.shape[0])
    return bs, ss, serrs


def _error_flag(obs_df):
    """ Checks what type of error case we are facing """

    error_flag = None

    # joins all the characters of the columns
    concat_columns = ''.join(map(str, obs_df.columns.to_list()[1:]))

    if '_err' in concat_columns:
        error_flag = 'ERR'
        print(f'{Dialogue.info()} ERROR_FLAG=SYM / Strictily symmetric errors detected')

    else:
        print(f'{Dialogue.info()} ERROR_FLAG=None / No errors detected')

    return error_flag

#######################
# FINDING COORDINATES #
#######################


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
            #print(T, (np.where(abs(myhrd.T_coord - T) == abs(myhrd.T_coord - T).min()))[0])
            # Finds the index that is at the minimum distance in Temperature space and adds it to the list
            T_i.append(int((np.where(abs(myhrd.T_coord - T) == abs(myhrd.T_coord - T).min()))[0]))

        except TypeError:
            T_i.append(int((np.where(abs(myhrd.T_coord - T) == abs(myhrd.T_coord - T).min()))[0][0]))

        except ValueError:
            warnings.warn("T=" + str(T) + " cannot be converted to a float", HokiUserWarning)
            T_i.append(np.nan)

        try:
            L = float(L)
            # Finds the index that is at the minimum distance in Luminosity space and adds it to the list
            L_i.append(int((np.where(abs(myhrd.L_coord - L) == abs(myhrd.L_coord - L).min()))[0]))

        except TypeError:
            L_i.append(int((np.where(abs(myhrd.L_coord - L) == abs(myhrd.L_coord - L).min()))[0][0]))

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

        except TypeError:
            col_i.append(int((np.where(abs(mycmd.col_range - col) == abs(mycmd.col_range - col).min()))[0][0]))

        except ValueError:
            warnings.warn("Colour=" + str(col) + " cannot be converted to a float", HokiUserWarning)
            col_i.append(np.nan)

        try:
            mag = float(mag)
            # Finds the index that is at the minimum distance in Magnitude space and adds it to the list
            mag_i.append(int((np.where(abs(mycmd.mag_range - mag) == abs(mycmd.mag_range - mag).min()))[0]))

        except TypeError:
            mag_i.append(int((np.where(abs(mycmd.mag_range - mag) == abs(mycmd.mag_range - mag).min()))[0][0]))

        except ValueError:
            warnings.warn("Magnitude=" + str(mag) + " cannot be converted to a float", HokiUserWarning)
            mag_i.append(np.nan)

    return col_i, mag_i


###############################
# CALCULATING INDIVIDUAL PDFS #
###############################
def calculate_individual_pdfs(obs_df, model, nsamples=100):
    """
    Calculates the individual age PDFs for each star

    Parameters
    ----------
    obs_df: pandas.DataFrame
        Observational data. MUST contain a logT and logL column (for HRD comparison) or a col and mag column
        (for CMD comparison)
    model: str or hoki.hrdiagrams.HRDiagrams() hoki.cmd.CMD()
        Location of the modeled HRD or CMD. This can be an already instanciated HRDiagram or CMD() object, or a
        path to an HR Diagram file or a pickled CMD.
    nsamples: int, optional
        Number of times each data point should be sampled from its error distribution. Default is 100.
        This only matters if you are taking errors into account.

    Returns
    -------

    """
    flag = _error_flag(obs_df)

    if flag is None:
        pdfs = calculate_individual_pdfs_None(obs_df, model)

    elif flag == 'ERR':
        pdfs = calculate_individual_pdfs_SYM_HRD(obs_df, model, nsamples=nsamples)

    return pdfs


def calculate_distributions(obs_df, model):
    """
    Given observations and an HR Diagram, calculates the distribution across ages (not normalised)
    Note to self: KEEP THIS I NEED IT

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

    source_names = obs_df.name
    distributions = []

    # Time to calcualte the pdfs
    for i, name in zip(range(obs_df.shape[0]), source_names):
        xi, yi = x_coord[i], y_coord[i]  # just saving space

        # Here we take care of the possibility that a coordinate is a NaN
        if np.isnan(xi) or np.isnan(yi):
            warnings.warn("NaN Value encountered in coordinates for source: " + name, HokiUserWarning)
            distributions.append([0] * 51)  # Probability is then 0 at all times - That star doesn't exist in our models
            continue

        # Here we fill our not-yet-nromalised distribution
        distrib_i = []
        for model_i in model:
            # For each time step i, we retrieve the proba_bool in CMD_i or HRD_i and fill our distribution element distrib_i
            # with it. At the end of the for loop we have iterated over all 51 time bins
            distrib_i.append(model_i[xi, yi])

        # Then we normalise, so that we have proper probability distributions
        # pdf_i = normalise_1d(distrib_i)

        # finally our pdf is added to the list
        distributions.append(distrib_i)

    # Our list of pdfs (which is a list of lists) is turned into a PDF with the source names as column names
    distributions_df = pd.DataFrame((np.array(distributions)).T, columns=source_names)
    # We add the time bins in there because it can make plotting extra convenient.
    # distributions_df['time_bins'] = hoki.constants.BPASS_TIME_BINS

    return distributions_df

##### SYMMETRIC ERROR ONLY

# TODO: make a CMD version of SYM
def calculate_individual_pdfs_SYM_HRD(obs_df, model, nsamples=100):
    # If source names not given we make our own
    try:
        source_names = obs_df.name
    except AttributeError:
        warnings.warn("No source names given so I'll make my own", HokiUserWarning)
        source_names = ["s" + str(i) for i in range(obs_df.shape[0])]
        obs_df['name']=source_names
    # If duplicates in source names
    if obs_df.name.unique().shape[0] - obs_df.name.shape[0] != 0.0:
        raise HokiFormatError(f"Duplicate names detected\n{Dialogue.debugger()} "
                              f"Please make sure the names of your sources are unique.")


    obs_df.index=obs_df.name
    pdfs=pd.DataFrame(np.zeros((obs_df.name.shape[0], 51)).T,
                      columns = obs_df.name.values)

    #### SAMPLING SYMMETRICAL ERRORS
    df_Ls= pd.DataFrame(np.zeros((nsamples, obs_df.name.shape[0])).T, index=obs_df.name.values)
    df_Ts= df_Ls.copy()

    for col in pdfs.columns:
        # For each star (column in main) we sample n times
        message = None
        try:
            df_Ls.loc[col] = np.random.normal(obs_df.loc[col].logL, obs_df.loc[col].logL_err, nsamples)
        except AttributeError:
            message=f"{Dialogue.info()} No error on L"
            df_Ls.loc[col] = [obs_df.loc[col].logL]*nsamples
        try:
            df_Ts.loc[col] = np.random.normal(obs_df.loc[col].logT, obs_df.loc[col].logT_err, nsamples)
        except AttributeError:
            message=f"{Dialogue.info()} No error on T"
            df_Ts.loc[col] = [obs_df.loc[col].logT]*nsamples

    print(message)

    # We're going to need to create temprary 'obs_df' that fit in pre-existing functions
    # This is the 'template' dataframe we're going to modify in every loop
    obs_df_temp = obs_df.copy()[['name', 'logT', 'logL']]

    for i in range(nsamples):
        obs_df_temp.logL = df_Ls[i]
        obs_df_temp.logT = df_Ts[i]

        distribs_i = calculate_distributions(obs_df_temp, model)
        pdfs += distribs_i

    print(f"{Dialogue.info()} Distributions Calculated Successfully")

    # and now that we've got our distributions all added up we normalise them!
    for col in pdfs.columns:
        pdfs[col]=normalise_1d(pdfs[col].values, crop_the_future=False)

    print(f"{Dialogue.info()} Distributions Normalised to PDFs Successfully")
    # this "main" dataframe can then just be fed into calculate_sample_pdf as distributions_df
    return pdfs


# #### NO ERRORS

def calculate_individual_pdfs_None(obs_df, model, nsamples=100, p0=1):
    # If source names not given we make our own
    try:
        source_names = obs_df.name
    except AttributeError:
        warnings.warn("No source names given so I'll make my own", HokiUserWarning)
        source_names = ["s" + str(i) for i in range(obs_df.shape[0])]
        obs_df['name']=source_names
    # If duplicates in source names
    if obs_df.name.unique().shape[0] - obs_df.name.shape[0] != 0.0:
        raise HokiFormatError(f"Duplicate names detected\n{Dialogue.debugger()} "
                              f"Please make sure the names of your sources are unique.")

    obs_df.index = obs_df.name

    pdfs = calculate_distributions(obs_df, model)
    print(f"{Dialogue.info()} Distributions Calculated Successfully")

    # and now that we've got our distributions all added up we normalise them!
    for col in pdfs.columns:
        pdfs[col] = normalise_1d(pdfs[col].values, crop_the_future=False)
    print(f"{Dialogue.info()} Distributions Normalised to PDFs Successfully")
    # this "main" dataframe can then just be fed into calculate_sample_pdf as distributions_df
    return pdfs

#####################################
# PUTTING PDFS TOGETHER IN SOME WAY #
#####################################


def calculate_sample_pdf(distributions_df, not_you=None):
    """
    Adds together all the columns in given in DataFrame apart from the "time_bins" column

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
                     'Make sure the labels you listed are spelled correctly:)'
            warnings.warn(message, HokiUserWarning)

    # We also must be careful not to multiply the time bin column in there so we have a list of the column names
    # that remain after the "not_you" exclusion minus the time_bins column.
    # columns = [col for col in distributions_df.columns if "time_bins" not in col]

    columns = []
    if "time_bins" not in distributions_df.columns:
        for col in distributions_df.columns:
            columns.append(col)

    for col in columns:
        # for col in distributions_df.columns:
        combined_pdf += distributions_df[col].values

    combined_df = pd.DataFrame(normalise_1d(combined_pdf))
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
