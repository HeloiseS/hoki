"""
Utilities to be used in the complex stellar populations
"""

from hoki.constants import BPASS_METALLICITIES
import hoki.load
import numpy as np
import numba
from hoki.constants import *
import pandas as pd
import matplotlib.pyplot as plt
from hoki.utils.exceptions import *
from hoki.data_compilers import SpectraCompiler


class CSP(object):
    now = HOKI_NOW

    def __init__(self):
        pass

    def _type_check_histories(self, sfh_functions, Z_functions):
        if not isinstance(sfh_functions, list):
            raise HokiTypeError("`sfh_functions` is not a list. Only lists are taken as input.")
        if not isinstance(Z_functions, list):
            raise HokiTypeError("`Z_functions` is not a list. Only lists are taken as input.")
        if len(sfh_functions) != len(Z_functions):
            raise HokiFormatError("sfh_functions and Z_functions must have the same length.")
        if not all(callable(val) for val in sfh_functions):
            raise HokiTypeError("sfh_functions must only contain functions.")
        if not all(callable(val) for val in Z_functions):
            raise HokiTypeError("Z_functions must only contain functions.")


########################
# Calculations per bin #
########################
# TODO add check if time outside of age universe range
# Can these be turned into numba functions?

def mass_per_bin(sfh_function, time_edges, sample_rate=100):
    """
    Gives the mass per bin for the given edges in time

    Notes
    -----
    The default `sample_rate` is set to 100 for a untested balance between
    speed and accuracy

    Input
    -----
    sfh_function : `function`
        A function giving the stellar formation history given a lookback time.
    time_edges : `numpy.ndarray`
        The edges of the bins in which the mass per bin is wanted in yrs.
    sample_rate : `int`
        The number of samples to take to use for the trapezodial integration.
        Default = 100

    Output
    ------
    `numpy.ndarray`
        The mass per time bin.
    """

    return np.array([np.trapz(sfh_function(np.linspace(t1, t2, sample_rate)),
                     np.linspace(t1, t2, sample_rate))
                     for t1, t2 in zip(time_edges[:-1], time_edges[1:])
                     ])



def metallicity_per_bin(Z_function, time_edges):
    """
    Gives the metallicity per bin for the given edges in time.

    Input
    -----
    Z_function : `function`
        A function giving the metallicity history given a lookback time.
    time_edges : `numpy.ndarray`
        The edges of the bins in which the mass per bin is wanted in yrs.

    Output
    ------
    `numpy.ndarray`
        The average metallicity per bin
    """
    Z_values = Z_function(time_edges)
    Z_average = (Z_values[1:] + Z_values[:-1])/2
    return np.array(Z_average)




########################
#  BPASS File Loading  #
########################


def load_rates(data_folder, imf, binary=True):
    """Loads the BPASS supernova event count files.

    Notes
    -----
    The rates are just read from file and not normalised.

    Input
    -----
    data_folder : `str`
        The filepath to the folder containing the BPASS data
    binary : `bool`
        Use the binary files or just the single stars. Default=True
    imf : `str`
        BPASS Identifier of the IMF to be used.

    Returns
    -------
    `pandas.DataFrame`
        A pandas MultiIndex dataframe containing the BPASS number of events
        per metallicity per type. Usage: rates.loc[log_age, (type, metallicity)]
    """
    if binary:
        star = "bin"
    else:
        star = "sin"

    # Create output DataFrame
    arrays = [BPASS_NUM_METALLICITIES, BPASS_EVENT_TYPES]
    columns = pd.MultiIndex.from_product(arrays, names=["Metallicicty", "Event Type"])
    rates = pd.DataFrame(index=np.linspace(6,11, 51), columns=columns, dtype=np.float64)
    rates.index.name = "log_age"

    # load supernova count files
    for num, metallicity in enumerate(BPASS_METALLICITIES):
        data = hoki.load.model_output(f"{data_folder}/supernova-{star}-{imf}.z{metallicity}.dat")
        data = data.loc[:,slice(BPASS_EVENT_TYPES[0],BPASS_EVENT_TYPES[-1])]
        rates.loc[:, (BPASS_NUM_METALLICITIES[num], slice(None))] = data.to_numpy()

    return rates.swaplevel(0, 1, axis=1)


def load_spectra(data_folder, imf, binary=True):
    """
    Load all BPASS spectra.

    Notes
    -----
    The first time this function is ran on a folder it will generate a pickle
    file containing all the BPASS spectra per metallicity for faster loading
    in the future. It stores the file in the same folder with the name:
    `all_spectra-[bin/sin]-[imf].pkl`

    The spectra are just read from file and not normalised.

    Input
    -----
    data_folder : `str`
        The path to the folder containing the BPASS spectra.
    binary : `bool`
        Use the binary files or just the single stars. Default=True
    imf : `str`
        BPASS Identifier of the IMF to be used.

    Returns
    -------
    `pandas.DataFrame`
        A pandas DataFrame containing the high quality BPASS spectra

    """
    if binary:
        star = "bin"
    else:
        star = "sin"

    # Check if compiled spectra are already present in data folder
    try:
        spectra = pd.read_pickle(f"{data_folder}/all_spectra-{star}-{imf}.pkl")

    # Compile the spectra for faster reading next time
    except FileNotFoundError:
        spec = SpectraCompiler(data_folder, data_folder,imf)
        spectra = spec.spectra
    return spectra

def _normalise_rates(rates):
    """Normalise the BPASS rates.

    Input
    -----
    rates : `pandas.DataFrame`
        A pandas DataFrame containing the the events per bin


    Returns
    -------
    `pandas.DataFrame`
        A pandas DataFrame containing the events/yr/M_\\odot
    """
    return rates.div(1e6*BPASS_LINEAR_TIME_INTERVALS, axis=0)


def _normalise_spectrum(spectra):
    return spectra.div(1e6, axis=0)


def _find_bpass_metallicities(Z_values):
    """Finds the nearest BPASS metallicities for each item in the list.

    Input
    -----
    Z_values : `list` or `numpy.array`
        A list of metallicities

    Returns
    -------
    `numpy.array`
        A list of the nearest BPASS metallicity for each given metallicity
    """
    return BPASS_NUM_METALLICITIES[[np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values]]



########################################
# Complex Stellar History Calculations #
########################################
# replace Z_index_per_bin to _find_bpass_metallicities


@numba.njit
def _over_time(Z_values, mass_per_bin, edges, rates):
    """Calculates the events rates per bin over the given bin edges.

    Parameters
    ----------
    Z_values : `numpy.ndarray`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per bin in the final binning.
    edges : `numpy.ndarray`
        The bin edges of the Z_values and mass_per_bin
    rates : `numpy.ndarray`
        A 2D array containig the different metallicities over time
        in BPASS binning. Format rates[metallicity][time]

    Returns
    -------
    `numpy.ndarray`
        The number of events per bin

    """
    Z_index_per_bin = np.array([np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values])
    event_rate = np.zeros(len(mass_per_bin))

    for count in range(len(mass_per_bin)):
            t = edges[count+1]
            for j in range(0,count+1):
                p1 = t - edges[j]
                p2 = t - edges[j+1]
                bin_events = _integral(p2,
                                       p1,
                                       BPASS_LINEAR_TIME_EDGES,
                                       rates[Z_index_per_bin[count]],
                                       BPASS_LINEAR_TIME_INTERVALS)
                event_rate[j] += bin_events*mass_per_bin[count]
    return event_rate


@numba.njit
def _at_time(Z_values, mass_per_bin, edges, rates):
    """Calculates the number of events at a specific time.

    Note
    ----
    edges[0] defines the time at which the events are calculated.

    Input
    -----
    Z_values : `numpy.ndarray`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per bin in the final binning.
    edges : `numpy.ndarray`
        The bin edges of the Z_values and mass_per_bin with usage
    rates : `numpy.ndarray`
        A 2D array containig the different metallicities over time
        in BPASS binning. Usage: rates[metallicity][time]

    Returns
    -------
    `float`
        The number of events happening at edges[0]
    """
    Z_index_per_bin = np.array([np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values])
    bin_index = np.array([_get_bin_index(i, BPASS_LINEAR_TIME_EDGES) for i in edges])
    out = 0.0
    for count in range(len(mass_per_bin)):
        out += rates[Z_index_per_bin[count]][bin_index[count]]*mass_per_bin[count]
    return out

@numba.njit
def _over_time_spectra(Z_values, mass_per_bin, edges, spectra):
    """Calculates the events rates per bin over the given bin edges.

    Parameters
    ----------
    Z_values : `numpy.ndarray`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per bin in the final binning.
    edges : `numpy.ndarray`
        The bin edges of the Z_values and mass_per_bin
    rates : `numpy.ndarray`
        A 2D array containig the different metallicities over time
        in BPASS binning. Format rates[metallicity][time]

    Returns
    -------
    `numpy.ndarray`
        The number of events per bin

    """
    Z_index_per_bin = np.array([np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values])
    output_spectra = np.zeros((len(mass_per_bin), 100000))

    for count in range(len(mass_per_bin)):
        t = edges[count+1]
        for j in range(0,count+1):
            p1 = t - edges[j]
            p2 = t - edges[j+1]
            for wl in np.arange(100000):
                bin_events = _integral(p2,
                                       p1,
                                       BPASS_LINEAR_TIME_EDGES,
                                       spectra[Z_index_per_bin[count]][wl],
                                       BPASS_LINEAR_TIME_INTERVALS)
                output_spectra[j][wl] += bin_events*mass_per_bin[count]
    return output_spectra



@numba.njit
def _at_time_spectrum(Z_values, mass_per_bin, edges, spectra):
    """Calculates the spectrum at a specific moment in lookback time.

    Note
    ----
    edges[0] defines the time at which the spectrum is calculated.

    Input
    -----
    Z_values : `numpy.ndarray`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per bin in the final binning.
    edges : `numpy.ndarray`
        The bin edges of the Z_values and mass_per_bin
    spectra : `numpy.ndarray`
        A numpy array containing the spectra with usage
        edges[age_bin][metallicity]

    Returns
    -------
    `float`
        The number of events happening at edges[0]
    """
    Z_index_per_bin = np.array([np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values])
    bin_index = np.array([_get_bin_index(i, BPASS_LINEAR_TIME_EDGES) for i in edges])
    out = np.zeros(100000)
    for count in range(len(mass_per_bin)):
        out += spectra[bin_index[count]][Z_index_per_bin[count]]*mass_per_bin[count]
    return out

##########################
#    HELPER FUNCTIONS    #
##########################
@numba.njit
def _integral(x1, x2, edges, values, bin_width):
    """The numba wrapper around a basic integration for the histogram.

    Parameters
    ----------
    x1 : float
        lower bound of the integration
    x2 : float
        upper bound of the integration
    edges : array
        The histogram bin edges
    values : array
        The values in each bin of the histogram
    bin_width : array
        The width of each bin in the historgam

    Returns
    -------
    float
        The integral between **x1** and **x2**

    """
    lower_bin = _get_bin_index(x1, edges)
    upper_bin = _get_bin_index(x2, edges)

    total = 0
    if lower_bin == upper_bin:
        total = values[lower_bin] * (x2-x1)
    else:

        ledge = lower_bin+1
        total += (values[lower_bin] * (edges[ledge]-x1))
        total += (values[upper_bin] * (x2 - edges[upper_bin]))

        if ledge < upper_bin:
            total += np.sum(bin_width[ledge:upper_bin]*values[ledge:upper_bin])

    return total

@numba.njit
def _get_bin_index(x, edges):
    """Get the bin number given the edges.

    Note
    -----
    Used to speed up the calculation.

    Parameters
    ----------
    x : float
        value where you want to know the bin number
    edges: array
        An array with the edges of the histogram
    Returns
    -------
    int
        bin index

    """
    if x < edges[0] or x > edges[-1]:
        # CODEREVIEW [H]: Use Hoki Exception (this is probably a hoki formatting problem if it's a formatting you chose
        # and want to impose.
        raise Exception("x outside of range")
    out = 0

    for i, val in enumerate(edges):
        if val > x:
            out = i-1
            break
    if x == edges[-1]:
        out = len(edges)-2
    return int(out)
    # if x < edges[0] or x > edges[-1]:
    #     raise Exception("x outside of range")
    # mids = (edges[1:] + edges[:-1])/2   # calculate bin midpoints
    # d = np.abs(x - mids)                # distance to midpoints
    # outer = np.where(d == d.min())      # find shortest
    # return outer[0][-1]           # select last, such that lower edge inclusive
