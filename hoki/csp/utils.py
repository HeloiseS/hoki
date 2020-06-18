"""
Utilities to be used in the complex stellar populations
"""

from hoki.constants import BPASS_METALLICITIES
import hoki.load
import numpy as np
import numba
from scipy import interpolate
from hoki.constants import *
import pandas as pd
import matplotlib.pyplot as plt

# CODEREVIEW [H]: This is currently useless - There are two options here: Either the functionalities in
# CSP spectra and CSP rates can be streamlined and some of the global ones can be put in here OR this is useless
# and we drop this entierly.

class CSP(object):
    pass

########################
# Calculations per bin #
########################
# TODO add check if time outside of age universe range

def mass_per_bin(sfh, time_edges):
    """
    Gives the mass per bin for the given edges in time.

    Input
    -----
    sfh_arr : scipy.interpolate.splrep
        A scipy spline representation of the stellar formation history.
    time_edges : numpy array
        The edges of the bins in which the mass per bin is wanted in yrs.

    Output
    ------
    numpy.array
        The mass per time bin.
    """

    # CODEREVIEW [H]: Can spline be replaced by numpy interp?
    return np.array([interpolate.splint(t1, t2, sfh)
                        for t1, t2 in zip(time_edges[:-1], time_edges[1:])])



def metallicity_per_bin(metallicity, time_edges):
    """
    Gives the metallicity per bin for the given edges in time.

    Input
    -----
    sfh_arr : scipy.interpolate.splrep
        A scipy spline representation of the metallicity history
    time_edges : numpy array
        The edges of the bins in which the mass per bin is wanted in yrs.

    Output
    ------
    numpy.array
        The average metallicity per bin
    """

    # CODEREVIEW [H]: Can spline be replaced by numpy interp or a numpy funcitonality?
    Z_values = np.array(interpolate.splev(time_edges, metallicity))
    Z_average = (Z_values[1:] + Z_values[:-1])/2
    return np.array(Z_average)




########################
#  BPASS File Loading  #
########################


def load_rates(data_folder, binary=True, imf="imf135_300"):
    """Returns the requested file types.

    Input
    -----
    data_folder : str
        The filepath to the folder containing the BPASS data
    file_type : str
        The type of files to load (spectra, supernovae)
    binary : boolean
        Use the binary files or just the single stars. Default=True
    imf : str
        BPASS Identifier of the IMF to be used.

    Returns
    -------
    pandas.DataFrame
        A pandas MultiIndex dataframe containing the BPASS number of events
        per metallicity per type.
    """
    if binary:
        star = "bin"
    else:
        star = "sin"

    arrays = [BPASS_NUM_METALLICITIES, BPASS_EVENT_TYPES]
    columns = pd.MultiIndex.from_product(arrays, names=["Metallicicty", "Event Type"])

    rates = pd.DataFrame(index=np.linspace(6,11, 51), columns=columns, dtype=np.float64)
    rates.index.name = "log_age"
    for num, metallicity in enumerate(BPASS_METALLICITIES):
        data = hoki.load.model_output(f"{data_folder}/supernova-{star}-{imf}.z{metallicity}.dat")
        data = data.loc[:,slice(BPASS_EVENT_TYPES[0],BPASS_EVENT_TYPES[-1])]
        rates.loc[:, (BPASS_NUM_METALLICITIES[num], slice(None))] = data.to_numpy()

    return rates.swaplevel(0, 1, axis=1)


def load_spectra(data_folder, binary=True, imf="imf135_300"):
    """
    Load all BPASS spectra.

    Notes
    -----
    The first time this function is ran on a folder it will generate a pickle
    file containing all the BPASS spectra per metallicity for faster loading
    in the future. It stores the file in the same folder with the name:
    `all_spectra-[bin/sin]-[imf].pkl`

    Input
    -----
    data_folder : `str`
        The path to the folder containing the BPASS spectra.
    binary : boolean
        Use the binary files or just the single stars. Default=True
    imf : str
        BPASS Identifier of the IMF to be used.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the high quality BPASS spectra

    """
    if binary:
        star = "bin"
    else:
        star = "sin"

    try:
        spectra = pd.read_pickle(f"{data_folder}/all_spectra-{star}-{imf}.pkl")

    # CODEREVIEW [H]: No pokemon exceptions please ;)
    except:
        arrays = [BPASS_NUM_METALLICITIES, np.linspace(1, 100000, 100000)]
        columns = pd.MultiIndex.from_product(arrays, names=["Metallicicty", "wavelength"])
        print("Allocating memory")
        spectra = pd.DataFrame(index=np.linspace(6,11, 51), columns=columns, dtype=np.float64)
        spectra.index.name = "log_age"

        for num, metallicity in enumerate(BPASS_METALLICITIES):
            print(f"Loading metallicity: {metallicity}")
            data = hoki.load.model_output(f"{data_folder}/spectra-{star}-{imf}.z{metallicity}.dat")
            data = data.loc[:, slice("6.0", "11.0")].T
            spectra.loc[:,(BPASS_NUM_METALLICITIES[num], slice(None))] = data.to_numpy()
        spectra = spectra.swaplevel(0,1,axis=1)
        spectra.to_pickle(f"{data_folder}/all_spectra-{star}-{imf}.pkl")

    return spectra

def _normalise_rates(rates):
    """Normalise the BPASS rates.

    Input
    -----
    rates : pandas.DataFrame
        A pandas DataFrame containing the the events per bin


    Returns
    -------
    pandas.DataFrame
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
    Z_values : `numpy.array`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.array`
        An array containig the amount of mass per bin in the final binning.
    edges : `numpy.array`
        The bin edges of the Z_values and mass_per_bin
    rates : `2D numpy.array`
        A 2D array containig the different metallicities over time
        in BPASS binning. Format rates[metallicity][time]

    Returns
    -------
    `numpy.array`
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
def _at_time(Z_values, mass_per_bin, edges,rates):
    """Calculates the number of events at a specific time.

    Note
    ----
    edges[0] defines the time at which the events are calculated.

    Input
    -----
    Z_values : `numpy.array`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.array`
        An array containig the amount of mass per bin in the final binning.
    edges : `numpy.array`
        The bin edges of the Z_values and mass_per_bin.

    """
    Z_index_per_bin = np.array( [np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values])
    bin_index = np.array([_get_bin_index(i, BPASS_LINEAR_TIME_EDGES) for i in edges])
    out = 0.0
    for count in range(len(mass_per_bin)):
        out += rates[Z_index_per_bin[count]][bin_index[count]]*mass_per_bin[count]
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


def _type_check_histories(metallicity, SFH):
    # CODEREVIEW [H]: BEAU-TI-FULL
    if isinstance(metallicity, type(list)):
        raise HokiFatalError("metallicity is not a list. Only list are taken as input")
    if isinstance(SFH, type(list)):
        raise HokiFatalError("sfr is not a list. Only lists are taken as input.")
    if len(metallicity) != len(SFH):
        raise HokiFatalError("metallicity and sfr are not of equal length.")
