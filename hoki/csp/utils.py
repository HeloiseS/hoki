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

class CSP(object):
    pass


########################
# Calculations per bin #
########################

def mass_per_bin(sfh, time_edges):
    """
    Gives the mass per bin for the given edges in time.

    Input
    -----
    sfh : scipy.interpolate.splrep
        A scipy spline representation of the stellar formation history.
    time_edges : numpy array
        The edges of the bins in which the mass per bin is wanted in yrs.

    Output
    ------
    numpy.array
        The mass per time bin.
    """
    return np.array([interpolate.splint(t1, t2, sfh)
                        for t1, t2 in zip(time_edges[:-1], time_edges[1:])])



def metallicity_per_bin(metallicity, time_edges):
    """
    Gives the metallicity per bin for the given edges in time.

    Input
    -----
    sfh : scipy.interpolate.splrep
        A scipy spline representation of the metallicity history
    time_edges : numpy array
        The edges of the bins in which the mass per bin is wanted in yrs.

    Output
    ------
    numpy.array
        The average metallicity per bin
    """
    Z_values = np.array(interpolate.splev(time_edges, metallicity))
    Z_average = (Z_values[1:] + Z_values[:-1])/2
    return np.array(Z_average)




########################
#  BPASS File Loading  #
########################


def _load_files(data_folder, file_type, binary=True, imf="imf135_300"):
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
    numpy.array
        A numpy array with the files loaded according to metallicty.
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
        data = hoki.load.model_output(f"{data_folder}/{file_type}-{star}-{imf}.z{metallicity}.dat")
        data = data.loc[:,slice(BPASS_EVENT_TYPES[0],BPASS_EVENT_TYPES[-1])]
        rates.loc[:, (BPASS_NUM_METALLICITIES[num], slice(None))] = data.to_numpy()

    return rates.swaplevel(0,1, axis=1)

def _normalise_rates(rates, bin_width):
    return rates.div(1e6*bin_width, axis=0)


def _find_bpass_metallicities(Z_values):
    return BPASS_NUM_METALLICITIES[[np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values]]



########################################
# Complex Stellar History Calculations #
########################################


@numba.njit
def _over_time(Z_values, mass_per_bin, edges, rates, DTD_width):
    """
    Numba function to calculate the event rates for the given rates, SFR,
    and mass per bin

    Parameters
    ----------
    Z_values: numpy.array
        An array containing the metallicity values at each bin.
    mass_per_bin : numpy.array
        An array containig the amount of mass per bin in the final binning.
    np_rates : 2D numpy.array
        A 2D array containig the different metallicities over time
        in BPASS binning. Format np_rates[metallicity][time]
    DTD_width : numpy.array
        An array containing the bin widths from the Delay Time Distribution.

    Returns
    -------
    An non-normalised event rate. Number of events per bin.

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
                                       DTD_width)
                event_rate[j] += bin_events*mass_per_bin[count]
    return event_rate


def _at_time():
    pass


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
    """
    The numba version to get the bin number.
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
        raise Exception("x outside of range")
    out = 0
    for i,val in enumerate(edges):
        if val > x:
            out = i-1
            break
    if x == edges[-1]:
        out = len(edges)-2
    return int(out)
