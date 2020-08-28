"""
Author: Max Briel

Utilities to be used in the complex stellar populations
"""

import numba
import numpy as np
import pandas as pd

import hoki.load
from hoki.constants import *
from hoki.utils.exceptions import *

########################
# Calculations per bin #
########################
# TODO add check if time outside of age universe range

@numba.njit(cache=True)
def _optimised_trapezodial_rule(y,x):
    """Basic Trapezodial rule integration

    Parameters
    ---------
    y : 'numpy.ndarray' (N)
        y values corresponding to `x`
    x : 'numpy.ndarray' (N)
        x values

    Returns
    -------
    float
        The integral over the given values

    """
    s = 0
    for j in range(1, len(x)):
        s += (x[j]-x[j-1])*(y[j]+y[j-1])
    return s/2

@numba.njit(cache=True)
def trapz_loop(dp, fp, sample_rate):
    """Loop over a trapezodial integration with a sample rate to determine bins

    Parameters
    ----------
    dp : `numpy.ndarray` (N)
        time points
    fp : `numpy.ndarray` (N)
        function values at the time points
    sample_rate : int
        the sample rate over which to integrate the values

    Returns
    -------
    `numpy.ndarray` (N-1)/sample_rate
        An array containing the integrals over bins separated by the sample_rate
    """
    l = int((len(dp)-1)/sample_rate)
    out = np.zeros(l)
    for i in range(l):
        j1 = i*sample_rate
        j2 = (i+1)*sample_rate + 1
        out[i] = _optimised_trapezodial_rule(fp[j1:j2],dp[j1:j2])
    return out


@numba.njit(cache=True)
def _optimised_mass_per_bin(time_points, sfh, time_edges, sample_rate):
    """Mass per bin calculation from grid data

    Parameters
    ----------
    time_point : `numpy.ndarray` (N)
        The time points at which the SFH is sampled

    sfh : `numpy.ndarray` (N)
        The SFH at the time_point samples

    time_edges : `numpy.ndarray` (M)
        The edges of the bins in which the mass per bin is wanted in yrs.

    sample_rate : `int`
        The number of samples to take to use for the trapezodial integration.
        Default = 25

    Returns
    -------
    `numpy.ndarray` (M-1)
        An array of the mass per time bin
    """
    l = len(time_edges)-1
    out = np.zeros(l)
    for i in range(l):
        x = np.linspace(time_edges[i],time_edges[i+1], sample_rate)
        y = np.interp(x, time_points, sfh)
        out[i] = _optimised_trapezodial_rule(y,x)
    return out

def mass_per_bin(sfh_function, time_edges, sample_rate=25):
    """
    Gives the mass per bin for the given edges in time

    Notes
    -----
    The default `sample_rate` is set to 25 for a untested balance between
    speed and accuracy

    Input
    -----
    sfh_function : `callable`
        A python callable (function) giving the stellar formation rate at a
        given lookback time.
        For faster calculations give a vector optimised function.

    time_edges : `numpy.ndarray`
        The edges of the bins in which the mass per bin is wanted in yrs.

    sample_rate : `int`
        The number of samples to take to use for the trapezodial integration.
        Default = 25

    Output
    ------
    `numpy.ndarray` [len(time_edges)-1]
        The mass per time bin.
    """

    # Subsample the time phase space
    # check if equal distance
    dif = np.diff(time_edges)
    if np.allclose(dif,dif[0]):
        dp = np.linspace(time_edges[0],
                        time_edges[-1],
                        (sample_rate-1)*(len(time_edges)-1)+len(time_edges))
    # not equal distance
    else:
        dp =  np.array(
            [np.linspace(t1, t2, sample_rate+1)[:-1]          # Remove last one to avoid repetition
                for t1, t2 in zip(time_edges[:-1], time_edges[1:])])
        dp = np.append(dp, time_edges[-1])                    # add final time edge back


    fp = sfh_function(dp)

    # make sure fp is a vector, even if sfh_function isn't a vectorized function
    if type(fp) != np.ndarray or len(fp) != len(dp):
        fp = np.vectorize(sfh_function)(dp)

    return trapz_loop(dp, fp, sample_rate)


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
    `numpy.ndarray` [len(time_edges)-1]
        The average metallicity per time bin
    """

    Z_values = Z_function(time_edges)
    if type(Z_values) != np.ndarray or len(Z_values) != len(time_edges):
        Z_values = np.vectorize(Z_function)(time_edges)

    Z_average = (Z_values[1:] + Z_values[:-1]) / 2
    return np.array(Z_average)


###########################
#  Normasise BPASS Files  #
###########################

def _normalise_rates(rates):
    """
    Normalise the BPASS event rates.

    Input
    -----
    rates : `pandas.DataFrame`
        A pandas DataFrame containing the the events per bin

    Returns
    -------
    `pandas.DataFrame`
        A pandas DataFrame containing the events/yr/M_\\odot
    """
    return rates / (1e6 * BPASS_LINEAR_TIME_INTERVALS[:, None])


def _normalise_spectrum(spectra):
    """
    Normalises the BPASS spectra.

    Input
    -----
    spectra : `numpy.ndarray`
        A numpy.ndarray containing the spectra for a 1e6 M_\\odot population.

    Returns
    -------
    `numpy.ndarray`
        A numpy.ndarray containing the spectra per mass (L_\\odot/M_\\odot).
    """
    return spectra*1e-6


###########################
#   BPASS Metallicities   #
###########################


def _find_bpass_metallicities(Z_values):
    """
    Finds the nearest BPASS metallicities for each item in the list.

    Input
    -----
    Z_values : `list` or `numpy.array`
        A list of metallicities

    Returns
    -------
    `numpy.array` [len(Z_values)]
        A list of the nearest BPASS metallicity for each given metallicity
    """
    return BPASS_NUM_METALLICITIES[[np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_values]]


########################################
# Complex Stellar History Calculations #
########################################


@numba.njit(cache=True)
def _at_time(Z_per_bin, mass_per_bin, time_edges, bpass):
    """
    Calculates the number of events at a specific time.

    Note
    ----
    time_edges[0] defines the time at which the events are calculated.

    Input
    -----
    Z_per_bin : `numpy.ndarray`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per bin in the final binning.
    time_edges : `numpy.ndarray`
        The bin edges of the Z_per_bin and mass_per_bin with usage
    bpass : `numpy.ndarray` (13, 51) or (13, 51, 100000)
        A ndarray containing either the event rates (13, 51)
        or the luminosity (13, 51, 100000) per metallicity and BPASS time bins.

        Usage: bpass_rates[0][2]
                        (gives the event rate at the metallicity 0.00001 and
                         a log_age of 6.2)

        Usage: `bpass_spectra[0][1][99]` or `bpass_spectra[0, 1, 99]`
                     (gives the L_\\odot/M_\\odot at Z=0.0001, 100 Angstrom at
                     log_age 6.0)

    Returns
    -------
    `float` or `numpy.ndarray` (100000)
        The number of events or the spectrum at time_edges[0]
    """
    Z_index_per_bin = np.array(
        [np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_per_bin])
    time_index_per_bin = np.array(
        [_get_bin_index(i, BPASS_LINEAR_TIME_EDGES) for i in time_edges])

    if len(bpass.shape) == 2:
        out = np.zeros(1)
    else:
        out = np.zeros(100000)

    for count in range(len(mass_per_bin)):
        rate = bpass[Z_index_per_bin[count], time_index_per_bin[count]] * mass_per_bin[count]
        out += rate
    return out



@numba.njit(cache=True)
def _over_time(Z_per_bin, mass_per_bin, time_edges, bpass_rates):
    """
    Calculates the events rates per bin over the given bin edges.

    Parameters
    ----------
    Z_per_bin : `numpy.ndarray`
        An array containing the metallicity values in each time bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per time bin.
    time_edges : `numpy.ndarray`
        The bin edges of the Z_per_bin and mass_per_bin

    bpass_rates : `numpy.ndarray`
        A 2D array containig the event rates for all BPASS metallicities over
        time in BPASS bins.
        Shape: (13x51)
        Usage: bpass_rates[0][2]
                (gives the event rate at the metallicity 0.00001 and
                 a log_age of 6.2)

    Returns
    -------
    `numpy.ndarray`
        The number of events per bin

    """
    Z_index_per_bin = np.array(
        [np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_per_bin])
    event_rate = np.zeros(len(mass_per_bin))

    for count in range(len(mass_per_bin)):
        t = time_edges[count + 1]
        for j in range(0, count + 1):
            p1 = t - time_edges[j]
            p2 = t - time_edges[j + 1]
            bin_events = _integral(p2,
                                   p1,
                                   BPASS_LINEAR_TIME_EDGES,
                                   bpass_rates[Z_index_per_bin[count]],
                                   BPASS_LINEAR_TIME_INTERVALS)
            event_rate[j] += bin_events * mass_per_bin[count]
    return event_rate


@numba.njit
def _over_time_spectrum(Z_per_bin, mass_per_bin, time_edges, bpass_spectra):
    """
    Calculates the spectra per bin over the given bin edges.

    Parameters
    ----------
    Z_per_bin : `numpy.ndarray`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per bin in the final binning.
    time_edges : `numpy.ndarray`
        The bin edges of the Z_per_bin and mass_per_bin
    bpass_spectra : `numpy.ndarray`
        A 3D array containig the BPASS spectra luminosities per solar mass for
        the BPASS metallicities and time bins.
        Shape: (13, 51, 100000) ([metallicities][log_ages][wavelength])
        Usage: `bpass_spectra[0][1][99]` or `bpass_spectra[0, 1, 99]`
                (gives the L_\\odot/M_\\odot at Z=0.00001, 100 Angstrom at
                log_age 6.1)

    Returns
    -------
    `numpy.ndarray`
        An `numpy.ndarray` containing a spectrum per time bin.
        Shape: (len(time_edges)-1, 100000)

    """
    Z_index_per_bin = np.array(
        [np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_per_bin])
    output_spectra = np.zeros((len(mass_per_bin), 100000))

    for count in range(len(mass_per_bin)):
        t = time_edges[count + 1]
        for j in range(0, count + 1):
            p1 = t - time_edges[j]
            p2 = t - time_edges[j + 1]
            for wl in np.arange(100000):
                bin_events = _integral(p2,
                                       p1,
                                       BPASS_LINEAR_TIME_EDGES,
                                       bpass_spectra[Z_index_per_bin[count], :, wl],
                                       BPASS_LINEAR_TIME_INTERVALS)
                output_spectra[j][wl] += bin_events * mass_per_bin[count]
    return output_spectra


##########################
#    HELPER FUNCTIONS    #
##########################

@numba.njit(cache=True)
def _integral(x1, x2, edges, values, bin_width):
    """
    Perfoms an intergral over histogram-like binned data.

    Parameters
    ----------
    x1 : `float`
        lower bound of the integration
    x2 : `float`
        upper bound of the integration
    edges : `numpy.ndarray`
        The edges of the bins
    values : `numpy.ndarray`
        The values in each bin
    bin_width : `numpy.ndarray`
        The width of each bin

    Returns
    -------
    `float`
        The integral between **x1** and **x2**

    """
    lower_bin = _get_bin_index(x1, edges)
    upper_bin = _get_bin_index(x2, edges)

    total = 0

    # Values within the same bin. Return fraction of the bin
    if lower_bin == upper_bin:
        total = values[lower_bin] * (x2 - x1)
    else:
        # Add part of the lower and upper bin to the total
        ledge = lower_bin + 1
        total += (values[lower_bin] * (edges[ledge] - x1))
        total += (values[upper_bin] * (x2 - edges[upper_bin]))

        # Add any remaining bins to the total
        if ledge < upper_bin:
            total += np.sum(bin_width[ledge:upper_bin]
                            * values[ledge:upper_bin])

    return total


@numba.njit(cache=True)
def _get_bin_index(x, edges):
    """
    Get the bin number given the edges.

    Note
    -----
    Numba is used to speed up the bin index calculation.

    Parameters
    ----------
    x : `float`
        value where you want to know the bin number
    edges: `numpy.ndarray`
        An array with the edges of the histogram
    Returns
    -------
    `int`
        bin index of x

    """
    # Check if x within edge ranges
    if x < edges[0] or x > edges[-1]:
        raise HokiFormatError("x outside of range")

    out = 0
    # if x is equal to the last bin, return len-2
    if x == edges[-1]:
        out = len(edges) - 2
    # Loop over bin edges to find bin
    for i, val in enumerate(edges):
        if val > x:  # x larger than bin, return answer
            out = i - 1
            break
    return int(out)
