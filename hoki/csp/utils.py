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
# Can these be turned into numba functions?
@numba.njit
def _optimised_trapezodial_rule(y,x):
    s = 0
    for j in range(1, len(x)):
        s += (x[j]-x[j-1])*(y[j]+y[j-1])
    return s/2

@numba.njit
def _optimised_mass_per_bin(x_data, y_data, time_edges, sample_rate):
    l = len(time_edges)-1
    out = np.zeros(l)
    for i in range(l):
        x = np.linspace(time_edges[i],time_edges[i+1], sample_rate)
        y = np.interp(x, x_data, y_data)
        out[i] = _optimised_trapezodial_rule(y,x)
    return out

def mass_per_bin(sfh_function, time_edges, sample_rate=25):
    """
    Gives the mass per bin for the given edges in time

    Notes
    -----
    The default `sample_rate` is set to 100 for a untested balance between
    speed and accuracy

    Input
    -----
    sfh_function : `callable
        A python callable (function) giving the stellar formation rate at a
        given lookback time.
    time_edges : `numpy.ndarray`
        The edges of the bins in which the mass per bin is wanted in yrs.
    sample_rate : `int`
        The number of samples to take to use for the trapezodial integration.
        Default = 100

    Output
    ------
    `numpy.ndarray` [len(time_edges)-1]
        The mass per time bin.
    """
    # Vectorize function to allow numpy array input
    vec_func = np.vectorize(sfh_function)
    return np.array([np.trapz(vec_func(np.linspace(t1, t2, sample_rate)),
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
    `numpy.ndarray` [len(time_edges)-1]
        The average metallicity per time bin
    """
    # Vectorize function to allow numpy array input
    vec_func = np.vectorize(Z_function)
    Z_values = vec_func(time_edges)
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

@numba.njit
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
def _at_time(Z_per_bin, mass_per_bin, time_edges, bpass_rates):
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
    bpass_rates : `numpy.ndarray`
        A 2D array containig the different metallicities over time
        in BPASS binning.
        Shape: (13x51)
        Usage: bpass_rates[0][2]
                        (gives the event rate at the metallicity 0.00001 and
                         a log_age of 6.2)

    Returns
    -------
    `float`
        The number of events happening at time_edges[0]
    """
    Z_index_per_bin = np.array(
        [np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_per_bin])
    time_index_per_bin = np.array(
        [_get_bin_index(i, BPASS_LINEAR_TIME_EDGES) for i in time_edges])
    out = 0.0
    for count in range(len(mass_per_bin)):
        out += bpass_rates[Z_index_per_bin[count]
                           ][time_index_per_bin[count]] * mass_per_bin[count]
    return out


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


@numba.njit
def _at_time_spectrum(Z_per_bin, mass_per_bin, time_edges, bpass_spectra):
    """
    Calculates the spectrum at a specific moment in lookback time.

    Note
    ----
    time_edges[0] defines the time at which the spectrum is calculated.

    Input
    -----
    Z_per_bin : `numpy.ndarray`
        An array containing the metallicity values at each bin.
    mass_per_bin : `numpy.ndarray`
        An array containig the amount of mass per bin in the final binning.
    time_edges : `numpy.ndarray`
        The bin edges of the Z_per_bin and mass_per_bin
    bpass_spectra : `numpy.ndarray`
        A 3D array containig the BPASS spectra luminosities per solar mass for
        the BPASS metallicities and time bins.
        Shape: (13x51x100000) ([metallicities][log_ages][wavelength])
        Usage: `bpass_spectra[0][1][99]` or `bpass_spectra[0, 1, 99]`
                (gives the L_\\odot/M_\\odot at Z=0.0001, 100 Angstrom at
                log_age 6.0)

    Returns
    -------
    `numpy.ndarray` (100000)
        The spectrum at at time_edges[0]
    """
    Z_index_per_bin = np.array(
        [np.argmin(np.abs(i - BPASS_NUM_METALLICITIES)) for i in Z_per_bin])
    time_index_per_bin = np.array(
        [_get_bin_index(i, BPASS_LINEAR_TIME_EDGES) for i in time_edges])
    out = np.zeros(100000)
    for count in range(len(mass_per_bin)):
        out += bpass_spectra[Z_index_per_bin[count],
                             time_index_per_bin[count], :] * mass_per_bin[count]
    return out


##########################
#    HELPER FUNCTIONS    #
##########################

@numba.njit
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


@numba.njit
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
