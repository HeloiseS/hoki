"""
Module to hold functions and utilities to be applied to spectra, especially
BPASS synthetic spectra
"""
import numpy as np
from numba import jit
import numbers


def dopcor(df, z, wl_col_index=0):
    """
    Basis doppler correction for hoki's dataframes

    Notes
    -----
    The correction is applied IN PLACE.
    """
    wl_dopcor = (df.iloc[:, wl_col_index].values) - (df.iloc[:, wl_col_index].values * z)
    df.iloc[:, wl_col_index] = wl_dopcor
    return


def bin_luminosity(wl, spectra, bins=10):
    """
    Bin spectra conserving luminosity.

    Given spectra sampled at certain wavelengths/frequencies will compute their
    values in given wavelength/frequency bins. These values are bin averages
    computed using trapezoidal integration, which ensures that the luminosity
    per bin is conserved. Of course, only downsampling really makes sense here,
    i.e. the input spectra should be well sampled compared to the desired
    output bins.

    Effectively converts input spectra to step functions of
    wavelength/frequency. Note, in particular, that this means that only
    rectangle rule integration can sensibly be performed on the output
    spectra. Other integration methods are not meaningful.

    Parameters
    ----------
    wl : `numpy.ndarray` (N_wl,)
        Wavelengths or frequencies at which spectra are known.
    spectra : `numpy.ndarray` (N, N_wl)
        The spectra to bin given as L_lambda [Energy/Time/Wavelength] or L_nu
        [Energy/Time/Frequency] in accordance with `wl`.
    bins : int or `numpy.ndarray` (N_edges,), optional
        Either an integer giving the number `N_bins` of desired equal-width
        bins or an array of bin edges, required to lie within the range given
        by `wl`. In the latter case, `N_bins=N_edges-1`.

    Returns
    -------
    wl_new : `numpy.ndarray` (N_bins,)
        The wavelength/frequency values to which spectra were binned,
        i.e. centre bin values.
    spectra_new : `numpy.ndarray` (N, N_bins)
        The binned spectra.

    Notes
    -----
    For the actual integration, `wl` has to be sorted in ascending or
    descending order. If this is not the case, `wl` and `spectra` will be
    sorted/re-ordered. `bins` will always be sorted in the same order as `wl`
    as it is assumed to generally be relatively small.

    Although the language used here refers to spectra, the primary intended
    application area, the code can naturally be used to bin any function with
    given samples, conserving its integral bin-wise.
    """
    for arr, ndim in zip([wl, spectra, bins], [[1], [2], [0, 1]]):
        if np.ndim(arr) not in ndim:
            raise ValueError("Wrong dimensionality of input arrays.")
    if spectra.shape[1] != len(wl):
        raise ValueError("Shapes of `wl` and `spectra` are incompatible.")

    diff = np.diff(wl)
    if np.all(diff > 0):
        asc = True
    elif np.all(diff < 0):
        asc = False
    else:
        if np.any(diff == 0):
            raise ValueError("Identical values provided in `wl`.")
        ids = np.argsort(wl)
        wl = wl[ids]
        spectra = spectra[:, ids]
        asc = True

    if isinstance(bins, numbers.Integral):
        bins = np.linspace(wl[0], wl[-1], num=bins+1)
    else:
        if asc:
            bins = np.sort(bins)
        else:
            bins = bins[np.argsort(-1*bins)]
    if not (np.amax(bins) <= np.amax(wl) and np.amin(bins) >= np.amin(wl)):
        raise ValueError("Bin edges outside of valid range!")

    wl_new = (bins[1:] + bins[:-1])/2
    spectra_new = _binwise_trapz_sorted(wl, spectra, bins) \
        / np.diff(bins)

    return wl_new, spectra_new


@jit(nopython=True, nogil=True, cache=True)
def _binwise_trapz_sorted(x, y, bin_edges):
    """
    Trapezoidal integration over bins.

    Integrate each row of `y(x)` over each bin defined by `bin_edges` using
    trapezoidal integration. The values of `bin_edges` do not have to coincide
    with values given in `x`, the rows of `y` are linearly interpolated
    correspondingly.

    Parameters
    ----------
    x : `numpy.ndarray` (N_x,)
        `x`-values corresponding to each column of `y`. Assumed to be sorted in
        ascending or descending order. Integrated values will be negative for
        descending order.
    y : `numpy.ndarray` (N, N_x)
        N functions of `x` evaluated at each of its values.
    bin_edges : `numpy.ndarray` (N_bins+1,)
        Edges of the bins over which to perform integration. Assumed to be
        sorted in same order as `x` and to span a range <= the range spanned by
        `x`.

    Returns
    -------
    res : `numpy.ndarray` (N, N_bins)
        Integral over each bin of each row of `y`.
    """
    res = np.empty((y.shape[0], len(bin_edges)-1))

    i1 = 0
    i2 = 0
    y1 = np.empty((y.shape[0]))
    y2 = np.empty((y.shape[0]))
    for j in range(res.shape[1]):
        x1 = bin_edges[j]
        x2 = bin_edges[j+1]

        # ascending
        if x[0] < x[1]:
            # Find last element <x1 and last element <x2 in x.
            while x1 > x[i1+1]:
                i1 += 1
            i2 = i1
            while x2 > x[i2+1]:
                i2 += 1
        # descending
        elif x[0] > x[1]:
            # Find last element >x1 and last element >x2 in x.
            while x1 < x[i1+1]:
                i1 += 1
            i2 = i1
            while x2 < x[i2+1]:
                i2 += 1
        else:
            raise ValueError("Identical values in `x`!")

        # Find y1=y(x1) and y2=y(x2) by interpolation.
        y1 = (
            (x[i1+1]-x1)*y[:, i1] + (x1-x[i1])*y[:, i1+1]
        ) / (x[i1+1]-x[i1])
        y2 = (
            (x[i2+1]-x2)*y[:, i2] + (x2-x[i2])*y[:, i2+1]
        ) / (x[i2+1]-x[i2])

        if i1 == i2:
            # Have only area from x1 to x2.
            res[:, j] = (x2-x1)*(y1+y2)/2
        else:
            # Area from x1 to x(i1+1).
            res[:, j] = (x[i1+1]-x1)*(y1+y[:, i1+1])/2
            # Add area from x(i1+1) to x(i2-1).
            for i in range(i1+1, i2):
                res[:, j] += (x[i+1]-x[i])*(y[:, i]+y[:, i+1])/2
            # Add area from x(i2) to x2.
            res[:, j] += (x2-x[i2])*(y2+y[:, i2])/2

    return res
