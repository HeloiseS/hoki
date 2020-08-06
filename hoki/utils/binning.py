"""
Routines to bin spectra.

Author: Martin Glatzle
"""
import numpy as np
from numba import jit


def bin_spectra(wave, spectra, bins, edges=False):
    """
    Bin spectra conserving luminosity.

    Given SEDs sampled at certain wavelengths/frequencies will compute their
    values in given wavelength/frequency bins. The new SED values are bin
    averages computed using trapezoidal integration. This ensures that the
    luminosity per bin is conserved. Of course, only downsampling really makes
    sense here, i.e. the input SEDs should be well sampled compared to the
    desired output bins.

    Effectively converts input spectra to step functions of
    wavelength/frequency. Note, in particular, that this means that only
    rectangle rule integration can sensibly be performed on the output
    spectra. Higher order integration methods are not meaningful.

    Parameters
    ----------
    wave : `numpy.ndarray` (N_wave,)
        Wavelengths or frequencies at which spectra are known.
    spectra : `numpy.ndarray` (N, N_wave)
        The SEDs to resample given as L_lambda [Energy/Time/Wavelength] or L_nu
        [Energy/Time/Frequency] in accordance with `wave`.
    bins : `numpy.ndarray` (N_bins,)
        The bins to which to resample spectra. Either values in the bins or
        their edges. See `edges`. Required to lie within the range provided by
        `wave`.
    edges : bool, optional
        Whether the values given in `bins` are bin edges or values in the
        bins. If `True`, `N_wave_new=N_bins-1`. If `False`, `N_wave_new=N_bins`
        and in this case bin edges are constructed such that they always lie
        between neighbouring points. The first/last bin is assumed to be
        symmetric around the first/last value in bins.

    Returns
    -------
    wave_new : `numpy.ndarray` (N_wave_new,)
        The wavelength/frequency values to which spectra were binned. If edges
        is `False`, this will be identical to `bins`. Otherwise it will be the
        bin centers.
    spectra_new : `numpy.ndarray` (N, N_wave_new)
        The binned spectra.

    Notes
    -----
    For the actual integration, `wave` has to be sorted in ascending or
    descending order. If this is not the case, `wave` and `spectra` will be
    sorted/re-ordered, which, depending on their sizes, might imply significant
    overhead. `bins` will always be sorted in the same order as `wave` as it is
    assumed to generally be relatively small.
    """
    for arr, ndim in zip([wave, spectra, bins], [1, 2, 1]):
        if np.ndim(arr) != ndim:
            raise ValueError("Wrong dimensionality of input arrays.")
    if spectra.shape[1] != len(wave):
        raise ValueError("Shapes of `wave` and `spectra` are incompatible.")

    diff = np.diff(wave)
    if np.all(diff > 0):
        asc = True
    elif np.all(diff < 0):
        asc = False
    else:
        if np.any(diff == 0):
            raise ValueError("Identical values provided in `wave`.")
        ids = np.argsort(wave)
        wave = wave[ids]
        spectra = spectra[:, ids]
        asc = True
    if asc:
        bins = np.sort(bins)
    else:
        bins = bins[np.argsort(-1*bins)]

    if edges:
        wave_new = (bins[1:] + bins[:-1])/2
        bin_edges = bins
    else:
        wave_new = bins
        bin_edges = np.empty((len(bins) + 1))
        bin_edges[1:-1] = (bins[1:] + bins[:-1])/2
        bin_edges[0] = bins[0] - (bin_edges[1]-bins[0])
        bin_edges[-1] = bins[-1] + (bins[-1]-bin_edges[-2])
    if not (bin_edges[0] >= np.amax(wave[0])
            and bin_edges[-1] <= np.amin(wave[-1])):
        raise ValueError("bin_edges outside of valid range!")

    spectra_new = _binwise_trapz_sorted(wave, spectra, bin_edges) \
        / np.diff(bin_edges)

    return wave_new, spectra_new


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
