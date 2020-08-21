"""
Tools that allow to interpolate BPASS quantities on metallicity-age grids.

Author: Martin Glatzle
"""

import numpy as np
from scipy import interpolate
from hoki.constants import BPASS_NUM_METALLICITIES, BPASS_TIME_BINS
import warnings


class GridInterpolator():
    """
    Interpolate BPASS quantities on a metallicity-age grid.

    Base class for the interpolation of BPASS data (possibly vector valued,
    e.g. spectra) given on a grid of metallicities and ages. Using this class
    directly is discouraged, it has no public methods or attributes. It is
    advisable to implement convenience child classes for each specific
    quantity.

    Parameters
    ----------
    grid : `numpy.ndarray` (N_Z, N_a, D)
        A 3D numpy array containing the `D`-dimensional quantity to be
        interpolated as a function of metallicity and age.
    metallicities : `numpy.ndarray` (N_Z,), optional
        The metallicities at which `grid` is evaluated. Defaults to the full
        array of BPASS output metallicities.
    ages : `numpy.ndarray` (N_a,), optional
        The ages at which `grid` is evaluated. Defaults to the full array of
        BPASS output ages in log scale.
    dtype : `type`, optional
        The data type to be used by an instance of this class. Defaults to
        `numpy.float64`. Can be used to reduce memory footprint.

    Notes
    -----
    Uses `scipy.interpolate.LinearNDInterpolator`, which performs triangulation
    of the input data and linear barycentric interpolation on each
    triangle. Support for other interpolation methods should be fairly easy to
    implement.
    """
    def __init__(self, grid,
                 metallicities=BPASS_NUM_METALLICITIES,
                 ages=BPASS_TIME_BINS,
                 dtype=np.float64):
        for arr, ndim in zip([grid, metallicities, ages], [3, 1, 1]):
            if np.ndim(arr) != ndim:
                raise ValueError("Wrong dimensionality of input arrays.")
        if grid.shape[0] != len(metallicities):
            raise ValueError(
                "Shapes of `grid` and `metallicities` are incompatible."
            )
        if grid.shape[1] != len(ages):
            raise ValueError(
                "Shapes of `grid` and `ages` are incompatible."
            )
        self._dtype = dtype
        self._metallicities = metallicities.astype(self._dtype, copy=True)
        self._zMin = np.amin(self._metallicities)
        self._zMax = np.amax(self._metallicities)
        self._ages = ages.astype(self._dtype, copy=True)
        self._aMin = np.amin(self._ages)
        self._aMax = np.amax(self._ages)
        self._construct_interpolator(
            grid.astype(dtype, copy=False),
        )
        return

    def _construct_interpolator(self, grid):
        """
        Construct an interpolator on metallicity-age grid.

        Parameters
        ----------
        grid : `numpy.ndarray` (N_Z, N_a, D)
            A 3D numpy array containing the `D`-dimensional quantity to be
            interpolated as a function of metallicity and age.
        """
        zz, aa = np.meshgrid(self._metallicities, self._ages, indexing='ij')
        points = np.stack((zz, aa), -1).reshape((-1, 2))
        self._interpolator = interpolate.LinearNDInterpolator(
            points, grid.reshape((-1, grid.shape[2]))
        )
        return

    def _interpolate(self, metallicities, ages, masses=1):
        """
        Perform interpolation on this instance's grid.

        Parameters
        ----------
        metallicities : `numpy.ndarray` (N,)
            Stellar metallicities at which to interpolate. Same units as those
            used in the construction of this instance.
        ages : `numpy.ndarray` (N,)
            Stellar ages at which to interpolate. Same units as those used in
            the construction of this instance.
        masses : `numpy.ndarray` (N,) or `float`, optional
            Stellar masses in units of 1e6 M_\\odot. Used to scale the
            interpolation result. Defaults to unity.

        Returns
        -------
         : `numpy.ndarray` (N, D)
            Interpolation result.
        """
        # check dtypes
        if metallicities.dtype != self._dtype:
            warnings.warn(
                "Input metallicities for interpolation of wrong dtype, "
                "attempting copy and cast.",
                UserWarning
            )
            metallicities = np.array(
                metallicities, dtype=self._dtype
            )
        if ages.dtype != self._dtype:
            warnings.warn(
                "Input ages for interpolation of wrong dtype, "
                "attempting copy and cast.",
                UserWarning
            )
            ages = np.array(
                ages, dtype=self._dtype
            )

        # clipping
        if np.amax(metallicities) > self._zMax or \
           np.amin(metallicities) < self._zMin:
            warnings.warn(
                "Input metallicities for interpolation outside of available "
                f"range {self._zMin} -- {self._zMax} provided. "
                "They will be clipped.",
                UserWarning
            )
            metallicities = np.clip(metallicities, self._zMin, self._zMax)
        if np.amax(ages) > self._aMax or \
           np.amin(ages) < self._aMin:
            warnings.warn(
                "Input ages for interpolation outside of available "
                f"range {self._aMin} -- {self._aMax} provided. "
                "They will be clipped.",
                UserWarning
            )
            ages = np.clip(ages, self._aMin, self._aMax)
        return self._interpolator(metallicities, ages) * \
            self._check_masses(masses)

    def _check_masses(self, masses):
        """
        Make sure `masses` has correct dimensionality.

        Reshapes `masses` if it is a 1D array, does nothing if it is
        scalar. Also warns about masses potentially being too small.

        Parameters
        ----------
        masses : `numpy.ndarray` (N,) or `float`
            Stellar masses in units of 1e6 M_\\odot.

        Returns
        -------
        masses : `numpy.ndarray` (N, 1) or `float`
            Input reshaped to be multiplicable by interpolation result.
        """
        if np.amin(masses) < 1e-3:
            warnings.warn(
                "Input masses below 1000 M_sun! For such small populations,"
                " single stars can contribute a significant fraction of the"
                " population mass and re-scaling BPASS values averaged over"
                " more massive populations likely yields incorrect results.",
                UserWarning
            )
        if np.ndim(masses) == 0:
            return masses
        elif np.ndim(masses) == 1:
            return masses[:, None]
        else:
            raise ValueError("Wrong dimensionality of `masses`.")
