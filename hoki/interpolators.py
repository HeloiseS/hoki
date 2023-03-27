"""
Tools that allow to interpolate BPASS quantities on metallicity-age grids.

Author: Martin Glatzle
"""

import numpy as np
from scipy import interpolate
from hoki.constants import (
    BPASS_NUM_METALLICITIES, BPASS_TIME_BINS, BPASS_WAVELENGTHS
)
from hoki import load
import warnings


class GridInterpolator():
    """
    Interpolate BPASS quantities on a metallicity-age grid.

    Base class for the interpolation of BPASS quantities (possibly vector
    valued) given on a grid of metallicities and ages. Using this class
    directly is discouraged, it has no public methods or attributes. It is
    advisable to use/implement convenience child classes for each specific
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
        BPASS output ages [yr] in log scale.
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
                f"Shapes of `grid` {grid.shape} and "
                f"`metallicities` {metallicities.shape} are incompatible."
            )
        if grid.shape[1] != len(ages):
            raise ValueError(
                f"Shapes of `grid` {grid.shape} and "
                f"`ages` {ages.shape} are incompatible."
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

    def _interpolate(self, metallicities, ages):
        """
        Perform interpolation on this instance's grid.

        Input parameters will be clipped to available range.

        Parameters
        ----------
        metallicities : `numpy.ndarray` (N,)
            Stellar metallicities at which to interpolate. Same units as those
            used in the construction of this instance.
        ages : `numpy.ndarray` (N,)
            Stellar ages at which to interpolate. Same units as those used in
            the construction of this instance.

        Returns
        -------
         : `numpy.ndarray` (N, D)
            Interpolation result.
        """
        # check dtypes
        if np.asarray(metallicities).dtype != self._dtype:
            warnings.warn(
                "Input metallicities for interpolation of wrong dtype, "
                "attempting copy and cast.",
                UserWarning
            )
            metallicities = np.array(
                metallicities, dtype=self._dtype
            )
        if np.asarray(ages).dtype != self._dtype:
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
                f"log range {self._aMin} -- {self._aMax} [yr] provided. "
                "They will be clipped.",
                UserWarning
            )
            ages = np.clip(ages, self._aMin, self._aMax)
        return self._interpolator(metallicities, ages)


class GridInterpolatorMassScaled(GridInterpolator):
    """
    Interpolate BPASS quantities that scale with population mass on a
    metallicity-age grid.

    Base class for the interpolation of BPASS quantities (possibly vector
    valued) which scale with stellar population mass given on a grid of
    metallicities and ages. Using this class directly is discouraged, it has no
    public methods or attributes. It is advisable to use/implement convenience
    child classes for each specific quantity.

    Parameters
    ----------
    grid : `numpy.ndarray` (N_Z, N_a, D)
        A 3D numpy array containing the `D`-dimensional quantity to be
        interpolated as a function of metallicity and age, normalized to a
        population mass of 1e6 M_\\odot.
    metallicities : `numpy.ndarray` (N_Z,), optional
        The metallicities at which `grid` is evaluated. Defaults to the full
        array of BPASS output metallicities.
    ages : `numpy.ndarray` (N_a,), optional
        The ages at which `grid` is evaluated. Defaults to the full array of
        BPASS output ages [yr] in log scale.
    dtype : `type`, optional
        The data type to be used by an instance of this class. Defaults to
        `numpy.float64`. Can be used to reduce memory footprint.
    """
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
            Stellar population masses in units of 1e6 M_\\odot. Used to scale
            the interpolation result. Defaults to unity.

        Returns
        -------
         : `numpy.ndarray` (N, D)
            Interpolation result.
        """
        return super()._interpolate(metallicities, ages) * \
            self._check_masses(masses)

    @staticmethod
    def _check_masses(masses):
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


class SpectraInterpolator(GridInterpolatorMassScaled):
    """
    Interpolate BPASS SSP spectra on a metallicity-age grid.

    Interpolate a spectrum grid for single stellar populations (SSPs) with
    fixed IMF provided by BPASS over its metallicity-age grid. The wavelength
    range can be limited to something smaller than the BPASS default to reduce
    memory footprint. Provided limits beyond the available range will be
    clipped.

    Parameters
    ----------
    data_path : `str`
        The path to the folder containing the BPASS spectra files.
    imf : `str`
        BPASS Identifier of the IMF to be used, e.g. `"imf_chab100"`.
    binary : `bool`, optional
        Use spectra including binaries or only single stars. Defaults to
        `True`.
    lam_min : `float`, optional
        Lower limit of the wavelength range on which this instance will perform
        interpolation. Defaults to `None`, using the minimal wavelength
        available.
    lam_max : `float`, optional
        Upper limit of the wavelength range on which this instance will perform
        interpolation. Defaults to `None`, using the maximal wavelength
        available.
    dtype : `type`, optional
        The data type to be used by an instance of this class. Defaults to
        `numpy.float64`. Can be used to reduce memory footprint.
    """

    def __init__(self, data_path, imf, binary=True,
                 lam_min=None, lam_max=None,
                 dtype=np.float64):
        if lam_min is not None and lam_max is not None:
            if lam_min >= lam_max:
                raise ValueError("lam_min is larger than or equal to lam_max!")

        lam = BPASS_WAVELENGTHS
        if lam_min is not None:
            idx_min = np.searchsorted(lam, lam_min, side='left')
        else:
            idx_min = None
        if lam_max is not None:
            idx_max = np.searchsorted(lam, lam_max, side='right')
        else:
            idx_max = None
        self._wavelengths = lam[idx_min:idx_max].astype(
            dtype, copy=True)

        self._spectra = load.spectra_all_z(
            data_path, imf, binary=binary)[:, :, idx_min:idx_max].astype(
                dtype, copy=True
        )

        if len(self._wavelengths) != self._spectra.shape[2]:
            raise ValueError(
                "Incompatible dimesions for wavelengths "
                f"{self._wavelengths.shape} and spectra {self._spectra.shape}."
            )
        super().__init__(self._spectra, dtype=dtype)

        return

    def __call__(self, metallicities, ages, masses=1):
        """
        Perform interpolation on this instance's spectrum grid.

        Parameters
        ----------
        metallicities : `numpy.ndarray` (N,)
            Absolute initial stellar metallicities at which to
            interpolate.
        ages : `numpy.ndarray` (N,)
            Stellar ages [yr] in log scale at which to interpolate.
        masses : `numpy.ndarray` (N,) or `float`, optional
            Stellar population masses in units of 1e6 M_\\odot. Used to scale
            the interpolation result. Defaults to unity.

        Returns
        -------
         : `numpy.ndarray` (N_lam,)
            The wavelengths [angstrom] at which interpolated spectra are
            provided.
         : `numpy.ndarray` (N, N_lam)
            Interpolated SEDs [L_\\odot/angstrom].
        """
        return self._wavelengths, \
            self._interpolate(metallicities, ages, masses)


class EmissivitiesInterpolator(GridInterpolatorMassScaled):
    """
    Interpolate BPASS SSP emissivities on a metallicity-age grid.

    Interpolate a grid of emissivities for single stellar populations (SSPs)
    with fixed IMF provided by BPASS over its metallicity-age grid.

    Parameters
    ----------
    data_path : `str`
        The path to the folder containing the BPASS emissivity files.
    imf : `str`
        BPASS Identifier of the IMF to be used, e.g. `"imf_chab100"`.
    binary : `bool`, optional
        Use spectra including binaries or only single stars. Defaults to
        `True`.
    dtype : `type`, optional
        The data type to be used by an instance of this class. Defaults to
        `numpy.float64`. Can be used to reduce memory footprint.
    """

    def __init__(self, data_path, imf, binary=True,
                 dtype=np.float64):

        self._emissivities = 10**(
            load.emissivities_all_z(data_path, imf, binary=binary)
        ).astype(dtype, copy=True)

        super().__init__(self._emissivities, dtype=dtype)

        return

    def __call__(self, metallicities, ages, masses=1):
        """
        Perform interpolation on this instance's emissivities grid.

        Parameters
        ----------
        metallicities : `numpy.ndarray` (N,)
            Absolute initial stellar metallicities at which to
            interpolate.
        ages : `numpy.ndarray` (N,)
            Stellar ages [yr] in log scale at which to interpolate.
        masses : `numpy.ndarray` (N,) or `float`, optional
            Stellar population masses in units of 1e6 M_\\odot. Used to scale
            the interpolation result. Defaults to unity.

        Returns
        -------
         : `numpy.ndarray` (N, 4)
            Interpolated emissivities:

            Nion in 1/s:
                ionizing photon production rate
            L_Halpha in ergs/s:
                Balmer H line luminosity, assuming =log(Nion/s)-11.87
            L_FUV in ergs/s/A:
                luminosity in the FUV band (mean flux from 1556 to 1576A)
            L_NUV in ergs/s/A:
                luminosity in the NUV band (mean flux from 2257 to 2277A)
        """
        return self._interpolate(metallicities, ages, masses)
