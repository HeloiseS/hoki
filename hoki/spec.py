"""
Utility functions related to observational and synthetic (BPASS) spectra
"""
import numpy as np
from numba import jit
import numbers
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from specutils.analysis import equivalent_width
from specutils import Spectrum1D
from astropy.units.quantity import Quantity

c = 299792.458

# TODO: Write test DispersionFromCaIR3

class DispersionFromCaIR3(object):
    """Pipeline to determine the dispersion of a stellar population from the Calcium triplet"""
      
    def __init__(self, wl, f, wl_bounds, fwhm_guess):
        """

        Parameters
        ----------
        wl: 1D array
            Wavelength
        f: 1D array
            Spectrum
        wl_bounds: list of lists or tuples (3 lists of 2 values)
            Bounds of the wavelength ranges for the three lines of the triple. There should be 3 lists, each lists
            containing two values marking the wavelengths on either side of the line.
        fwhm_guess: float
            Guess of the FWHM (just look at the plot)
        """
        self.wl = wl
        self.f = f

        # Create masks to select the lines of the triplet
        self.mask1 = (self.wl > wl_bounds[0][0]) & (self.wl < wl_bounds[0][1])
        self.mask2 = (self.wl > wl_bounds[1][0]) & (self.wl < wl_bounds[1][1])
        self.mask3 = (self.wl > wl_bounds[2][0]) & (self.wl < wl_bounds[2][1])

        # Find pseudo continuum for each line in the triplet
        self.pseudo1 = self.pseudo_cont(wl_bounds[0])
        self.pseudo2 = self.pseudo_cont(wl_bounds[1])
        self.pseudo3 = self.pseudo_cont(wl_bounds[2])

        # Remove the pseudo_continuum
        self.cont_removed1 = (self.f - self.pseudo1)[self.mask1]
        self.cont_removed2 = (self.f - self.pseudo2)[self.mask2]
        self.cont_removed3 = (self.f - self.pseudo3)[self.mask3]

        # Now we fit a Gaussian to each line with curve_fit
        self.popt1, __ = curve_fit(self._gaus, self.wl[self.mask1],
                                   -self.cont_removed1, p0=[1, np.mean(wl_bounds[0]), fwhm_guess])

        self.popt2, __ = curve_fit(self._gaus, self.wl[self.mask2],
                                   -self.cont_removed2, p0=[1, np.mean(wl_bounds[1]), fwhm_guess])

        self.popt3, __ = curve_fit(self._gaus, self.wl[self.mask3],
                                   -self.cont_removed3, p0=[1, np.mean(wl_bounds[2]), fwhm_guess])

        self.fit1 = self._gaus(self.wl[self.mask1], self.popt1[0], self.popt1[1], self.popt1[2])
        self.fit2 = self._gaus(self.wl[self.mask2], self.popt2[0], self.popt2[1], self.popt2[2])
        self.fit3 = self._gaus(self.wl[self.mask3], self.popt3[0], self.popt3[1], self.popt3[2])

        self.caIR3 = [8498, 8542, 8662]
        self.dispersion = None
        # center wavelengths for each line and fwhm as calculated by the fit
        self.centers_list = [self.popt1[1], self.popt2[1], self.popt3[1]]
        self.fwhms_list = [self.popt1[2], self.popt2[2], self.popt3[2]]

        self.calculate_dispersion()
        
        # TODO: add option to not plot
        self._plot()
        
    def pseudo_cont(self, wl_bounds):
        # TODO: write docstring
        # TODO: write test
        """ Calculates a pseudo continuum (linear)"""
        # index WL bounds
        i_lo = (np.abs(self.wl - wl_bounds[0])).argmin()
        i_hi = (np.abs(self.wl - wl_bounds[1])).argmin()

        # WL bounds
        wl_lo = self.wl[i_lo]
        wl_hi = self.wl[i_hi]

        # Corresponding F
        f_lo = self.f[i_lo]
        f_hi = self.f[i_hi]

        # pseudo_cont
        return np.interp(self.wl, [wl_lo, wl_hi], [f_lo, f_hi])

    def calculate_dispersion(self):
        """
        Calculates the dispersion from the Ca triplet gaussian fits

        Returns
        -------
        The mean of the dispersion values found for the three lines
        """
        # TODO: test (although I guess that's jsut testing teh class result)
        zs = [] # list of redshift values
        dispersions = [] # list of dispersion values
        recession_vels = [] # list of recession velocities

        # perform the same operation on all 3 lines
        for center, fwhm, emit in zip(self.centers_list, self.fwhms_list, self.caIR3):
            # TODO: refactor the names? If i need a damn comment it's not a good sign
            # center => wavelength at the center of line AS OBSERVED
            # emit => wavelength of line AT REST

            # We calculate the velocity corresponding to the wavelengths at the lower end, upper end, and center
            # of the line. The center_vel is the recessional velocity estimate, the other two are the
            # dispersion velocity estimates on either side of the line, that is the absolute value of the
            # difference between the recession velocity and the velocity calculated at that end.
            center_vel = self._velocity(center, emit)
            upper_vel = self._velocity(center+fwhm, emit)-center_vel
            lower_vel = center_vel-self._velocity(center-fwhm, emit)
            # note that upper and lower just refers to the upper and lower end in WAVELENGTH, the velocity values
            # are not expected to be either lower or higher -  it'll vary on a case by case basis.

            recession_vels.append(center_vel)
            zs.append((center-emit)/emit)
            dispersions.append(upper_vel)
            dispersions.append(lower_vel)

        self.z = np.mean(zs)
        self.recession_vel=np.mean(recession_vels)
        # the final estimate for the dispersion is the mean of all 6 values (3 lines with 2 values each).
        self.dispersion=np.mean(dispersions)

        return np.mean(dispersions)
    
    def _plot(self):
        f, ax = plt.subplots(ncols=3, figsize=(9,3))

        ax[0].plot(self.wl[self.mask1], self.cont_removed1)
        ax[0].plot(self.wl[self.mask1], -self.fit1)

        ax[1].plot(self.wl[self.mask2], self.cont_removed2)
        ax[1].plot(self.wl[self.mask2], -self.fit2)

        ax[2].plot(self.wl[self.mask3], self.cont_removed3)
        ax[2].plot(self.wl[self.mask3], -self.fit3)
        plt.suptitle(f'Dispersion: {int(self.dispersion)}km/s // z estimate: {round(self.z,4)}')
        plt.show()

    def _gaus(self, x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def _velocity(self, l_obs, l_emit):
        return ((l_obs - l_emit) / l_emit) * c


# TODO: Write test LickWizard
class LickWizard(object):
    """
    Wizard to calculate the Lick index for atomic lines in a given spectrum.
    """

    def __init__(self, wl, f):
        """
        Instanciate with the spectrum (wavelength and flux)

        Parameters
        ----------
        wl: 1D array
            Wavelength
        f: 1D array
            Spectrum
        """
        self.wl = wl
        self.f = f

    def calculate_index(self, line, cont1, cont2):
        """
        Method to calculate a specific index.

        Parameters
        ----------
        line: list or tuple of 2 elements
            Wavelength range defining the feature
        cont1: list or tuple of 2 elements
            Wavelength range defining the continuum BEFORE the feature
        cont2: list or tuple of 2 elements
            Wavelength range defining the continuum AFTER the feature

        Returns
        -------
        The value for the index
        """

        # We first create masks to isolate the wavelengths associated with the feature and the continuua region
        # on either side of it.
        self.mask_cont1 = (self.wl > cont1[0]) & (self.wl < cont1[1])
        self.mask_cont2 = (self.wl > cont2[0]) & (self.wl < cont2[1])
        self.mask_line = (self.wl > line[0]) & (self.wl < line[1])

        # Once those masks have been defined, the method below will do the shenanigans to calculate the pseudo-continuum
        self._calculate_continuum()
        # ... we then remove the continuum from the feature by dividing.
        spec_nocont = self.f[self.mask_line] / self.pseudo_cont

        # In order to take advantage of the equivalent width function from specutils we need to instantiate
        # a Spectrum1D object and shove our continuum removed data into it. Mind the units!
        self.spec1d = Spectrum1D(Quantity(spec_nocont, unit='erg/s/cm^2/A'),
                                 spectral_axis=Quantity(self.wl[self.mask_line], unit='Angstrom'))

        # finally the measure of our lick index is just the equivalent width of our feature.
        self.index = equivalent_width(self.spec1d)
        return self.index

    def _calculate_continuum(self):
        """ Calculates the pseudo continuum """

        # The pseudo continuum is calculated by draw a staight line from the midpoints of both cont1 and cont2

        # First we do a linear fit across the cont1 wavelengths
        self.m1, self.b1 = np.polyfit(self.wl[self.mask_cont1], self.f[self.mask_cont1], 1)
        # From that linear fit we take the mid point
        self.mid1_f = np.mean(self.m1 * self.wl[self.mask_cont1] + self.b1)  # the flux at midpoint
        self.mid1_wl = np.mean(self.wl[self.mask_cont1])  # the wavelength at midpoint

        # We repeat the process above for the second continuum region
        self.m2, self.b2 = np.polyfit(self.wl[self.mask_cont2], self.f[self.mask_cont2], 1)
        self.mid2_f = np.mean(self.m2 * self.wl[self.mask_cont2] + self.b2)
        self.mid2_wl = np.mean(self.wl[self.mask_cont2])

        # Then the pseudo continuum is a simple linear interpolation between the two midpoints calculated above
        # the attribute below can then be used by LickWizard to do the rest of the process.
        self.pseudo_cont = np.interp(self.wl[self.mask_line],
                                     [self.mid1_wl, self.mid2_wl],
                                     [self.mid1_f, self.mid2_f])


def dopcor(df, z, wl_col_index=0):
    """
    Basis doppler correction for hoki's dataframes

    Notes
    -----
    The correction is applied IN PLACE.
    """
    wl_dopcor = (df.iloc[:, wl_col_index].values) - (df.iloc[:, wl_col_index].values * z)
    df.iloc[:, wl_col_index] = wl_dopcor


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
