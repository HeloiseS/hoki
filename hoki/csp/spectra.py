"""
Object to calculate the spectra at a certain
lookback time or over a binned lookback time
"""

import hoki.csp.utils as utils
from hoki.utils.hoki_object import HokiObject
from hoki.constants import *
import hoki.csp.utils as utils


# TODO add IMF selection

class CSPSpectra(HokiObject, utils.CSP):
    """Object to calculate synthetic spectra with complex stellar formation histories.

    Parameters
    ----------
    data_folder : `str`
        folder containing the BPASS data files
    binary : boolean
        If `True`, loads the binary files. Otherwise, just loads single stars.
        Default=True
    """
    def __init__(self, data_folder, imf, binary=True):
        self.bpass_spectra = utils._normalise_spectrum(utils.load_spectra(data_folder, imf, binary=binary))

    def calculate_spec_over_time(self,
                                 sfh_functions,
                                 Z_functions,
                                 nr_bins,
                                 return_edges = False
                                 ):
        """
        Calculates spectra over lookback time.

        Parameters
        ----------
        sfh_functions : `list(function, )`
            An array containing the stellar formation histories functions
            in units M_\\odot per yr over lookback time.
        Z_functions : `list(function, )`
            An array containing functions describing the metallicity over
            lookback time.
        nr_bins : `int`
            The number of bins to split the lookback time into.
        return_edges : `bool`
            If `True`, also returns the edges of the lookback time bins.
            Default=False

        Returns
        -------
        `numpy.ndarray`
            If `return_edges=False`, returns a numpy array containing a
            spectrum per bin.
            If `return_edges=True`, returns a numpy array containing a spectrum
            per bin and the bin edges, eg. `out[0]=event_rates` `out[1]=time_edges`.
        """
        self._type_check_histories(sfh_functions, Z_functions)
        nr_sfh = len(sfh_functions)
        time_edges = np.linspace(0, self.now, nr_bins+1)

        spectra = np.zeros((nr_sfh, nr_bins, 100000), dtype=np.float64)

        mass_per_bin = np.array([utils.mass_per_bin(i, time_edges) for i in sfh_functions])
        metallicity_per_bin = np.array([utils.metallicity_per_bin(i, time_edges) for i in Z_functions])

        # make numpy array with usage [age][metallicity][wavelength]
        # for faster access
        np_spectra = self.bpass_spectra.to_numpy().reshape(51, 13, 100000)

        for counter, (mass, Z) in enumerate(zip(mass_per_bin,  metallicity_per_bin)):
            spec = utils._over_time(Z,
                                    mass,
                                    time_edges,
                                    self.bpass_spectra[count].T.to_numpy())

            spectra[counter, :, count-1] = spec/np.diff(time_edges)

        if return_edges:
            return np.array([spectra, time_edges])
        else:
            return spectra

    def calculate_spec_at_time(self,
                               sfh_functions,
                               Z_functions,
                               t,
                               sample_rate=None
                               ):
        """
        Calculates the spectrum at lookback time `t`.

        Parameters
        ----------
        sfh_functions : `list(function, )`
            An array containing the stellar formation histories functions
            in units M_\\odot per yr over lookback time.
        Z_functions : `list(function, )`
            An array containing functions describing the metallicity over
            lookback time.
        t : `float`
            The lookback time, where to calculate the event rate.
        sample_rate : `int`
            The number of samples to take from the SFH and metallicity evolutions.
            Default = None. Uses the BPASS bins.

        Returns
        -------
        float
            Returns the event rate at the given lookback time `t`.
        """

        self._type_check_histories(sfh_functions, Z_functions)

        # The setup of these elements could almost all be combined into a function
        # with code that's repeated above. Similarly, with the event rate calculation.
        nr_sfh = len(sfh_functions)

        spectrum = np.zeros((nr_sfh, 100000), dtype=np.float64)

        if sample_rate is None:
            time_edges = BPASS_LINEAR_TIME_EDGES
        else:
            time_edges = np.linspace(0, 13.8e9, sample_rate+1)

        mass_per_bin = np.array([utils.mass_per_bin(i, time_edges) for i in sfh_functions])
        metallicity_per_bin = np.array([utils.metallicity_per_bin(i, time_edges)
                                                for i in Z_functions])

        np_spectra = self.bpass_spectra.to_numpy().reshape(51, 13, 100000)

        for counter, (mass, Z) in enumerate(zip(mass_per_bin, metallicity_per_bin)):
            spectrum[counter] = utils._at_time_spectrum(Z,
                                                        mass,
                                                        time_edges,
                                                        np_spectra
                                                        )

        return spectrum
