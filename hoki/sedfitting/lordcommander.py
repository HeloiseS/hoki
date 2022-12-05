import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

from hoki.utils.hoki_object import HokiObject


class LordCommander(HokiObject):
    """
    LordCommander is a pipeline to run the ppxf fits on a whole array of spectra (e.g. galaxy)
    It creates a bunch of tables (DataFrames) to store the data and then can save it to HDF5

    The Tables
    ----------
    SCALE_FACTOR:
        Contains the median of the observed spectra (used for the normalisation)
    DYNAMICS:
        Contains the best dispersion and Line Of Sight Velocity (LOSV)
    BEST_FIT:
        Contains the best fits from ppxf
    SFH:
        Contains the metalicity and ages of each of the templates selected to make the best fit
    MATCH_SPECTRA:
        Contains all the spectra used to create the best fit (that correspond to each row of SFH)
    MATCH_APOLY:
        Contains the additive polynomials needed to make the best fit - may be empty
    MATCH_MPOLY:
        Contains the multiplicative polynomials needed to make the best fit - may be empty
    CHI2:
        Contains the values of chi2 for each best fit
    FLAGS:
        Contains the flags. During the fitting procedure, flags are created when the chi2 value or
        dynamical parameters are higher than the median for the whole galaxy.
        => There are flags for deviations by 2, 3 and 5 standard deviations.
        The value of the flag is:
            - 2,3,5 for the Chi2
            - 20,30,50 for the LOSV
            - 200,300,500 for the dispersion.

        So a TOTAL flag with value 553 has a 3 sigma deviation in the Chi2, a 5 sigma diviation in the LOSV
        and a 5 sigma deviation in the dispersion.
    """

    def __init__(self, wl_obs, norm_fluxes, norm_noises, median_flux_ls, wl_fits, kvn):
        """
        Parameters
        ----------
        wl_obs: 1D array
            Observed data wavelength array
        norm_fluxes: 2D array
            "Normalised" fluxes (divided by median) - size N wl bins * N voronoi bins
        norm_noises: 2D array
            "Normalised" noise (actually divided by same median as the flux - same size as the fluxes
        median_flux_ls: 1D array
            The medians used to normalise the flux and noise
        wl_fits: 1D array
            The wavelegnth array for the fits in linear space from the log rebinned wavelength (np.exp(loglam))
        kvn: finalspace.KVN.kvn
            KVN object - templates already made!
        """
        self.wl_obs = wl_obs
        self.norm_fluxes = norm_fluxes
        self.norm_noises = norm_noises
        self.wl_fits = wl_fits
        self.kvn = kvn

        # ### MAKING SOME TABLES #### #


        ### SCALE_FACTOR ####
        self.SCALE_FACTOR = pd.DataFrame(np.array([np.array(median_flux_ls), np.array(median_flux_ls) * 1e-20]).T,
                                         columns=['median', 'to_ergs'])
        self.SCALE_FACTOR.index.name = 'bin_id'

        self.DYNAMICS = pd.DataFrame(columns=['los', 'disp'])
        # this requires to have done one fit first ... but that's okay. it's the best way to know whihc params to use
        # wl_fits = np.round(np.exp(loglamgalaxy),2)
        self.BEST_FIT = pd.DataFrame(columns=np.round(self.wl_fits, 2))
        self.SFH = pd.DataFrame(columns=['met', 'age', 'weights', 'bin_id'])
        self.MATCH_SPECTRA = pd.DataFrame(columns=np.round(self.wl_fits, 2))
        self.MATCH_APOLY = pd.DataFrame(columns=np.round(self.wl_fits, 2))
        self.MATCH_MPOLY = pd.DataFrame(columns=np.round(self.wl_fits, 2))
        self.REDDENING = None
        self.CHI2 = None
        self.FLAGS = None

        # the bin_id columns for the tables.
        # a bin_id column is added to table that has another main index (e.g. MATCH_SPECTRA will have more spectra
        # than there are voronoi bins so the main index will just be the lenght of the number of spectra in total)

        self.bin_id_sfh = []
        self.bin_id_spec = []
        self.bin_id_apoly = []
        self.bin_id_mpoly = []

        self.chi2_ls = []

    def run(self, start, moments, degree, vsyst, clean, goodpixels_func, reddening=None):
        """
        Fits the spectra in a big loop.

        Parameters
        ----------
        start: list of 2 values
            [LOSV_guess, dispersion_guess]
        moments: integer
            See ppxf manual
        degree: integer
            See ppxf manual
        vsyst: float
            See ppxf manual
        clean: bool
            Whether to do sigma clipping
        goodpixels_func: callable
            The function to determine the good pixels. It needs to take in a dictionary that contains they keys:
            `flux` and `wl`(Might expand that dictionary at a later date if I need more paramters).

        Returns
        -------
        None
        """
        i = 0
        eb_v=[]

        for norm_flux, norm_noise in tqdm(zip(self.norm_fluxes, self.norm_noises)):
            # find velscale and log rebin
            flux, loglamgalaxy, velscale = util.log_rebin([self.wl_obs[0], self.wl_obs[-1]], norm_flux)

            # sometimes nan values in noise, make them large noise values instead
            norm_noise = np.nan_to_num(norm_noise, nan=0.1)

            params_goodpixels_func = {'flux': flux, 'wl': self.wl_obs}
            goodpixels = goodpixels_func(params_goodpixels_func)

            # RUN PPXF ##
            pp = ppxf(self.kvn.templates, flux, norm_noise, velscale, start, goodpixels=goodpixels,
                      plot=False, moments=moments, degree=degree, vsyst=vsyst, clean=clean,
                      lam=self.wl_obs, quiet=True, reddening=reddening)
            
            # reddening if true
            if reddening is not None:
                eb_v.append(pp.reddening)

            self.kvn.make_results(pp)
            self.chi2_ls.append(self.kvn.ppxf.chi2)

            # RECORDS RESULTS
            los_vel, disp = self.kvn.ppxf.sol[0], self.kvn.ppxf.sol[1]
            # DYNAMICS: Table recording the dynamic information of the fits.
            self.DYNAMICS = self.DYNAMICS.append(pd.DataFrame([[los_vel, disp]], columns=['los', 'disp']))

            # BEST_FIT: table containing the full fits
            # made up of "matching spectra" according to the "weights" in SFH and the polynomial component if applicable

            self.BEST_FIT = self.BEST_FIT.append(pd.DataFrame([self.kvn.ppxf.bestfit], columns=self.wl_fits))

            # SFH: Table containing the ages weights and metalicities of each component
            self.SFH = self.SFH.append(self.kvn.results)
            self.bin_id_sfh += [i] * self.kvn.results.shape[
                0]  # record bin number / len of list == Num [ages, mets] needed

            # MATCH_SPECTRA: Table containing the spectra needed to create a matching "best_fit"
            self.MATCH_SPECTRA = self.MATCH_SPECTRA.append(pd.DataFrame(self.kvn.matching_spectra,
                                                                        columns=self.wl_fits))
            self.bin_id_spec += [i] * self.kvn.matching_spectra.shape[0]  # same as bins above

            # Polynomial components. Won't always be present so check if exist first.
            if self.kvn.matching_apolynomial is not None:
                self.MATCH_APOLY = self.MATCH_APOLY.append(pd.DataFrame([self.kvn.matching_apolynomial],
                                                                        columns=self.wl_fits))
                self.bin_id_apoly += [i]  # there is only every one polynomial

            if self.kvn.matching_mpolynomial is not None:
                self.MATCH_MPOLY = self.MATCH_MPOLY.append(pd.DataFrame([self.kvn.matching_mpolynomial],
                                                                        columns=self.wl_fits))
                self.bin_id_mpoly += [i]  # there is only every one polynomial
            i += 1
        
        # Chi squared table
        self.CHI2 = pd.DataFrame(np.array(self.chi2_ls), columns=['chi2'])
        # MAKE ERROR FLAGS
        self._make_flags()

        # CLEAN UP THE INDICES
        self.DYNAMICS.reset_index(inplace=True, drop=True)
        self.BEST_FIT.reset_index(inplace=True, drop=True)
        self.DYNAMICS.index.name = 'bin_id'
        self.BEST_FIT.index.name = 'bin_id'
        self.CHI2.index.name = 'bin_id'

        self.SFH = self._name_index_and_add_bin_id(self.SFH, self.bin_id_sfh)
        self.MATCH_SPECTRA = self._name_index_and_add_bin_id(self.MATCH_SPECTRA,  self.bin_id_spec)
        self.MATCH_APOLY = self._name_index_and_add_bin_id(self.MATCH_APOLY, self.bin_id_apoly)
        self.MATCH_MPOLY = self._name_index_and_add_bin_id(self.MATCH_MPOLY, self.bin_id_mpoly)
        
        # reddning table if relevant
        if reddening is not None:
            self.REDDENING = pd.DataFrame(np.array(eb_v), columns=['E(B-V)'])
            self.REDDENING.index.name = 'bin_id'

    def _make_flags(self):
        """ Creates the FLAG table """
        chi2_med, chi2_std = np.median(self.chi2_ls), np.std(self.chi2_ls)
        los_med, los_std = np.median(self.DYNAMICS.los), np.std(self.DYNAMICS.los)
        disp_med, disp_std  = np.median(self.DYNAMICS.disp), np.std(self.DYNAMICS.disp)

        # TODO: the logic below could be a function

        ### Finds which values are MORE THAN 2 3 or 5 std away from the median -> True or False
        CHI2_2_SIGMA = np.array([1 if chi2 > chi2_med + 2 * chi2_std else 0 for chi2 in self.chi2_ls])
        CHI2_3_SIGMA = np.array([1 if chi2 > chi2_med + 3 * chi2_std else 0 for chi2 in self.chi2_ls])
        CHI2_5_SIGMA = np.array([1 if chi2 > chi2_med + 5 * chi2_std else 0 for chi2 in self.chi2_ls])

        LOS_2_SIGMA = np.array([1 if np.abs(los - los_med) > 2 * los_std else 0 for los in self.DYNAMICS.los])
        LOS_3_SIGMA = np.array([1 if np.abs(los - los_med) > 3 * los_std else 0 for los in self.DYNAMICS.los])
        LOS_5_SIGMA = np.array([1 if np.abs(los - los_med) > 5 * los_std else 0 for los in self.DYNAMICS.los])

        DISP_2_SIGMA = np.array([1 if np.abs(disp - disp_med) > 2 * disp_std else 0 for disp in self.DYNAMICS.disp])
        DISP_3_SIGMA = np.array([1 if np.abs(disp - disp_med) > 3 * disp_std else 0 for disp in self.DYNAMICS.disp])
        DISP_5_SIGMA = np.array([1 if np.abs(disp - disp_med) > 5 * disp_std else 0 for disp in self.DYNAMICS.disp])

        # turns the boolean arrays into the flags
        CHI2_FLAG = [5 if chi2_5 == 1 else 3 if chi2_3 == 1 else 2 if chi2_2 == 1 else 0 for chi2_5, chi2_3, chi2_2 in
                     np.array([CHI2_5_SIGMA, CHI2_3_SIGMA, CHI2_2_SIGMA]).T]
        LOS_FLAG = [50 if los_5 == 1 else 30 if los_3 == 1 else 20 if los_2 == 1 else 0 for los_5, los_3, los_2 in
                    np.array([LOS_5_SIGMA, LOS_3_SIGMA, LOS_2_SIGMA]).T]
        DISP_FLAG = [500 if los_5 == 1 else 300 if los_3 == 1 else 200 if los_2 == 1 else 0 for los_5, los_3, los_2 in
                     np.array([DISP_5_SIGMA, DISP_3_SIGMA, DISP_2_SIGMA]).T]

        # Puts flags in the table and calculates the TOTAL flag which is the sum of the flags.
        self.FLAGS = pd.DataFrame(np.array([CHI2_FLAG, LOS_FLAG, DISP_FLAG]).T,
                             columns=['CHI2', 'LOS', 'DISP'])
        self.FLAGS['TOTAL'] = self.FLAGS.CHI2 + self.FLAGS.LOS + self.FLAGS.DISP
        self.FLAGS.index.name = 'bin_id'

    def _name_index_and_add_bin_id(self, df, bin_id):
        """ For tables where the main index isn't bin_id, names the main index spec_id and adds bin_id column"""
        df.reset_index(inplace=True, drop=True)
        df.index.name='spec_id'
        df['bin_id']=bin_id
        return df

    def save(self, filepath, folder):
        """
        Saves data to HDF5 file.

        Parameters
        ----------
        filepath: str
            path to the file
        folder: str
            path WITHIN the file

        Returns
        -------
        None
        """
        self.DYNAMICS.to_hdf(f'{filepath}', key=f'{folder}/DYNAMICS', mode='a')
        self.BEST_FIT.to_hdf(f'{filepath}', key=f'{folder}/BEST_FIT', mode='a')
        self.SFH.to_hdf(f'{filepath}', key=f'{folder}/SFH', mode='a')
        self.MATCH_SPECTRA.to_hdf(f'{filepath}', key=f'{folder}/MATCH_SPECTRA', mode='a')
        self.MATCH_APOLY.to_hdf(f'{filepath}', key=f'{folder}/MATCH_APOLY', mode='a')
        self.MATCH_MPOLY.to_hdf(f'{filepath}', key=f'{folder}/MATCH_MPOLY', mode='a')
        self.SCALE_FACTOR.to_hdf(f'{filepath}', key=f'{folder}/SCALE_FACTOR', mode='a')
        self.FLAGS.to_hdf(f'{filepath}', key=f'{folder}/FLAGS', mode='a')
        self.CHI2.to_hdf(f'{filepath}', key=f'{folder}/CHI2', mode='a')
        
        if self.REDDENING is not None:
            self.REDDENING.to_hdf(f'{filepath}', key=f'{folder}/REDDENING', mode='a')

        file = h5py.File(filepath, 'r+')

        file[f'{folder}'].attrs['TITLE'] = 'Test data for the new schema'
        file[f'{folder}'].attrs['BEST_FIT'] = 'Table containing the best fit from ppxf - They are a combination of ' \
                                              'several templates, a polynomial (optional) as well as the dynamical ' \
                                              'information (line of sight velocity and dispersion). ' \
                                              'One spectrum per voronoi bin.'

        file[f'{folder}'].attrs['DYNAMICS'] = 'Dynamical information retrieved from the fits (Line of sight velocity ' \
                                              'and dispersion). One row per voronoi bin.'

        file[f'{folder}'].attrs['MATCH_APOLY'] = 'The additive polynomial needed for a match (optional)'

        file[f'{folder}'].attrs['MATCH_MPOLY'] = 'The multiplicative polynomial needed for a match (optional)'

        file[f'{folder}'].attrs['MATCH_SPECTRA'] = 'The spectra needed for a match. One or several spectra ' \
                                                   'per voronoi bin (bin_id). They still have a unique id (spec_id). '\
                                                   'The contain the dynamical information'

        file[f'{folder}'].attrs['SFH'] = 'Table containing the metallicity, age or each match_spectra ' \
                                         'and its corresponding weight'

        file[f'{folder}'].attrs['SCALE_FACTOR'] = 'Table containing the scale factors used. The median is what I ' \
                                                  'used to normalise. The "to_ergs" is the median*1e-20 which is ' \
                                                  'contains the MUSE offset to turn the data to erg/s/cm2/A. ' \
                                                  'Multiplying the spectra by "to_ergs" therefore makes them ' \
                                                  'into observer units!'

        file[f'{folder}'].attrs['FLAGS'] = 'Table flagging potential problems with the fit - each column tells ' \
                                           'the number of sigma deviations from the median of the whole sample ' \
                                           'in chi squared (ones), line of sight velocity (tens) ' \
                                           'and dispersion (hundreds)'

        file[f'{folder}'].attrs['CHI2'] = 'Table of the Chi squared values for each of the BESTFIT spectra calc by ppxf'
        
        file[f'{folder}'].attrs['REDDENING'] = 'The reddening fit by ppxf if I asked it to do it'
