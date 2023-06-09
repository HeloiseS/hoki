"""
KVN helper class
----------------
Author: H. F. Stevance
e-mail: hfstevance@gmail.com

/!\ IS BEING DEPRECATED AND REPLACED BY PPXF WIZARD /!\

This needs to stay in hoki.sedfitting because it is what I used for the Nature Astro paper
Stevance et al. 2023 and if I accidentally introduce (or fix) a bug during the refactor the results
will no longer be reproducable.

Some minor changes were made on this code upon release of hoki v1.7 (December 2022)
- KVN.template_properties replaced by KVN.template_properties_BROKEN
- KVN.template_properties_ppxf_order replaced by KVN.template_properties
- KVN header removed from verbose mode altogether. Still defined for posterity and nostalgia.

"""


# classic python
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from scipy import ndimage

# Other astro packages
from astropy.io import fits
from ppxf import ppxf_util
from ppxf.ppxf import ppxf, losvd_rfft, rebin
import ppxf as ppxf_package

# My packages
from hoki import load
from hoki.constants import BPASS_TIME_BINS, BPASS_METALLICITIES, BPASS_NUM_METALLICITIES, BPASS_TIME_INTERVALS
from hoki.utils.hoki_object import HokiObject
from hoki.utils.hoki_dialogue import dialogue
from hoki.utils.exceptions import HokiFatalError
from hoki.utils.progressbar import print_progress_bar

plt.style.use('hfs')

####  Some useful definitions

#__all__ = ['KVN']

met_to_num={'zem5':1e-5,  'zem4':1e-4, 'z001':1e-3, 'z002':2e-3,'z003':3e-3,'z004':4e-3,'z006':6e-3,'z008':8e-3,
            'z010':1e-2,'z014':1.4e-2,'z020':2e-2,'z030':3e-2, 'z040':4e-2}

c = 299792.458

kvn_header=" \n"+ \
"      ___                        ___     \n"+ \
"     /__/|          ___         /__/\    \n"+ \
"    |  |:|         /__/\        \  \:\   \n"+ \
"    |  |:|         \  \:\        \  \:\  \n"+ \
"  __|  |:|          \  \:\   _____\__\:\ \n"+ \
" /__/\_|:|____  ___  \__\:\ /__/::::::::\ \n"+ \
" \  \:\/:::::/ /__/\ |  |:| \  \:\~~\~~\/\n"+ \
"  \  \::/~~~~  \  \:\|  |:|  \  \:\  ~~~ \n"+ \
"   \  \:\       \  \:\__|:|   \  \:\     \n"+ \
"    \  \:\       \__\::::/     \  \:\    \n"+ \
"     \__\/           ~~~~       \__\/  v2.1  \n"


#### TESTS TO DO ########
# TODO: kvn with full time res
# TODO: kvn with half time res
# TODO: kvn with full time res and select age columns
# TODO: kvn with half time res and select age columns
# TODO: need to test that the spectra in the ppxf matrix from teh matching indices are consistent with BPASS spectra of matchign indices
# see `tracing_bug_in_kvn_make_results.ipynb`


class KVN(HokiObject):
    """
    Kevin - my pPXF helper

    Attributes
    ----------
    model_path : str
        Path to the BPASS model outputs where the BPASS SED files are located
    bpass_list_spectra: [str]
        List of BPASS SED file paths
    z_list : [str]
        List of metallicities to include in the templates (BPASS string names)
    num_z_list : [float]
        List of metallicities to include in the templates (numerical)
    wl_obs : 1D array, optional
        (Linear) Wavelength array of the observational data. If given, don't need log_wl_obs
    log_wl_obs : 1D array, optional
        (Natural Logarithm) Wavelength array of the observational data. If given, don't need wl_obs
    wl_range_tem : np.array
        Minimum and Maximum wavelength in the template SEDs. Padding is applied to these boundaries.
    wl_tem : 1D np.array
        (Linear) Wavelength array of the template SEDs.
    fwhm_tem : float
        Resolution element of the templates (it's 1 A in BPASS v2.2.1 unless you're doing something fancy).
        It's not technically a FWHM but it's the theoretical equivalent to the resolution element calculated
        for the galaxy at some point in this code so I named it that for visual consistency.
    velscale_obs : float
        Velocity scale of the observed spectrum
    fwhm_obs : float
        Resolution element of the observed spectrum calculated from dispersion_obs
    fwhm_dif : float
        Difference in the resolution element of the observation and templates (subtracted in quadrature)
    sigma : float
        fwhm_dif/2.355/fwhm_tem
    log_age_cols : [float]
        list of log ages to use from BPASS SEDs
    templates : 3D np.array
        Template SEDs. The dimensions are [wavelength, age, metallicity]
    _template_properties_BROKEN : 2D np.array
        THIS IS THE ONE THAT IS DEPRECATED.
    template_properties : 2D np.array
        Records the weights from ppxf (MAYBE SHOULDN'T BE CALLED PROPERTIES). Is the fixed version of template_properties
        after I realised that it was in the wrong order compared to ppxf (see BUG FIX note in code).
    ppxf : ppxf.ppxf.ppxf
        The ppxf object returned from fitting with PPXF. It contains a whole load of information that is worth
        saving. See the ppxf manual for more detail.
    matching_indices : 1D np.array
        Indices in template_properties corresponding to the sepctra in `matching_raw_spectra`, `matching_spectra`
    matching_raw_spectra : 2D np.array
        The 'raw' templates as calculated by KVN during `make_templates` that are used in SED fit. Shape: [N spectra,WL]
    matching_spectra : 2D np.array
        The same matching templates as in `matching_raw_spectra` with the kinematic information added in
        (recession velocity and dispersion).
    matching_apolynomial : 1D np.array
        Additive polynomial fitted by ppxf (if you asked for it, otherwise None)
    matching_mpolynomial : 1D np.array
        Multiplicative polynomial fitted by ppxf (if you asked for it otherwise None).
    results : pandas.DataFrame

    Methods
    -------
    make_templates : None
        Returns none but creates `templates`, `template_properties` (and above)
    make_results : None
        Returns none but creates `results`, `matching_indices`, `matching_raw_spectra`, `matching_spectra`,
        `matching_mpolynomial`, `matching_mpolynomial`, `ppxf`
    save : None
        Creates a pickle file of the KVN object
    load : None
        Loads in a pickle file of a previously created KVN object

    Example
    -------
    Below is a very quick 'cheat sheet' of the steps to take to create and fit with KVN

    # Create the templates
    >>> recessional_vel = z*c # z is the redshift and c the speed of light
    >>> WL = (WL - WL*z)
    >>> flux, wl_log_bin, velscale = ppxf.ppxf_util.log_rebin([WL[0], WL[-1]], F)
    >>> wl_fits = np.exp(wl_log_bin)
    >>> kvn=hsed.KVN()
    >>> kvn.make_templates(BPASS_MODEL_PATH,
                           fwhm_obs=1.25, # delta lambda or resolution element observations
                           wl_obs=WL, # Wavelength bins of the observations (not log rebinned)
                           wl_range_obs=[WL[0], WL[-1]], # wavelength range of the observations
                           velscale_obs=velscale,
                           wl_range_padding=[-50,50], # number of extra Angstroms befrore and after the range
                           z_list=['z010','z020','z030'],
                           verbose=True,
                           )

    # fit with ppxf
    >>> start = [0, 180]
    >>> goodpixels = np.argwhere((flux < FLUXMAX) & (flux > FLUXMIN) # Flux condition
                                 & ((WL < WLMIN) | (WL > WLMAX)) # Wavelength condition
                                 ).T[0] # Make it a 1D horizontal vector
    >>> dv = c*np.log(kvn.wl_tem[0]/kvn.wl_obs[0])
    >>> ppxf_object  = ppxf.ppxf.ppxf(kvn.templates,           # these are the BPASS SED templates
                                      flux,                    # observed flux vector
                                      noise,                   # the standard deviation vector
                                      velscale,                # the velocity scale calculated by the log rebin ppxf utility
                                      start,                   # the start guesses for recession vel and dispersion
                                      goodpixels=goodpixels,   # our goodpixels
                                      plot=False,              # I set this to false because I made my own in hoki.sedfitting
                                      moments=4,               # order of the Gauss-Hermit moment to fit (see above)
                                      degree=2,                # order of the polynomial to fit on top of the SED mixture
                                      vsyst=dv,                # see above
                                      clean=False,             # whether to perform sigma clipping and itterative fitting
                                      lam=WL                   # the observed wavelength vector
                                     )

    >>> res = hoki.sedfitting.ppxf_plot_fixed(ppxf_object, wl_fits=wl_fits, WL=WL, F=F)
    >>> kvn.make_results(ppxf_boject)
    >>> kvn.results

    Notes
    -----
    It's important to know which wavelength scale to use to plot which spectra - below a summary table.

    |    WL scale   |       Plotted spectra      |
    | ------------- | -------------------------- |
    |  `kvn.wl_tem` | `kvn.matching_raw_spectra` |
    |    `wl_fits`  | `kvn.matching_spectra`     |
    |    `wl_fits`  | `kvn.matching_apolynomial` |
    |    `wl_fits`  | `kvn.matching_mpolynomial` |
    """

    # TODO: remove? this is a legacy of a test function that i didn't implement in the end - i just rebinned the files
    def halve_time_res_spectra(self, filename):
        spec = load.model_output(filename)
        WL = spec.WL
        spectra = spec.drop(['WL'], axis=1)
        # step 1
        spectra_weighted = spectra.multiply(BPASS_TIME_INTERVALS, axis=1)
        # step 2
        summed_spectra = spectra_weighted.values[:, :-1:2] + spectra_weighted.values[:, 1::2]
        # step 3
        # calculating Delta t_j
        new_bins_weights = BPASS_TIME_INTERVALS[:-1:2] + BPASS_TIME_INTERVALS[1::2]
        # weighting
        new_spectra = summed_spectra / new_bins_weights
        # making a data frame
        new_spec_df = pd.DataFrame(new_spectra, columns=np.round(BPASS_TIME_BINS[:-1:2], 2).astype('str'))
        new_spec_df = pd.concat([pd.DataFrame(WL, columns=['WL']), new_spec_df], axis=1)
        new_spec_df.to_csv(filename[-31:], sep=' ', index=False)

    def make_templates(self, # In the refactor this should be split into several functions
                       path_bpass_spectra,
                       wl_obs=None,
                       log_wl_obs=None,
                       fwhm_obs=None,
                       dispersion_obs=None,
                       wl_range_obs=None,
                       wl_range_padding=[-1,1],
                       velscale_obs=None,
                       binary=True,
                       single=False,
                       z_list=None,
                       oversample=1,
                       log_age_cols=None,
                       _max_age=10.2,
                       verbose=True
                      ):
        """
        Makes the templates

        Parameters
        ----------
        path_bpass_spectra : str
            Location of the folder where the bpass spectra are located
        wl_obs : 1D array, optional
            (Linear) Wavelength array of the observational data. If given, don't need log_wl_obs
        log_wl_obs : 1D array, optional
            (Natural Logarithm) Wavelength array of the observational data. If given, don't need wl_obs
        fwhm_obs : float, optional
            FWHM of the observations. Applies if the dispersion is not wavelength dependent.
        dispersion_obs : 1D array, optional
            Dispersion array. Same size ase wl_obs of log_wl_obs. Applies when the dispersion is wavelength dependent.
        wl_range_obs : [min, max], optional
            Wavelength range of the observations, optional
        wl_range_padding : [-X,Y], optional
            Padding given to the templates on creation. The template wl range will be [wl_range_obs-X, wl_range_obs+Y].
            Default is [-1,1]
        binary : bool, optional
            Whether to include the binary model spectra. Default is True
        single : bool, optional
            Whether to include the single star model spectra. Default is False.
        z_list : list of str, optional
             Which metallicities to consider, e.g. ['z020', 'z014'].
             Default is None. In which case all 13 BPASS metallicites are included.
        oversample : int, optional
            The oversample keyword in ppxf
        log_age_cols : list of valid BPASS log ages
            The columns to include from the BPASS spectra dataframes. NOT REBINNED
        _max_age_index : int from 1 to 51, optional
            Maximum age index to include. Default is 42 which corresponds to a Hubble time.
        verbose : bool
            whether to print the text

        Returns
        -------
        None. But it creates the following: # TODO: finish this section
        """
        if verbose:
            print(f"{dialogue.info()} TemplateMaker Starting")
            print(f"{dialogue.running()} Initial Checks")

        # Does the path exist? GLOB DOESN'T CHECK IT FOR US
        assert os.path.exists(path_bpass_spectra), "HOKI ERROR: The path to the BPASS spectra is incorrect."

        self.model_path = path_bpass_spectra

        self.log_age_cols = log_age_cols

        ### TODO: put this in a function and make bpass_list_spectra a cached property? (Make binray and single priv.att)
        # Making list of relevant spectra files
        if binary and single:
            self.bpass_list_spectra = glob.glob(self.model_path + 'spectra*')
        elif binary and not single:
            self.bpass_list_spectra = glob.glob(self.model_path  + 'spectra*bin*')
        elif single and not binary:
            self.bpass_list_spectra = glob.glob(self.model_path  + 'spectra*sin*')
        else:
            raise HokiFatalError(f"binary and single set to False \n\n{dialogue.debugger()} "
                                 f"You must choose whether to include binary models, single models or both"
                                 f"\nAt least one of the following parameters must be True  when you instanciate:"
                                 f"\nbinary; single")

        self.bpass_list_spectra.sort()

        ### TODO: also make this a cached property (z_list will need to be a private attribute we can use in the func)
        # Allow user to select a list of metallicities?
        if z_list is not None:
            self.z_list=z_list
            self.num_z_list=[met_to_num.get(key) for key in z_list]
        else:
            self.z_list=BPASS_METALLICITIES
            self.num_z_list=BPASS_NUM_METALLICITIES


        not_you=[]
        for filepath in self.bpass_list_spectra:
            if filepath[-8:-4] not in self.z_list:
                not_you.append(filepath)

        self.bpass_list_spectra = list(set(self.bpass_list_spectra)-set(not_you))
        self.bpass_list_spectra.sort() # need to sort again so metallicities are in ascending order consistently
        # TODO: make a mask: if {met} in "path" and apply to bpass_list_spectra

        #self.wl_range_tem = np.array((10**self.log_wl_obs)[[0,-1]].astype(int))
        if wl_obs is None and log_wl_obs is None:
            raise HokiFatalError(f"wavelength of observational data not provided\n\n{dialogue.debugger()}"
                                 f"At least one of the following parameters must be provided when you instanciate:"
                                 f"\nwl_obs (wavelength arr in linear space); "
                                 f"log_wl_obs (wavelength arr in log10 space)")
        elif wl_obs is not None and log_wl_obs is None:
            self.wl_obs = wl_obs
            self.log_wl_obs = np.log(self.wl_obs)
            self.wl_range_tem = np.array(self.wl_obs[[0,-1]].astype(int))

        elif log_wl_obs is not None and wl_obs is None:
            self.log_wl_obs = log_wl_obs
            self.wl_obs=np.e**self.log_wl_obs
            self.wl_range_tem = np.array(self.wl_obs[[0,-1]].astype(int))

        ### just to be sure we have enough wavelength coverage
        self.wl_range_tem+=[1,1] # shifted by one angstrom otherwise?
        self.wl_range_tem+=wl_range_padding

        if verbose: print(f"{dialogue.running()} Loading  model spectrum")

        # Load one spectrum
        _ssp=load.model_output(self.bpass_list_spectra[0])
        _ssp.index=_ssp.WL # turning  the WL column into an index so crop the dataframe easily with WL
        _ssp.drop('WL', axis=1, inplace=True)

        self.wl_tem = np.arange(self.wl_range_tem[0], self.wl_range_tem[-1]+1) # is this the same as ssp.WL?

        self._ssp60=_ssp.loc[self.wl_range_tem[0]:self.wl_range_tem[-1]]['6.0'].values
        #one spectrum at log(age)=6.0: needed to get log_wl_template, velscale_template

        self.fwhm_tem = 1 # known for BPASS, res is 1 Angstrom

        ## If disperion given
        if dispersion_obs is not None:
            if verbose: print(f"{dialogue.running()} Calulating obs. velocity scale")
            # Calc. velocity scale
            _frac = self.wl_obs[1]/self.wl_obs[0]
            _dwl_obs = (_frac - 1)*self.wl_obs
            self.velscale_obs = np.log(_frac)*c #speed of light # Velocity scale in km/s per pixel

            if verbose: print(f"{dialogue.running()} Calulating FWHM")
            # Calc. FWHM galaxy (res.)
            _fwhm_obs = 2.355*dispersion_obs*_dwl_obs
            self.fwhm_obs = np.interp(self.wl_tem, self.wl_obs, _fwhm_obs)
            self.fwhm_dif = np.sqrt((self.fwhm_obs**2 - self.fwhm_tem**2).clip(0))

        elif dispersion_obs is None and fwhm_obs is not None:
            self.wl_range_obs = wl_range_obs
            self.fwhm_obs = fwhm_obs
            if verbose: print(f"{dialogue.running()} Calulating obs. velocity scale -- No dispersion")
            __, __, self.velscale_obs = ppxf_util.log_rebin(self.wl_range_obs, self.wl_obs)
            #self.velscale_obs=c*np.log(self.wl_obs[1]/self.wl_obs[0])
            self.fwhm_dif = np.sqrt((self.fwhm_obs**2 - self.fwhm_tem**2))

        elif dispersion_obs is None and fwhm_obs is None:
            raise HokiFatalError(f"dispersion_obs and fwhm_obs are both None \n\n{dialogue.debugger()}\
                    \nAt least one of the following parameters must be provided when you instanciate:\
                    \ndispersion_obs; fwhm_obs")

        if velscale_obs is not None:
            self.velscale_obs = velscale_obs


        if verbose: print(f"{dialogue.running()} Calculating template wavelength (log rebin) and velocity scale")

        _ssp_temp, self.log_wl_tem, self.velscale_tem = ppxf_util.log_rebin(list(self.wl_range_tem),
                                                                            self._ssp60, oversample=oversample,
                                                                            # introduced a severe bug where the
                                                                            # templates wouldn't have the right
                                                                            # size for the observations
                                                                            #
                                                                            velscale=self.velscale_obs#/self.velscale_ratio
                                                                            )

        self._max_age_index = np.argwhere(_ssp.columns[1:].values.astype(float)>=_max_age)[0][0]+1
        if verbose: print(f"{dialogue.running()} Calculating sigma")

        self.sigma = self.fwhm_dif/2.355/self.fwhm_tem

        if verbose: print(f"{dialogue.running()} Instanciating container arrays")

        # # WL bins, # Ages, # mets
        if self.log_age_cols is not None:
            if verbose: print(f"{dialogue.info()} Using your custom set of ages")
            # instanciate empty arrays
            self.templates = np.empty((_ssp_temp.size, len(self.log_age_cols), len(self.bpass_list_spectra)))

            # TODO: this will need a try and except for when we have an array instead of a list or if log_age_cols is just wrong (e.g. bad log ages)
            l = len(self.bpass_list_spectra)*len(self.log_age_cols)

            # Record the index of each bpass time bin that is contained in the list of age columns we are actually interested in
            # this is a bit convoluted and doesn't use the simple formulat (log_age-6)*10 because doesn't work at half time res
            AGE_INDICES = np.argwhere(np.isin(_ssp.columns.values.astype('float'), log_age_cols)).T[0]

            # needed to be strings to compare to the _ssp columns but will need to be numbers later on
            # - what a cluster fuck
            self.log_age_cols=np.array(self.log_age_cols).astype(float)

        elif self.log_age_cols is None:
            if verbose: print(f"{dialogue.info()} Using all ages from 6.0 to {_max_age} (included) - log_age_cols now set")
            # instanciate empty arrays
            self.templates = np.empty((_ssp_temp.size, self._max_age_index, len(self.bpass_list_spectra)))

            # length of the template properties array
            l = int(len(self.bpass_list_spectra)*self._max_age_index)

            # indices used for the j/age dimension
            AGE_INDICES = range(self._max_age_index)
            # this is a clusterfuck of a line...
            # it takes the columns of a bpass spectra dataframe, converts the first col is WL, not an age
            self.log_age_cols = np.round(_ssp.columns.to_numpy()[1:self._max_age_index].astype(float),2)

        self._template_properties_BROKEN =  np.zeros((l, 2)) # has shape Num_met*Num_ages x 2

        if verbose: print(f"{dialogue.running()} Compiling your templates")

        i = 0

        # index k (for the metallicity location in stars_tempaltes), and file path
        for k, path in enumerate(self.bpass_list_spectra):
            # string to identify the metallicity, like 'z040' above
            met = self.num_z_list[np.argwhere(np.array(self.z_list)==path[-8:-4])[0][0]]

            # load the file corresponding to that metallicity
            # getting the SSPs for metallciity at index k
            ssps_k=load.model_output(path)
            ssps_k.index=ssps_k.WL # make the wavelength the index, it makes things much easier
            ssps_k.drop('WL', axis=1, inplace=True) # drop the now unnecessary column
            cropped_ssps_k=ssps_k.loc[self.wl_range_tem[0]:self.wl_range_tem[-1]]

            # index j (age), and column '[age]' - only iterate till the current age of the universe
            for j, col in enumerate(cropped_ssps_k.columns[AGE_INDICES]):
                # get the simple star population for that age
                ssp_jk = cropped_ssps_k[col].values
                # gaussian filter to change the resolution of the OG template (I think... check that)
                if isinstance(self.sigma, float):
                    ssp_jk = ndimage.gaussian_filter1d(ssp_jk, self.sigma)
                else:
                    ssp_jk = ppxf_util.gaussian_filter1d(ssp_jk, self.sigma)
                    # not the scipy convolution because it can't do variable sigma


                # logarithm binning
                sspNew, self.log_wl_tem, __ = ppxf_util.log_rebin(self.wl_range_tem,
                                                     ssp_jk,
                                                     velscale=self.velscale_obs,
                                                     # introduced a severe bug where the templates wouldn't have
                                                     # the right size fore the observations
                                                     # velscale=self.velscale_obs/self.velscale_ratio
                                                     )


                # sspNew = ssp_jk
                # add to the template array
                self.templates[:, j, k] = sspNew/np.median(sspNew) # the templates are normalised
                self._template_properties_BROKEN[i, :] = [met, float(col)]
                i+=1

                print_progress_bar(i, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

            self.wl_tem = np.exp(self.log_wl_tem)

        # TODO: refactor
        ## Some details about the code below (Oct 20th 2021)
        # I realised the order in which the _template_properties_BROKEN was recording thing
        # was scrambled compared to ppxf (which isn't obvious from the manual tbf!)
        # ppxf.weights packet the ages toghether whereas i packet the metallicities together
        # since my for loop above loops over the different filenames for each metallicities
        # a.k.a ppxf order is: [[age1 met1], [age1 met2, [age1 met3], [age2 met1], etc....]
        # and my order is: [[age1 met1], [age2 met1, [age3 met1], [age1 met2], etc...]
        # i am not deleting the _template_properties_BROKEN attribute cuz i'm tired and i don't
        # want to accidentally break something - instead i'm adding a new attribute that recreates ppxf order form mine
        self.template_properties = []
        for a in np.unique(self._template_properties_BROKEN[:, 1]):
            for z in np.unique(self._template_properties_BROKEN[:, 0]):
                self.template_properties.append([z, a])

        self.template_properties = np.array(self.template_properties)
        ####  BUG FIX END oct 20 2021 ###

        if verbose: print(f"{dialogue.complete()} Templates compiled successfully")

    def make_results(self, ppxf):
        """
        Makes the results

        Parameters
        ----------
        ppxf: ppxf.ppxf object
            ppxf object AFTER the fit has been performed


        Returns
        -------
        None. But it creates the following

        self.ppxf
        self.matching_indices
        self.matching_raw_spectra
        self.matching_spectra
        self.matching_apolynomial
        self.matching_mpolynomial
        self.results

        """
        self.ppxf=ppxf
        # indices of weights that are non-zero
        self.match_indices = np.argwhere(ppxf.weights != 0.0).T[0]

        # the non-zero weights
        weights = ppxf.weights[self.match_indices]
        self.matching_raw_spectra = []

        """
        # iterating over every solution
        for i in range(self._template_properties_BROKEN[self.match_indices].shape[0]):
            # finding matching age
            i_age = np.argwhere(np.round(self.t,2)==self._template_properties_BROKEN[self.match_indices][i,1])[0]
            # finding matching metallicity
            i_met = np.argwhere(np.round(self.num_z_list,5)==self._template_properties_BROKEN[self.match_indices][i,0])[0]
            # compiling the spectra
            self.matching_raw_spectra.append(self.templates[:, i_age, i_met])
        """

        for INDEX in self.match_indices:
            i_met = int(np.argwhere(self.num_z_list == self.template_properties[INDEX][0]))
            i_age = int(np.argwhere(np.isclose(self.log_age_cols, self.template_properties[INDEX][1])))
            self.matching_raw_spectra.append(self.templates[:, i_age, i_met]) # wtf

        self.matching_raw_spectra = np.array(self.matching_raw_spectra)
        self.matching_spectra = ppxf.matrix[:, ppxf.degree+1:].T[self.match_indices]
        self.matching_apolynomial = ppxf.apoly
        self.matching_mpolynomial = ppxf.mpoly

        # compiling summary of the results
        # TODO: ADD LIGHTFRACTION - am tired of calculating it lol
        self.results = pd.DataFrame(np.vstack((self.template_properties[self.match_indices].T, weights.T)).T,
                                    columns=['met', 'age', 'weights'])

    def save(self, path):
        """
        Save the current KVN object to a pickle file

        Parameters
        ----------
        path : str
            location to save the pickle file. Suggested extension .pkl (not automatically added)

        Returns
        -------
        None
        """
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, path):
        """
        Loads a pickled KVN object

        Parameters
        ----------
        path : str
            location of the pickle file to load in.

        Returns
        -------
        None
        """
        f = open(path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)
