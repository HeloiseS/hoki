import pandas as pd
import hoki.hrdiagrams
import hoki.cmd
import hoki.load as load
from hoki.constants import BPASS_TIME_BINS
import warnings
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError, HokiFormatWarning
from hoki.utils.hoki_object import HokiObject
from hoki.utils.hoki_dialogue import HokiDialogue
import numpy as np
from scipy import stats, optimize
import hoki.age.utils as au

Dialogue = HokiDialogue()


class AgeWizard(HokiObject):
    """
    AgeWizard object
    """

    def __init__(self, obs_df, model, nsamples=100):
        """
        Initialisation of the AgeWizard object

        Parameters
        ----------
        obs_df: pandas.DataFrame
            Observational data. MUST contain a logT and logL column (for HRD comparison) or a col and mag column
            (for CMD comparison)
        model: str or hoki.hrdiagrams.HRDiagrams() hoki.cmd.CMD()
            Location of the modeled HRD or CMD. This can be an already instanciated HRDiagram or CMD() object, or a
            path to an HR Diagram file or a pickled CMD.
        nsamples: int, optional
            Number of times each data point should be sampled from its error distribution. Default is 100.
            This only matters if you are taking errors into account.
        """

        print(f"{Dialogue.info()} AgeWizard Starting")
        print(f"{Dialogue.running()} Initial Checks")

        # Making sure the osbervational properties are given in a format we can use.
        if not isinstance(obs_df, pd.DataFrame):
            raise HokiFormatError("Observations should be stored in a Data Frame")

        if 'name' not in obs_df.columns:
            warnings.warn("We expect the name of sources to be given in the 'name' column. "
                          "If I can't find names I'll make my own ;)", HokiFormatWarning)

        # Checking what format they giving for the model:
        if isinstance(model, hoki.hrdiagrams.HRDiagram):
            self.model = model
        elif isinstance(model, hoki.cmd.CMD):
            self.model = model
        elif isinstance(model, str) and 'hrs' in model:
            self.model = load.model_output(model, hr_type='TL')
        elif isinstance(model, str):
            try:
                self.model = load.unpickle(path=model)
            except AssertionError:
                print(f'{Dialogue.ORANGE}-----------------{Dialogue.ENDC}')
                print(
                    f'{Dialogue.debugger()}\nThe model param should be a path to \na BPASS HRDiagram output file or pickled CMD,'
                    'or\na hoki.hrdiagrams.HRDiagram or a hoki.cmd.CMD')
                print(f'{Dialogue.ORANGE}-----------------{Dialogue.ENDC}')
                raise HokiFatalError('model is ' + str(type(model)))

        else:
            print(f'{Dialogue.ORANGE}-----------------{Dialogue.ENDC}')
            print(f'{Dialogue.debugger()}\nThe model param should be a path to \na BPASS HRDiagram output file or pickled CMD,'
                  'or\na hoki.hrdiagrams.HRDiagram or a hoki.cmd.CMD')
            print(f'{Dialogue.ORANGE}-----------------{Dialogue.ENDC}')
            raise HokiFatalError('model is ' + str(type(model)))

        print(f"{Dialogue.complete()} Initial Checks")

        self.obs_df = obs_df.copy()

        # not needed?
        # self.coordinates = find_coordinates(self.obs_df, self.model)

        # This line is obsolete but might need revival if we ever want to add the not normalised distributions again
        # self._distributions = calculate_distributions_normalised(self.obs_df, self.model)

        self.pdfs = au.calculate_individual_pdfs(self.obs_df, self.model, nsamples=nsamples).fillna(0)
        self.sources = self.pdfs.columns.to_list()
        self.sample_pdf = None
        self._most_likely_age = None

    def calculate_sample_pdf(self, not_you=None, return_df=False):
        # self.sample_pdf = calculate_sample_pdf(self._distributions, not_you=not_you)
        self.sample_pdf = au.calculate_sample_pdf(self.pdfs, not_you=not_you)
        if return_df: return self.sample_pdf

    @property
    def most_likely_age(self):
        """
        Finds  the most likely age by finding the max value in self.calculate_sample_pdf
        """
        if self._most_likely_age is not None: return self._most_likely_age

        if self.sample_pdf is None:
            warnings.warn('self.multiplied_pdf is not yet defined -- running AgeWizard.combined_pdfs()',
                          HokiUserWarning)
            self.calculate_sample_pdf()

        index = self.sample_pdf.index[self.sample_pdf.pdf == max(self.sample_pdf.pdf)].tolist()
        return self.t[index]

    @property
    def most_likely_ages(self):
        """
        Finds the most likely ages for all the sources given in the obs_df DataFrame.
        """
        # index = self.pdfs.drop('time_bins', axis=1).idxmax(axis=0).tolist()
        index = self.pdfs.idxmax(axis=0).tolist()
        return self.t[index]

    def calculate_p_given_age_range(self, age_range):
        """
        Calculates the probability that each source has age within age_range

        Parameters
        ----------
        age_range: list or tuple of 2 values
            Minimum and Maximum age to consider (inclusive).

        Returns
        -------
        numpy.array containing the probabilities.

        """
        # Selects only the rows corresponding to the range age_range[0] to age_range[1] (inclusive)
        # and then we sum the probabilities up for each column.
        probability = au.calculate_p_given_age_range(self.pdfs, age_range)

        return probability
