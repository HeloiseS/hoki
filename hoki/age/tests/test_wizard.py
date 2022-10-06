import hoki.age.utils as au
from hoki.age.wizard import AgeWizard
import hoki.load as load
import pkg_resources
import numpy as np
import pandas as pd
import pytest
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError


# Loading Data

data_path = pkg_resources.resource_filename('hoki', 'data')
hr_file = data_path + '/hrs-sin-imf_chab100.zem4.dat'
cmd_file = data_path + '/cmd_bv_z002_bin_imf135_300'
myhrd = load.model_output(hr_file, hr_type='TL')
mycmd = load.unpickle(cmd_file)
# Creating Test Inputs

fake_hrd_input = pd.DataFrame.from_dict({'name': ['star1', 'star2', 'star3'],
                                         'logT': np.array([4.58, 4.48, 4.14]),
                                         'logL': np.array([4.83, 5.07, 5.40])})

bad_hrd_input = pd.DataFrame.from_dict({'logT': np.array(['bla']),
                                        'logL': np.array([4.83])})

no_name_input = pd.DataFrame.from_dict({'logT': np.array([4.58, 4.48, 4.14]),
                                        'logL': np.array([4.83, 5.07, 5.40])})

bad_hrd_input2 = pd.DataFrame.from_dict({'logT': np.array([4.58, 'bla']),
                                         'logL': np.array([4.83, 2.0])})

fake_cmd_input = pd.DataFrame.from_dict({'name': ['star1', 'star2', 'STAR3'],
                                         'col': np.array([-0.3, 0.5, -0.25]),
                                         'mag': np.array([-5, -10, -1])})

bad_cmd_input = pd.DataFrame.from_dict({'col': np.array(['bla']),
                                        'mag': np.array([-5])})


stars_SYM = pd.DataFrame.from_dict({'name': np.array(['118-1', '118-2', '118-3', '118-4']),
                                    'logL': np.array([5.0, 5.1, 4.9, 5.9]),
                                    'logL_err': np.array([0.1, 0.2, 0.1, 0.1]),
                                    'logT': np.array([4.48, 4.45, 4.46, 4.47]),
                                    'logT_err': np.array([0.1, 0.2, 0.1, 0.1]),
                                    })

class TestAgeWizardBasic(object):
    def test_init_basic(self):
        assert AgeWizard(obs_df=fake_hrd_input, model=hr_file), "Loading HRD file path failed"
        assert AgeWizard(obs_df=fake_hrd_input, model=myhrd), "Loading with hoki.hrdiagrams.HRDiagram failed"
        assert AgeWizard(obs_df=fake_cmd_input, model=mycmd), 'Loading with hoki.cmd.CMD'
        assert AgeWizard(obs_df=fake_cmd_input, model=cmd_file), 'Loading CMD from frile failed'

    def test_bad_init(self):
        with pytest.raises(HokiFatalError):
            __, __ = AgeWizard(obs_df=fake_cmd_input, model='sdfghj'), 'HokiFatalError should be raised'

        with pytest.raises(HokiFormatError):
            __, __ = AgeWizard(obs_df='edrftgyhu', model=cmd_file), 'HokiFormatError should be raised'

    def test_combine_pdfs_not_you(self):
        wiz = AgeWizard(fake_hrd_input, myhrd)
        wiz.calculate_sample_pdf(not_you=['star1'])
        cpdf = wiz.sample_pdf.pdf
        assert np.sum(np.isclose([cpdf[0], cpdf[9]], [0.0,  0.7231526323765232])) == 2, "combined pdf is not right"

    def test_most_likely_age(self):
        wiz = AgeWizard(obs_df=fake_hrd_input, model=hr_file)
        assert np.isclose(wiz.most_likely_age[0], 6.9), "Most likely age wrong"

    def test_most_likely_ages(self):
        wiz = AgeWizard(obs_df=fake_hrd_input, model=hr_file)
        a = wiz.most_likely_ages
        assert np.sum(np.isclose([a[0], a[1], a[2]], [6.9, 6.9, 6.9])) == 3, "Most likely ages not right"

    def test_combine_pdfs(self):
        wiz = AgeWizard(fake_hrd_input, myhrd)
        wiz.calculate_sample_pdf()
        assert np.isclose(wiz.sample_pdf.pdf[9],0.551756734145878), "Something is wrong with the combined_Age PDF"

    def test_calculate_p_given_age_range(self):
        wiz = AgeWizard(fake_hrd_input, myhrd)
        probas = wiz.calculate_p_given_age_range([6.7, 6.9])
        assert np.sum(np.isclose([probas[0], probas[1], probas[2]],
                                 [0.515233714952414, 0.7920611550946726, 0.6542441096583737])) == 3, \
            "probability given age range is messed up"


class TestAgeWizardErrors(object):
    def test_agewizard_with_errors_runs(self):
        wiz = AgeWizard(stars_SYM, myhrd, nsamples=200)
        wiz.calculate_sample_pdf()


