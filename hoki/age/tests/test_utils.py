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

#### Creating Test Inputs

# Version 1 AgeWizard
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

# Version 2 AgeWizard
stars_none = pd.DataFrame.from_dict({'name': np.array(['118-1', '118-2', '118-3', '118-4']),
                                     'logL': np.array([5.0, 5.1, 4.9, 5.9]),
                                     'logT': np.array([4.48, 4.45, 4.46, 4.47]),
                                     })

stars_SYM = pd.DataFrame.from_dict({'name': np.array(['118-1', '118-2', '118-3', '118-4']),
                                    'logL': np.array([5.0, 5.1, 4.9, 5.9]),
                                    'logL_err': np.array([0.1, 0.2, 0.1, 0.1]),
                                    'logT': np.array([4.48, 4.45, 4.46, 4.47]),
                                    'logT_err': np.array([0.1, 0.2, 0.1, 0.1]),
                                    })

stars_SYM_T_err_missing = pd.DataFrame.from_dict({'name': np.array(['118-1', '118-2', '118-3', '118-4']),
                                    'logL': np.array([5.0, 5.1, 4.9, 5.9]),
                                    'logL_err': np.array([0.1, 0.2, 0.1, 0.1]),
                                    'logT': np.array([4.48, 4.45, 4.46, 4.47]),
                                    'logT_err': np.array([0.1, 0.2, 0.1, 0.1]),
                                    })

#########
# MISC  #
#########

class TestNormalise1D(object):
    def test_it_runs(self):
        au.normalise_1d(np.array([0, 1, 4, 5, 0, 1, 7, 8]))

    def test_basic(self):
        norm = au.normalise_1d(np.array([0, 0, 1, 0, 0, 0, 0]))
        assert norm[2] == 1, 'Normalisation done wrong'
        assert sum(norm) == 1, "Normalisaton done wrong"

class TestErrorFlag(object):
    def test_no_err(self):
        assert au._error_flag(stars_none) is None, "Error Flag should be None"

    def test_ERR(self):
        assert au._error_flag(stars_SYM) == 'ERR', "Error Flag should be ERR"


#######################
# FINDING COORDINATES #
#######################


class TestFindCoordinates(object):
    def test_hrd_input(self):
        T_coord, L_coord = au.find_coordinates(obs_df=fake_hrd_input, model=myhrd)
        assert np.sum(
            np.isclose([T_coord[0], T_coord[1], T_coord[2]], [45, 44, 40])) == 3, "Temperature coordinates wrong"
        assert np.sum(
            np.isclose([L_coord[0], L_coord[1], L_coord[2]], [77, 80, 83])) == 3, "Luminosity coordinates wrong"

    def test_cmd_input(self):
        col_coord, mag_range = au.find_coordinates(obs_df=fake_cmd_input, model=mycmd)
        assert np.sum(
            np.isclose([col_coord[0], col_coord[1], col_coord[2]], [27, 35, 27])) == 3, "color coordinates wrong"
        assert np.sum(
            np.isclose([mag_range[0], mag_range[1], mag_range[2]], [90, 40, 130])) == 3, "magnitude coordinates wrong"


class TestFindCMDCoordinates(object):
    def test_fake_input(self):
        col_coord, mag_range = au._find_cmd_coordinates(obs_df=fake_cmd_input, mycmd=mycmd)
        assert np.sum(
            np.isclose([col_coord[0], col_coord[1], col_coord[2]], [27, 35, 27])) == 3, "color coordinates wrong"
        assert np.sum(
            np.isclose([mag_range[0], mag_range[1], mag_range[2]], [90, 40, 130])) == 3, "magnitude coordinates wrong"

    def test_bad_input(self):
        with pytest.raises(HokiFormatError):
            col_coord, mag_range = au._find_cmd_coordinates(obs_df=bad_hrd_input, mycmd=mycmd)

    def test_bad_input_2(self):
        col_coord, mag_range = au._find_cmd_coordinates(obs_df=bad_cmd_input, mycmd=mycmd)
        assert np.isclose(mag_range[0], 90), "This L coordinate is wrong - test_bad_input."


class TestFindHRDCoordinates(object):
    def test_fake_input(self):
        T_coord, L_coord = au._find_hrd_coordinates(obs_df=fake_hrd_input, myhrd=myhrd)
        assert np.sum(
            np.isclose([T_coord[0], T_coord[1], T_coord[2]], [45, 44, 40])) == 3, "Temperature coordinates wrong"
        assert np.sum(
            np.isclose([L_coord[0], L_coord[1], L_coord[2]], [77, 80, 83])) == 3, "Luminosity coordinates wrong"

    def test_bad_input(self):
        with pytest.raises(HokiFormatError):
            __, __ = au._find_hrd_coordinates(obs_df=bad_cmd_input, mycmd=mycmd)

    def test_bad_input(self):
        T_coord, L_coord = au._find_hrd_coordinates(obs_df=bad_hrd_input, myhrd=myhrd)
        #assert np.isnan(T_coord[0]), "This should be a nan"
        assert np.isclose(L_coord[0], 77), "This L coordinate is wrong - test_bad_input."


###############################
# CALCULATING INDIVIDUAL PDFS #
###############################


class TestCalculatePDFs(object):
    def test_fake_input(self):
        pdf_df = au.calculate_individual_pdfs(fake_hrd_input, myhrd)
        assert 'star1' in pdf_df.columns, "Column name issue"
        assert int(sum(pdf_df.star1)) == 1, "PDF not calculated correctly"

    def test_input_without_name(self):
        pdf_df = au.calculate_individual_pdfs(no_name_input, myhrd)

        assert 's1' in pdf_df.columns, "Column names not created right"

    def test_bad_input(self):
        pdf_df = au.calculate_individual_pdfs(bad_hrd_input2, myhrd)
        assert not np.isnan(sum(pdf_df.s0)), "something went wrong"

    def test_symmetric_errors(self):
        pdf_df = au.calculate_individual_pdfs(stars_SYM, myhrd)
        assert not np.isnan(sum(pdf_df['118-4'])), "something went wrong with symmetric errors"

#####################################
# PUTTING PDFS TOGETHER IN SOME WAY #
#####################################


class TestCalculateSamplePDF(object):
    def test_basic(self):
        distributions = au.calculate_distributions(fake_hrd_input, myhrd)
        combined = au.calculate_sample_pdf(distributions)
        assert np.isclose(combined.pdf[9], 0.2715379752638662), "combined PDF not right"

    def test_drop_bad(self):
        distributions = au.calculate_distributions(fake_hrd_input, myhrd)
        combined = au.calculate_sample_pdf(distributions, not_you=[3])
        assert np.isclose(combined.pdf[9], 0.2715379752638662), "combined PDF not right"

    def test_drop_good(self):
        distributions = au.calculate_distributions(fake_hrd_input, myhrd)
        combined = au.calculate_sample_pdf(distributions, not_you=['star1'])
        assert np.isclose(combined.pdf[9], 0.774602971512809), "combined PDF not right"

