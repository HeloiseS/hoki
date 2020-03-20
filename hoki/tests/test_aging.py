import hoki.age_utils as au
import hoki.load as load
import pkg_resources
import numpy as np
import pandas as pd
import pytest

# Loading Data

data_path = pkg_resources.resource_filename('hoki', 'data')
hr_file = data_path+'/hrs-sin-imf_chab100.zem4.dat'
myhrd = load.model_output(hr_file, hr_type='TL')

# Creating Test Inputs

fake_input = pd.DataFrame.from_dict({'name': ['star1', 'star2', 'star3'],
                                     'logT': np.array([4.58, 4.48, 4.14]),
                                     'logL': np.array([4.83, 5.07, 5.40])})

bad_input = pd.DataFrame.from_dict({'logT': np.array(['bla']),
                                    'logL': np.array([4.83])})

no_name_input = pd.DataFrame.from_dict({'logT': np.array([4.58, 4.48, 4.14]),
                                        'logL': np.array([4.83, 5.07, 5.40])})

bad_input2 = pd.DataFrame.from_dict({'logT': np.array([4.58, 'bla']),
                                     'logL': np.array([4.83, 2.0])})


# Testing Suite

class TestAgeWizard(object):
    def test_init_basic(self):
        assert au.AgeWizard(obs_df=fake_input, model=hr_file), "Loading from file path failed"
        assert au.AgeWizard(obs_df=fake_input, model=myhrd), "Loading with hoki.hrdiagrams.HRDiagram failed"

    def test_init_exception(self):
        with pytest.raises(TypeError):
            au.AgeWizard(obs_df=fake_input, model=3)

    def test_combine_pdfs_not_you(self):
        wiz = au.AgeWizard(fake_input, myhrd)
        wiz.multiply_pdfs(not_you=['star1'])
        cpdf = wiz.multiplied_pdf.pdf
        assert np.sum(np.isclose([cpdf[0], cpdf[9]], [0.0, 0.878162355350702]))==2, "combined pdf is not right"

    def test_most_likely_age(self):
        wiz = au.AgeWizard(obs_df=fake_input, model=hr_file)
        assert np.isclose(wiz.most_likely_age[0],6.9), "Most likely age wrong"

    def test_most_likely_ages(self):
        wiz = au.AgeWizard(obs_df=fake_input, model=hr_file)
        a =  wiz.most_likely_ages
        assert np.sum(np.isclose([a[0], a[1], a[2]], [6.9, 6.9, 6.9]))==3, "Most likely ages not right"

    def test_combine_pdfs(self):
        wiz = au.AgeWizard(fake_input, myhrd)
        wiz.multiply_pdfs()
        assert np.isclose(wiz.multiplied_pdf.pdf[9], 0.9837195045903536), "Something is wrong with the combined_Age PDF"

    def test_calculate_p_given_age_range(self):
        wiz = au.AgeWizard(fake_input, myhrd)
        probas = wiz.calculate_p_given_age_range([6.7, 6.9])
        assert np.sum(np.isclose([probas[0], probas[1], probas[2]],
                                 [0.515233714952414, 0.7920611550946726, 0.6542441096583737]))==3, \
            "probability given age range is messed up"


class TestFindHRDCoordinates(object):
    def test_fake_input(self):
        T_coord, L_coord = au.find_hrd_coordinates(obs_df=fake_input, myhrd=myhrd)
        assert np.sum(np.isclose([T_coord[0], T_coord[1], T_coord[2]], [45,44,40]))==3, "Temperature coordinates wrong"
        assert np.sum(np.isclose([L_coord[0], L_coord[1], L_coord[2]], [77,80,83]))==3, "Luminosity coordinates wrong"

    def test_bad_input(self):
        T_coord, L_coord = au.find_hrd_coordinates(obs_df=bad_input, myhrd=myhrd)
        assert np.isnan(T_coord[0]), "This should be a nan"
        assert np.isclose(L_coord[0],77), "This L coordinate is wrong - test_bad_input."


class TestNormalise1D(object):
    def test_it_runs(self):
        au.normalise_1d(np.array([0,1,4,5,0,1,7,8]))

    def test_basic(self):
        norm = au.normalise_1d(np.array([0,0,1,0,0,0,0]))
        assert norm[2] == 1, 'Normalisation done wrong'
        assert sum(norm) == 1, "Normalisaton done wrong"


class TestCalculatePDFs(object):
    def test_fake_input(self):
        pdf_df = au.calculate_pdfs(fake_input, myhrd)
        assert 'star1' in pdf_df.columns, "Column name issue"
        assert int(sum(pdf_df.star1))== 1, "PDF not calculated correctly"

    def test_input_without_name(self):
        pdf_df = au.calculate_pdfs(no_name_input, myhrd)
        assert 's1' in pdf_df.columns, "Column names not created right"

    def test_bad_input(self):
        pdf_df = au.calculate_pdfs(bad_input2, myhrd)
        assert not np.isnan(sum(pdf_df.s0)), "somwthing went wrong"
        assert np.isnan(sum(pdf_df.s1)), "somwthing went wrong"


class TestMultiplyPDFs(object):
    def test_basic(self):
        pdfs_good = au.calculate_pdfs(fake_input, myhrd)
        combined = au.multiply_pdfs(pdfs_good)
        assert np.isclose(combined.pdf[9], 0.9837195045903536), "combined PDF not right"

    def test_drop_bad(self):
        pdfs_good = au.calculate_pdfs(fake_input, myhrd)
        combined = au.multiply_pdfs(pdfs_good, not_you=[3])
        assert np.isclose(combined.pdf[9], 0.9837195045903536), "combined PDF not right"

    def test_drop_good(self):
        pdfs_good = au.calculate_pdfs(fake_input, myhrd)
        combined = au.multiply_pdfs(pdfs_good, not_you=['star1'])
        assert np.isclose(combined.pdf[9], 0.878162355350702), "combined PDF not right"