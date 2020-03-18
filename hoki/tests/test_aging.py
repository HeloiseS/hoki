import hoki.age_utils as au
import hoki.load as load
import pkg_resources
import numpy as np
import pandas as pd

data_path = pkg_resources.resource_filename('hoki', 'data')

hr_file = data_path+'/hrs-sin-imf_chab100.zem4.dat'

myhrd = load.model_output(hr_file, hr_type='TL')
fake_input = pd.DataFrame.from_dict({'name': ['star1', 'star2', 'star3'],
                                     'logT': np.array([4.58, 4.48, 4.14]),
                                     'logL': np.array([4.83, 5.07, 5.40])})

bad_input = pd.DataFrame.from_dict({'logT': np.array(['bla']),
                                    'logL': np.array([4.83])})

no_name_input = pd.DataFrame.from_dict({'logT': np.array([4.58, 4.48, 4.14]),
                                        'logL': np.array([4.83, 5.07, 5.40])})

bad_input2 = pd.DataFrame.from_dict({'logT': np.array([4.58, 'bla']),
                                     'logL': np.array([4.83, 2.0])})

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
