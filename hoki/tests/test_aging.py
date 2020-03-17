import hoki.age_utils as au
import hoki.load as load
import pkg_resources
import numpy as np
import pandas as pd

data_path = pkg_resources.resource_filename('hoki', 'data')

hr_file = data_path+'/hrs-sin-imf_chab100.zem4.dat'

myhrd = load.model_output(hr_file, hr_type='TL')
fake_input = pd.DataFrame.from_dict({'logT': np.array([4.58, 4.48, 4.14]),
                                     'logL': np.array([4.83, 5.07, 5.40])})

bad_input = pd.DataFrame.from_dict({'logT': np.array(['bla']),
                                    'logL': np.array([4.83])})

class TestFindHRDCoordinates(object):
    def test_fake_input(self):
        T_coord, L_coord = au.find_hrd_coordinates(obs_df=fake_input, myhrd=myhrd)
        assert np.sum(np.isclose([T_coord[0], T_coord[1], T_coord[2]], [45,44,40]))==3, "Temperature coordinates wrong"
        assert np.sum(np.isclose([L_coord[0], L_coord[1], L_coord[2]], [77,80,83]))==3, "Luminosity coordinates wrong"

    def test_bad_input(self):
        T_coord, L_coord = au.find_hrd_coordinates(obs_df=bad_input, myhrd=myhrd)
        assert np.isnan(T_coord[0]), "This should be a nan"
        assert np.isclose(L_coord[0],77), "This L coordinate is wrong - test_bad_input."