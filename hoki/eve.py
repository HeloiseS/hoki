import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hoki import load
import matplotlib.colors as mcolors

plt.style.use('hfs')
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError
from hoki.utils.hoki_object import HokiObject
import warnings

# TODO: docstrings
# TODO: unittests

class Eve(object):
    def  __init__(self, met, eve_path, name='NONE'):
        self.ID_TABLE = pd.read_hdf(eve_path, f'{met}/ID_TABLE')
        self.SUMMARY = pd.read_hdf(eve_path, f'{met}/SUMMARY')
        self.CEE = pd.read_hdf(eve_path, f'{met}/CEE')
        self.MT = pd.read_hdf(eve_path, f'{met}/MT')
        self.DEATH = pd.read_hdf(eve_path, f'{met}/DEATH')
        self.met = met
        self.name=name
        
    def __repr__(self):
        obj_type=f'{type(self)}\n'
        header = f'METALICITY: {self.met} '
        name = f'\n\nPROJECT NAME:\t{self.name}'
        dashes='\n\n--------- SCHEMA -------\n\n'
        id_table=f'ID_TABLE\n{self.ID_TABLE.columns.values}\n\n'
        summary_table = f'SUMMARY\n{self.SUMMARY.columns.values}\n\n'
        cee_table = f'CEE\n{self.CEE.columns.values}\n\n'
        mt_table = f'MT\n{self.MT.columns.values}\n\n'
        death_table = f'DEATH\n{self.DEATH.columns.values}\n\n'
        return obj_type+header+name+dashes+id_table+summary_table+cee_table+mt_table+death_table


