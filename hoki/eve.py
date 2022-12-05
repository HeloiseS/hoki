"""
hoki.eve
--------
Author: H.F.Stevance
email: hfstevance@gmail.com

Description
-----------
Implements the `Eve` class to be used with EvE.hdf5(alpha) that can be downloaded on Zenodo
https://zenodo.org/record/7341382#.Y4y9cNLMLLo
**NOTE** You can just download the `EvE.hdf5` file - the code that is published with it just for record keeping
as that is the Fortran and python code I used to create the database. If there are issues in the database
and feel like going through the code to look for bugs.

"""

import pandas as pd
from hoki.constants import BPASS_METALLICITIES
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError
from os.path import exists


class Eve(object):
    def  __init__(self, met, eve_path, name='NONE'):
        """
        Utility object to contain the EvE subdatabase for a single BPASS metallicity

        Parameters
        ----------
        met: str
            BPASS metallicity string
        eve_path: str
            Path to the EvE.hdf5 file that contains the database
        name: str, optional
            Name of the project this object is related to. Default is NONE

        Example
        -------
        To instantiate an Eve object:
        >>> eve = Eve(met='z020', eve_path='../../EvE/EvE.hdf5', name='test')
        To check the schema you can just return the variable where your object is stored
        >>> eve
        <class 'hoki.eve.Eve'>
        METALICITY: z020 
        
        PROJECT NAME:	NONE
        
        --------- TABLES -------
        
        ID_TABLE
        ['filenames']
        
        SUMMARY
        ['M1_ZAMS' 'M2_ZAMS' 'P_ZAMS' 'lifetime_MS' 'MT_bool' 'CEE_bool' 'AGE_end'
         'M1_end' 'M2_end' 'P_end' 'modelimf' 'mixedimf' 'type']
        
        CEE
        ['TIMESTEP_start' 'AGE_start' 'M1_start' 'M2_start' 'P_start' 'LOGA_start'
         'TIMESTEP_end' 'AGE_end' 'M1_end' 'M2_end' 'P_end' 'LOGA_end'
         'avgDM1R_msolpyr' 'avgDM2R_msolpyr' 'avgDM1W_msolpyr' 'avgDM2W_msolpyr'
         'CASE']
        
        MT
        ['TIMESTEP_start' 'AGE_start' 'M1_start' 'M2_start' 'P_start' 'LOGA_start'
         'TIMESTEP_end' 'AGE_end' 'M1_end' 'M2_end' 'P_end' 'LOGA_end'
         'avgDM1R_msolpyr' 'avgDM2R_msolpyr' 'avgDM1W_msolpyr' 'avgDM2W_msolpyr'
         'CASE']
        
        DEATH
        ['timestep' 'age' 'log(R1)' 'log(T1)' 'log(L1)' 'M1' 'He_core1' 'CO_core1'
         'ONe_core1' 'Nan9' 'X' 'Y' 'C' 'N' 'O' 'Ne' 'MH1' 'MHe1' 'MC1' 'MN1'
         'MO1' 'MNe1' 'MMg1' 'MSi1' 'MFe1' 'envelope_binding_E' 'star_binding_E'
         'Mrem_weakSN' 'Mej_weakSN' 'Mrem_SN' 'Mej_SN' 'Mrem_superSN'
         'Mej_superSN' 'AM_bin' 'P_bin' 'log(a)' 'Nan36' 'M2' 'MTOT' 'DM1W' 'DM2W'
         'DM1A' 'DM2A' 'DM1R' 'DM2R' 'DAM' 'log(R2)' 'log(T2)' 'log(L2)' '?'
         'modelimf' 'mixedimf' 'V-I' 'U' 'B' 'V' 'R' 'I' 'J']
    
        """

        if met not in BPASS_METALLICITIES:                              # Checks user has given us a valid metallicity
            raise HokiFatalError(f'met must be in BPASS_METALLICITIES: {BPASS_METALLICITIES}')

        if exists(eve_path) is False:                                   # Ensures path to EvE is correct
            raise HokiFatalError(f'Path {eve_path} does not exist')
       
        self.ID_TABLE = pd.read_hdf(eve_path, f'{met}/ID_TABLE')        # Loading each table for a given metallicity
        self.SUMMARY = pd.read_hdf(eve_path, f'{met}/SUMMARY')          # set by the user as `met`
        self.CEE = pd.read_hdf(eve_path, f'{met}/CEE')
        self.MT = pd.read_hdf(eve_path, f'{met}/MT')
        self.DEATH = pd.read_hdf(eve_path, f'{met}/DEATH')
        self.met = met
        self.name = name
        
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


