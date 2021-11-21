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

class EvEWizard(HokiObject):
    def __init__(self, met, eve_path='EvE.h5'):
        # for now only support one metallicity

        super().__init__()
        self.eve_path = eve_path
        self.summary_table = pd.read_hdf(self.eve_path, f'/{met}/SUMMARY')
        self.cee_table = pd.read_hdf(self.eve_path, f'/{met}/COMMON_ENVELOPE')
        self.mt_table = pd.read_hdf(self.eve_path, f'/{met}/MASS_TRANSFER')
        self.death_table = pd.read_hdf(self.eve_path, f'/{met}/DEATH')
        self.dummy_df = None

    def _calc_logg(self):
        """Adds log_g column (OF STAR 1 - the one with detailed evol) to dummy_data_frame"""
        self.dummy_df['log_g1'] = np.log10((6.67259 * 10 ** (-8)) * (1.989 * 10 ** 33) * self.dummy_df.M1 / (
                    ((10 ** self.dummy_df['log(R1)']) * 6.9598 * (10 ** 10)) ** 2))

    def load_model(self, ID, columns=None):
        self.CURRENT_MODEL_ID = ID
        self.model_summary = self.summary_table[self.summary_table.MODEL_ID == ID]
        self.model_cee = self.cee_table[self.cee_table.MODEL_ID == ID]
        self.model_mt = self.mt_table[self.mt_table.MODEL_ID == ID]

        if columns is None:
            columns_of_interest = ['timestep',
                                   'age',
                                   'log(R1)',
                                   'log(T1)',
                                   'log(L1)',
                                   'log(a)',
                                   'DM1R',
                                   'DM1W',
                                   'M1',
                                   'M2',
                                   'X',
                                   'Y']
        elif columns == 'all':
            columns_of_interest = list(load.dummy_dict.keys())

        else:
            columns_of_interest = columns

        # loading the data
        self.dummy_df = load.dummy_to_dataframe(load.MODELS_PATH + self.model_summary.filenames.iloc[0])[
            columns_of_interest]
        self._calc_logg()

        self._T = self.dummy_df['log(T1)']
        self._L = self.dummy_df['log(L1)']

        ms_data = self.dummy_df[self.dummy_df.age <= self.model_summary.lifetime_MS.iloc[0]][['log(T1)', 'log(L1)']]
        self._T_ms = ms_data['log(T1)']
        self._L_ms = ms_data['log(L1)']

    def plots_RLOF(self, ax=None):
        if ax is None:
            f, ax = plt.subplots(ncols=2, figsize=(10, 3))

        ax1, ax2 = ax[0], ax[1]

        ax1.plot(self._T, self._L, c='grey', alpha=0.5)
        ax1.plot(self._T_ms, self._L_ms, c='k', alpha=0.5, lw=1, zorder=10, ls='--')

        for i in range(self.model_mt.shape[0]):
            step_mt_i, step_mt_f = self.model_mt[['TIMESTEP_start', 'TIMESTEP_end']].iloc[i]
            mask_mt = (self.dummy_df.timestep >= step_mt_i) & (self.dummy_df.timestep <= step_mt_f)
            ax1.plot(self.dummy_df[mask_mt]['log(T1)'], self.dummy_df[mask_mt]['log(L1)'],
                     c='orange', lw=2)

        for i in range(self.model_cee.shape[0]):
            step_cee_i, step_cee_f = self.model_cee[['TIMESTEP_start', 'TIMESTEP_end']].iloc[i]
            mask_cee = (self.dummy_df.timestep >= step_cee_i) & (self.dummy_df.timestep <= step_cee_f)
            ax1.plot(self.dummy_df[mask_cee]['log(T1)'], self.dummy_df[mask_cee]['log(L1)'],
                     c='crimson', lw=2)

        # f, ax = plt.subplots(ncols=1, figsize=(4,3))
        cb = ax2.scatter(self._T, self._L, c=np.abs(self.dummy_df['DM1R']), cmap='Blues', s=40)
        ax2.plot(self._T, self._L, c='grey', alpha=0.5, zorder=1)

        cbar = plt.colorbar(cb)

        for axis in ax:
            axis.set_xlim([self._T.max() + 0.2, self._T.min() - 0.2])
            axis.set_ylim([self._L.min() - 0.2, self._L.max() + 0.2])
            axis.set_xlabel('log(T1)')
            axis.set_ylabel('log(L1)')

        ax1.set_title('MS, MT, CEE')
        ax2.set_title('DM1R')

        return f, ax

    def plots_Lclass_n_winds(self, ax=None, Lclass_logg_boundaries=None):
        if ax is None:
            f, ax = plt.subplots(ncols=2, figsize=(10, 3))

        if Lclass_logg_boundaries is None:
            Lclass_logg_boundaries = [-1, 0, 1, 2, 3.5, 6]

        ax1, ax2 = ax[0], ax[1]

        cmap_rb = plt.get_cmap('RdYlBu')
        colors = cmap_rb(np.linspace(0, 1, len(Lclass_logg_boundaries) - 1))
        cmap, norm = mcolors.from_levels_and_colors(Lclass_logg_boundaries, colors)

        ax1.plot(T, L, c='grey', alpha=1, zorder=0.8)
        cb = ax1.scatter(T, L, c=model_data['log_g1'], cmap=cmap, s=40, norm=norm)
        cbar = f.colorbar(cb, ticks=Lclass_logg_boundaries, ax=ax1)

        ax2.plot(T, L, c='grey', alpha=10.5, zorder=1)
        cb = ax2.scatter(T, L, c=model_data['DM1W'], cmap='Purples', s=40)
        cbar = plt.colorbar(cb, ax=ax2)

        for axis in ax:
            axis.set_xlim([T.max() + 0.2, T.min() - 0.2])
            axis.set_ylim([L.min() - 0.2, L.max() + 0.2])
            axis.set_xlabel('log(T1)')
            axis.set_ylabel('log(L1)')

        ax1.set_title('Luminosity Class')
        ax2.set_title('DM1W')

        return f, ax

    def describe_evolution(self):
        # TODO: need to allow for several MT and CEE phases!!
        warnings.warn(HokiUserWarning('Multiple CEE and MT will not be described - implementation to come - '
                                      'CHECK self.model_mt and self.model_cee'))

        print(f'==================\nMODEL_ID = {self.CURRENT_MODEL_ID}\n==================')
        # MAIN SEQUENCE
        M1_endMS, M2_endMS = self.dummy_df[np.isclose(self.dummy_df.age.astype(float),
                                                      self.model_summary.lifetime_MS.astype(float))][['M1',
                                                                                                      'M2']].values[0]

        # print(f"ZAMS   |\tM1={round(self.model_summary.M1_zams.iloc[0],2)} Msol \tM2={round(self.model_summary.M2_zams.iloc[0],2)} Msol")
        # print(f"  => MS lifetime {round(self.model_summary.lifetime_MS.values[0]/1e6,3)} Myr")
        # print(f"END MS |\tM1={round(M1_endMS,2)} Msol \tM2={round(M2_endMS, 2)} Msol")

        print('MS   |        | START |\tM1={:.2f} Msol \tM2={:.2f} Msol '.format(self.model_summary.M1_zams.iloc[0],
                                                                                 self.model_summary.M2_zams.iloc[0],
                                                                                 ))

        print('MS   |        |  END  |\tM1={:.2f} Msol \tM2={:.2f} Msol \tage={:.3f} Myrs'.format(M1_endMS,
                                                                                                  M2_endMS,
                                                                                                  self.model_summary.lifetime_MS.iloc[
                                                                                                      0],
                                                                                                  ))

        if self.model_mt.size > 0:  # since MT contains CEE, this is the only necessary condition
            print('\n-------------------\nBINARY INTERACTIONS\n-------------------')
            print('RLOF | CASE {} | START |\tM1={:.2f} Msol \tM2={:.2f} Msol \tage={:.3f} Myrs'.format(
                self.model_mt.CASE.iloc[0],
                self.model_mt.M1_start.iloc[0],
                self.model_mt.M2_start.iloc[0],
                self.model_mt.AGE_start.iloc[0] / 1e6))
            if self.model_cee.size > 0:
                dt_mt = (self.model_cee.AGE_start - self.model_mt.AGE_start).iloc[0]
                print(f"  =>      RLOF in semi-detached binary for {round(dt_mt / 1e3, 3)} thousand years")
                dM1 = (self.model_cee.M1_start - self.model_mt.M1_start).iloc[0]
                dM2 = (self.model_cee.M2_start - self.model_mt.M2_start).iloc[0]
                dMsys = dM2 + dM1

                print("  =>      MASS CHANGE |\tM1={:.2f} Msol \tM2={:.2f} Msol \tMsys={:.2f} Msol".format(dM1, dM2,
                                                                                                           dMsys))

                print('\nCEE  | CASE {} | START |\tM1={:.2f} Msol \tM2={:.2f} Msol \tage={:.3f} Myrs'.format(
                    self.model_cee.CASE.iloc[0],
                    self.model_cee.M1_start.iloc[0],
                    self.model_cee.M2_start.iloc[0],
                    self.model_cee.AGE_start.iloc[0] / 1e6))

                print('CEE  |        |  END  |\tM1={:.2f} Msol \tM2={:.2f} Msol \tage={:.3f} Myrs'.format(
                    self.model_cee.M1_end.iloc[0],
                    self.model_cee.M2_end.iloc[0],
                    self.model_cee.AGE_end.iloc[0] / 1e6))
                dt_cee = self.model_cee.AGE_end.iloc[0] - self.model_cee.AGE_start.iloc[0]
                print(f"  =>      CEE phase lasts {round(dt_cee / 1e3, 3)} thousand years")
                print("  =>      MASS CHANGE |\tM1={:.2f} Msol \tM2={:.2f} Msol \tMsys={:.2f} Msol".format(
                    self.model_cee.DELTA_M1.iloc[0],
                    self.model_cee.DELTA_M2.iloc[0],
                    self.model_cee.DELTA_Msys.iloc[0]))

                print('\nRLOF |        |  END  |\tM1={:.2f} Msol \tM2={:.2f} Msol \tage={:.3f} Myrs'.format(
                    self.model_mt.M1_end.iloc[0],
                    self.model_mt.M2_end.iloc[0],
                    self.model_mt.AGE_end.iloc[0] / 1e6))

                dt_mt = (self.model_mt.AGE_end - self.model_cee.AGE_end).iloc[0]
                print(f"  =>      RLOF in semi-detached binary for another {round(dt_mt / 1e3, 3)} thousand years")
                dM1 = (self.model_mt.M1_end - self.model_cee.M1_end).iloc[0]
                dM2 = (self.model_mt.M2_end - self.model_cee.M2_end).iloc[0]
                dMsys = dM2 + dM1

                print("  =>      MASS CHANGE |\tM1={:.2f} Msol \tM2={:.2f} Msol \tMsys={:.2f} Msol".format(dM1, dM2,
                                                                                                           dMsys))


            else:
                print('RLOF |        |  END  |\tM1={:.2f} Msol \tM2={:.2f} Msol \tage={:.3f} Myrs'.format(
                    self.model_mt.M1_end.iloc[0],
                    self.model_mt.M2_end.iloc[0],
                    self.model_mt.AGE_end.iloc[0] / 1e6))

                dt_mt = (self.model_mt.AGE_end - self.model_mt.AGE_start).iloc[0]
                print(f"  =>      RLOF in semi-detached binary for {round(dt_mt / 1e3, 3)} thousand years")

            print("\nEND BINARY INTERACTION|                                 age = {:.3f} Myrs".format(
                self.model_mt.AGE_end.iloc[0] / 1e6))
            print("TOTAL MASS CHANGE     |\tM1={:.2f} Msol \tM2={:.2f} Msol \tMsys={:.2f} Msol".format(
                self.model_mt.DELTA_M1.iloc[0],
                self.model_mt.DELTA_M2.iloc[0],
                self.model_mt.DELTA_Msys.iloc[0]))


        else:
            print('-------------------\nNO BINARY INTERACTIONS')

        print('-------------------\n')
        print("PROPERTIES AT DEATH  |\tM1={:.2f} Msol \tM2={:.2f} Msol \tage={:.3f} Myrs".format(
            self.model_summary.M1_end.iloc[0],
            self.model_summary.M2_end.iloc[0],
            self.model_summary.lifetime_total.iloc[0] / 1e6))


''' OLD SHIT 

#TODO: SHOULD ADD LOGGING TO THIS?



#TODO: rename to reflect the fact it's a dictionary
#TODO: A LOT OF THIS IS ERELEVANT WHEN YOU HAVE EVE.H5

#TODO: MAKE PLOTING UTILITY THAT USES THE EVE TABLES
#TODO: MAKE UTILITY TO VERBOSE DESCRIBE EVOLUTIONARY PATH FROM EVE TABLES

evolution_vector_dict = dict({'0': 'MT CASE A', '1': 'CEE CASE A',
                          '2': 'MT CASE B', '3': 'CEE CASE B',
                          '4': 'MT CASE C', '5': 'CEE CASE C',
                          '6': 'MT @DEATH', '7': 'CEE @DEATH', })


def decode_and_print_evolution_vector(evolution_vector):
    for stage in evolution_vector:
        if stage[0] == "P":
            print(f'Primary {evolution_vector_dict[stage[1]]}')
        if stage[0] == "S":
            print(f'Secondary {evolution_vector_dict[stage[1]]}')

# TODO: might need to break this function down for unitesting
# TODO: might want a script that does this in a non OOP way to avoid overheads if runnign through whole dataset?
def make_evolution_vector(dummy_df, PRIM=True):
    """
    This makes a vector (LIST) describing the evolution of the
    Parameters
    ----------
    dummy_df
    PRIM  #TODO: MAYBE THIS SHOULD BE AUTOMATIC - from the filename?

    Returns
    -------

    """
    FLAG = [] # this is where we're going to record the warnings.

    #TODO: should add option to give path to dummy on top of df
    if PRIM is True:
        code='P' # marks primary models
    elif PRIM is False:
        code='S' # marks secondary models
    elif PRIM is None:
        code=' ' # for single stars
    else:
        raise HokiFatalError("PRIM needs to be True, False or None")
    #TODO: might need code for the type of system it is (-1, 0, 1, 2, 3, 4)

    evolution_vector=[]

    # MAIN SEQUENCE
    # TODO: check the condition for MS with jan
    try: # TODO: why this try and except
        i_end_MS = np.argwhere(dummy_df.He_core1.values>0.01)[1][0]
    except IndexError:
        i_end_MS = np.argwhere(dummy_df.He_core1.values>0.01)[0][0]

    try: # TODO: why this try and except
        i_start_caseC= np.argwhere(dummy_df.CO_core1.values>0.01)[1][0]
    except IndexError:
        try:
            i_start_caseC = np.argwhere(dummy_df.CO_core1.values>0.01)[0][0]
        except IndexError:
            i_start_caseC = None # doesn't get far enough in evolution to finish he core burning

    if i_end_MS > i_start_caseC: FLAG.append('BADCASEC')
    #TODO: LOG IT AS A PROBLEM WITH THE VALUES AND NAME OF FILE

    ms_lifetime = dummy_df.age.iloc[i_end_MS]

    # CEE AND MASS TRANSFER
    # TODO: check the condition of cee and mt with jan
    cee = dummy_df[dummy_df['log(a)']* 1.05<dummy_df['log(R1)']] # croped dataframe to contain only the CEE phase
    mt = dummy_df[dummy_df.DM1R<0] # cropped data frame to contain phases where mass is being lost by primary
    mt = mt.loc[mt.index.difference(cee.index)] # isolate the mass transfer phase by removing the rows of CEE

    try:
        mt_age_start = mt.age.iloc[0]  # Start of mass transfer in years
        mt_age_end = mt.age.iloc[-1]   # End of mass transfer in years

        if mt_age_start <= ms_lifetime: # CASE A MASS TRANSFER
            evolution_vector.append(f'{code}0')
            # CASE AB MASS TRANSFER ??? OR IS IT?
            # TODO: ASK JAN (I DON'T THINK THIS WAS RIGHT -
            #  CASE AB is when 2 distinct MT phases)
            #if mt_age_end > ms_lifetime:
            #    evolution_vector.append(f'{code}2')

        elif mt_age_start > ms_lifetime: # CASE B
            evolution_vector.append(f'{code}2')

        if i_start_caseC is not None and  mt_age_start >= dummy_df.age.iloc[i_start_caseC]:
            evolution_vector.append(f'{code}4') # CASE C

        if np.isclose(mt_age_end, dummy_df.age.iloc[-1]): # MT AT DEATH
            evolution_vector.append(f'{code}6')

    except IndexError:
        # catches cases where the data frame is empty if there is not mass transfer
        pass

    try:
        cee_age_start = cee.age.iloc[0] # Start of CEE in years
        cee_age_end = cee.age.iloc[-1] # End of CEE in years

        if cee_age_start <= ms_lifetime:
            evolution_vector.append(f'{code}1')
            #TODO: same question about case AB as above
            #if cee_age_end > ms_lifetime:
            #    evolution_vector.append(f'{code}3')

        elif cee_age_start > ms_lifetime:
            evolution_vector.append(f'{code}3') # CASE B

        if i_start_caseC is not None and cee_age_start >= dummy_df.age.iloc[i_start_caseC]:
            evolution_vector.append(f'{code}5') # CASE C

        if np.isclose(cee_age_end, dummy_df.age.iloc[-1]):
            evolution_vector.append(f'{code}7') # CEE AT DEATH
            FLAG.append("CEE@DEATH")

    except IndexError:
        # catches cases where the data frame is empty if there is not CEE
        pass

    return evolution_vector

# TODO: if this ends up being TUI specific should it go in the tui module?
class UniquePathwaysWithProbaForTUI(HokiObject):
    evolution_vector_dict=evolution_vector_dict
    def __init__(self, overview_df, proba_bool=True, proba_all_bool=True, models_path=None):
        self.proba_bool = proba_bool
        self.proba_all_bool = proba_all_bool

        self.OVERVIEW = overview_df
        self.unique_systems = self.OVERVIEW.SYS_ID.unique()
        # dictionary where the sys IDs will be key and the values will be the evolution vectors
        self.sys_evol_matrix = dict()

        if models_path is None:
            #in logs say we use default
            self.models_path = load.MODELS_PATH
        else:
            # TODO: assert model path is a correct path
            self.models_path = models_path

        for sys_id in self.unique_systems:
            # Load primary evolution table and make the evolution vector
            prim = load.dummy_to_dataframe(self.models_path + self.OVERVIEW[self.OVERVIEW.SYS_ID==sys_id].iloc[0].model_prim)
            evol_vector_prim = make_evolution_vector(prim, PRIM=True)
            # Load secondary evolution table and make the evolution vector
            sec = load.dummy_to_dataframe(self.models_path + self.OVERVIEW[self.OVERVIEW.SYS_ID==sys_id].iloc[0].model_sec)
            evol_vector_sec = make_evolution_vector(sec, PRIM=False)
            # add the two vectors (they're lists) into a tuple | making it immutable is good to conserve order
            self.sys_evol_matrix[sys_id] = tuple(evol_vector_prim + evol_vector_sec)

        # now we need to find the unique tuples. It gives a weird numpy 1D array filled with tuples of different sizes
        # kind of like this:
        # array([('P0', 'P2', 'P3', 'S1', 'S3', 'S7'),
        #        ('P2', 'P6', 'S2', 'S3'), ('P2', 'P6', 'S2', 'S6', 'S3'), [...]
        #        ('P2', 'P6', 'S3', 'S7')], dtype=object)
        self.unique_pathways = np.unique(list(self.sys_evol_matrix.values()))

        # these two things are the same maybe can be streamlined with a function
        if self.proba_bool:
            self.unique_pathways_proba = self._make_dict_proba_per_unique_pathway(proba_column_name='proba')
        if self.proba_all_bool:
            self.unique_pathways_proba_all = self._make_dict_proba_per_unique_pathway(proba_column_name='proba_all')

    def print_summary(self, decimal_places=4):
        for pathway in self.unique_pathways:
            if self.proba_bool:
                proba_pathway = np.round(self.unique_pathways_proba[pathway] * 100, decimal_places)
            else:
                proba_pathway = np.nan
            if self.proba_all_bool:
                proba_all_pathway = np.round(self.unique_pathways_proba_all[pathway] * 100, decimal_places)
            else:
                proba_all_pathway = np.nan

            print("========================")
            print(f"{proba_pathway}% of systems in this population ({proba_all_pathway}% across all pops) "
                  f"have the following evolution")
            print("------------------------")
            decode_and_print_evolution_vector(pathway)

    def _make_dict_proba_per_unique_pathway(self, proba_column_name='proba'):
        unique_pathways_probas_temp = self._make_dict_with_pathway_keys_filled_with_zeros()
        for sys_id in self.unique_systems:
            proba_sys = self.OVERVIEW[self.OVERVIEW.SYS_ID==sys_id][proba_column_name].sum()
            pathway_sys = self.sys_evol_matrix[sys_id]
            unique_pathways_probas_temp[pathway_sys]+=proba_sys
        return unique_pathways_probas_temp

    #this shit is defo a private method
    def _make_dict_with_pathway_keys_filled_with_zeros(self):
        mydict = dict()
        for pathways in self.unique_pathways:
            mydict[pathways] = 0
        return mydict

'''


