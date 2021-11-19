import pandas as pd
import numpy as np
from hoki import load
from hoki.utils.exceptions import HokiFatalError, HokiUserWarning, HokiFormatError
from hoki.utils.hoki_object import HokiObject

#TODO: SHOULD ADD LOGGING TO THIS?

#TODO: rename to reflect the fact it's a dictionary
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




