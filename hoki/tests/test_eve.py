from hoki.eve import Eve
import pkg_resources
import pytest
import pandas as pd
import numpy as np
from hoki.constants import *
from hoki.utils.exceptions import HokiFatalError

class TestEve(object):
    def test_bad_met(self):
        with pytest.raises(HokiFatalError) as e_info:
            eve =  Eve(met='010', eve_path='../../../EvE/EvE.hdf5')
            # THIS TEST ONLY "WORKS" BECAUSE METALLICITY CHECK DONE FIRST
            # BUT IT THAT IS CHANGED IN THE FUTURE IT WILL EQUALLY FALL APART
            # IF THE PATH IS INCORRECT.

    def test_bad_path(self):
        with pytest.raises(HokiFatalError) as e_info:
            eve = Eve(met='z010', eve_path='sadjkfh')

