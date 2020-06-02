"""
Object to calculate the spectra at a certain
lookback time or over a binned lookback time
"""

from hoki.csp.utils import CSP
from hoki.utils.hoki_object import HokiObject


class CSPSpectra(HokiObject, CSP):
    pass

class CSPSpectraAtTime(CSPSpectra):
    pass

class CSPSpectraOverTime(CSPSpectra):
    pass
