"""
Object to calculate the event rate at a certain
lookback time or over binned lookback time
"""

from hoki.csp.utils import CSP
from hoki.utils.hoki_object import HokiObject

class CSPEventRate(HokiObject, CSP):
    pass

class CSPEventRateOverTime(CSPEventRate):
    pass

class CSPEventRateAtTime(CSPEventRate):
    pass
