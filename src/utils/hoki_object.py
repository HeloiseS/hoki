from hoki.constants import BPASS_TIME_BINS, BPASS_TIME_INTERVALS, BPASS_TIME_WEIGHT_GRID


class HokiObject(object):
    t = BPASS_TIME_BINS
    dt = BPASS_TIME_INTERVALS
    time_weight_grid = BPASS_TIME_WEIGHT_GRID

    def __init__(self):
        pass
