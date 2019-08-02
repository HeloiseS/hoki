import numpy as np 

BPASS_TIME_BINS = np.arange(6, 11.1, 0.1)
BPASS_TIME_INTERVALS = np.array([10**(t+0.05) - 10**(t-0.05) for t in BPASS_TIME_BINS])
BPASS_TIME_WEIGHT_GRID = np.array([np.zeros((100,100)) + dt for dt in BPASS_TIME_INTERVALS])


