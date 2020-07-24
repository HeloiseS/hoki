from unittest import TestCase
from hoki import interpolators
from hoki.constants import BPASS_NUM_METALLICITIES, BPASS_TIME_BINS
import numpy as np


class TestGridInterpolator(TestCase):

    def test_init(self):
        for i in [1, 2, 3, 4]:
            grid = np.ones(
                (len(BPASS_NUM_METALLICITIES), len(BPASS_TIME_BINS), i)
            )
            interpolators.GridInterpolator(grid)

        with self.assertRaises(ValueError):
            interpolators.GridInterpolator(grid, metallicities=1)
        with self.assertRaises(ValueError):
            interpolators.GridInterpolator(grid, ages=1)
        with self.assertRaises(ValueError):
            interpolators.GridInterpolator(grid,
                                           metallicities=np.array([1, 2]))
        with self.assertRaises(ValueError):
            interpolators.GridInterpolator(grid,
                                           ages=np.array([1, 2]))
        return
