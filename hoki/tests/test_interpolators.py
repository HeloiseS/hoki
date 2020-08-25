"""
Tests for the interpolators module.

Author: Martin Glatzle
"""
from unittest import TestCase, mock
from hoki import interpolators
from hoki.constants import (
    BPASS_NUM_METALLICITIES, BPASS_TIME_BINS
)
import numpy as np


class TestGridInterpolator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tested_class = interpolators.GridInterpolator
        return

    def test_init(self):
        for i in [1, 2, 3, 4]:
            grid = np.ones(
                (len(BPASS_NUM_METALLICITIES), len(BPASS_TIME_BINS), i)
            )
            self.__class__.tested_class(grid)

        with self.assertRaises(ValueError):
            self.__class__.tested_class(grid, metallicities=1)
        with self.assertRaises(ValueError):
            self.__class__.tested_class(grid, ages=1)
        with self.assertRaises(ValueError):
            self.__class__.tested_class(grid,
                                        metallicities=np.array([1, 2]))
        with self.assertRaises(ValueError):
            self.__class__.tested_class(grid,
                                        ages=np.array([1, 2]))
        return

    def test__interpolate(self):
        Z = np.array([1, 2, 3, 4])
        ages = np.array([1e2, 1e3, 1e4])
        grid = np.random.random((len(Z), len(ages), 1))
        interpolator = self.__class__.tested_class(
            grid, metallicities=Z, ages=ages
        )

        # single value
        res = interpolator._interpolate(2., 1e3)
        self.assertEqual(
            (1,), res.shape
        )
        self.assertAlmostEqual(
            grid[1, 1, 0], res
        )
        # in between points
        res = interpolator._interpolate(1.5, 550.)
        self.assertEqual(
            (1,), res.shape
        )
        # Delaunay triangulation is not unique for regular grid, can get one of
        # two possible answers:
        opt1 = 0.5*(
            grid[0, 0, 0] +
            grid[1, 1, 0]
        )
        opt2 = 0.5*(
            grid[1, 0, 0] +
            grid[0, 1, 0]
        )
        self.assertTrue(
            np.allclose(opt1, res) or np.allclose(opt2, res)
        )

        # array of values
        res = interpolator._interpolate(
            np.array([2., 3.]), np.array([1e2, 1e3])
        )
        self.assertEqual(
            (2, 1), res.shape
        )
        self.assertAlmostEqual(
            grid[1, 0, 0], res[0, 0]
        )
        self.assertAlmostEqual(
            grid[2, 1, 0], res[1, 0]
        )

        # array valued qtty
        grid = np.random.random((len(Z), len(ages), 3))
        interpolator = self.__class__.tested_class(
            grid, metallicities=Z, ages=ages
        )
        res = interpolator._interpolate(2., 1e3)
        self.assertEqual(
            (3,), res.shape
        )
        self.assertTrue(
            np.allclose(grid[1, 1, :], res)
        )
        return


class TestGridInterpolatorMassScaled(TestGridInterpolator):

    @classmethod
    def setUpClass(cls):
        cls.tested_class = interpolators.GridInterpolatorMassScaled
        return

    def test__interpolate_with_masses(self):
        Z = np.array([1, 2, 3, 4])
        ages = np.array([1e2, 1e3, 1e4])
        grid = np.random.random((len(Z), len(ages), 1))
        interpolator = self.__class__.tested_class(
            grid, metallicities=Z, ages=ages
        )

        # single value
        res = interpolator._interpolate(2., 1e3, masses=2)
        self.assertEqual(
            (1,), res.shape
        )
        self.assertAlmostEqual(
            2*grid[1, 1, 0], res
        )

        # array of values
        res = interpolator._interpolate(
            np.array([2., 3.]), np.array([1e2, 1e3]), masses=np.array([1, 2])
        )
        self.assertEqual(
            (2, 1), res.shape
        )
        self.assertAlmostEqual(
            grid[1, 0, 0], res[0, 0]
        )
        self.assertAlmostEqual(
            2*grid[2, 1, 0], res[1, 0]
        )

        # array valued qtty
        grid = np.random.random((len(Z), len(ages), 3))
        interpolator = self.__class__.tested_class(
            grid, metallicities=Z, ages=ages
        )
        res = interpolator._interpolate(2., 1e3, masses=2.5)
        self.assertEqual(
            (3,), res.shape
        )
        self.assertTrue(
            np.allclose(2.5*grid[1, 1, :], res)
        )
        return


class TestSpectraInterpolator(TestCase):

    def setUp(self):
        self.spectra = np.random.random((4, 4, 100))
        self.lam = np.linspace(1, 10, num=100)
        self.metallicities = np.linspace(1, 10, num=4)
        self.ages = np.linspace(1, 10, num=4)

        self.mock_lam = mock.patch.multiple(
            'hoki.interpolators',
            BPASS_WAVELENGTHS=self.lam,
        )
        self.mock_constructor_defaults = mock.patch(
            'hoki.interpolators.GridInterpolator.__init__.__defaults__',
            (self.metallicities, self.ages, np.float64)
        )
        self.mock_all_spectra = mock.patch(
            'hoki.load.all_spectra',
            return_value=self.spectra
        )

        return

    def test_init(self):
        with self.mock_lam, self.mock_constructor_defaults, \
             self.mock_all_spectra:
            # standard
            interpolators.SpectraInterpolator(
                '', '',
            )
            # limit lam range
            interpolators.SpectraInterpolator(
                '', '',
                lam_min=3., lam_max=5.,
            )
        return

    def test_interpolate(self):
        with self.mock_lam, self.mock_constructor_defaults, \
             self.mock_all_spectra:
            interp = interpolators.SpectraInterpolator(
                '', '',
            )
        res = interp.interpolate(1., 1.)
        self.assertEqual(
            len(res[0]), len(res[1])
        )
        interp.interpolate(1., 1., 1.)
        res = interp.interpolate(
            np.array([1., 2.]),
            np.array([1., 2.]),
            np.array([1., 2.])
        )
        self.assertEqual(
            len(res[0]), res[1].shape[1]
        )
        return
