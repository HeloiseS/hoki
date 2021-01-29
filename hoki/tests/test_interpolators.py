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
        self.lam2 = np.linspace(1, 10, num=10)
        self.metallicities = np.linspace(1, 10, num=4)
        self.ages = np.linspace(1, 10, num=4)

        self.mock_lam = mock.patch(
            'hoki.interpolators.BPASS_WAVELENGTHS',
            self.lam,
        )
        self.mock_lam2 = mock.patch(
            'hoki.interpolators.BPASS_WAVELENGTHS',
            self.lam2,
        )
        self.mock_constructor_defaults = mock.patch(
            'hoki.interpolators.GridInterpolator.__init__.__defaults__',
            (self.metallicities, self.ages, np.float64)
        )
        self.mock_all_spectra = mock.patch(
            'hoki.load.spectra_all_z',
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

    def test_exceptions(self):
        with self.mock_lam, self.mock_constructor_defaults, \
             self.mock_all_spectra:
            with self.assertRaises(ValueError):
                interpolators.SpectraInterpolator(
                    '', '', lam_min=3, lam_max=2,
                )
        with self.mock_lam2, self.mock_constructor_defaults, \
             self.mock_all_spectra:
            with self.assertRaises(ValueError):
                interpolators.SpectraInterpolator(
                    '', '',
                )
        return

    def test_interpolate(self):
        with self.mock_lam, self.mock_constructor_defaults, \
             self.mock_all_spectra:
            interp = interpolators.SpectraInterpolator(
                '', '',
            )

        # simple case
        res = interp.interpolate(1., 1.)
        self.assertEqual(
            len(res[0]), len(res[1])
        )

        # with mass value
        res = interp.interpolate(1., 1., 1.)

        # multiple values
        res = interp.interpolate(
            np.array([3., 3.]),
            np.array([2., 2.]),
            np.array([1., 2.])
        )
        self.assertEqual(
            len(res[0]), res[1].shape[1]
        )
        self.assertEqual(
            2, res[1].shape[0]
        )
        self.assertTrue(
            np.allclose(2*res[1][0, :], res[1][1, :])
        )
        return


class TestEmissivitiesInterpolator(TestCase):

    def setUp(self):
        # simulate 4 emssivities evaluated at 5 metallicity values and 10 age
        # values
        self.emissivities = np.random.random((5, 10, 4))
        self.metallicities = np.linspace(1, 10, num=5)
        self.ages = np.linspace(1, 10, num=10)

        self.mock_constructor_defaults = mock.patch(
            'hoki.interpolators.GridInterpolator.__init__.__defaults__',
            (self.metallicities, self.ages, np.float64)
        )
        self.mock_all_emissivities = mock.patch(
            'hoki.load.emissivities_all_z',
            return_value=self.emissivities
        )
        return

    def test_init(self):
        with self.mock_constructor_defaults, self.mock_all_emissivities:
            # standard
            interpolators.EmissivitiesInterpolator('', '')
        return

    def test_interpolate(self):
        with self.mock_constructor_defaults, self.mock_all_emissivities:
            interp = interpolators.EmissivitiesInterpolator(
                '', '',
            )

        # simple case
        res = interp.interpolate(1., 1.)
        self.assertEqual(4, len(res))

        # with mass value
        res = interp.interpolate(1., 1., 1.)
        self.assertEqual(4, len(res))

        # multiple values
        res = interp.interpolate(
            np.array([3., 3.]),
            np.array([2., 2.]),
            np.array([1., 2.])
        )
        self.assertTrue(
            np.allclose(2*res[0, :], res[1, :])
        )
        return
