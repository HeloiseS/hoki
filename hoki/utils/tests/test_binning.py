"""
Tests for the utils.binning module.

Author: Martin Glatzle
"""
from unittest import TestCase
from hoki.utils import binning
import numpy as np


class TestBinwiseTrapz(TestCase):

    def test_std(self):
        x = np.linspace(1, 10)
        y = np.ones_like(x).reshape((1, -1))
        bin_edges = np.array([4, 5])
        y_new = binning._binwise_trapz_sorted(x, y, bin_edges)
        self.assertAlmostEqual(
            1.0, y_new[0, 0]
        )
        return

    def test_edge_cases(self):
        lam = np.linspace(1, 10)
        L_lam = np.ones_like(lam).reshape(1, -1)

        # bin edges at ends of lam
        bin_edges = np.array([lam[0], (lam[0]+lam[-1])/2, lam[-1]])
        L_bin_int = binning._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 1]/np.diff(bin_edges)[1]
        )

        # only one bin
        bin_edges = np.array([lam[0], lam[-1]])
        L_bin_int = binning._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )

        # one bin smaller than resolution
        bin_edges = np.array([lam[0], (lam[0]+lam[1])/2])
        L_bin_int = binning._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )

        # one bin covering neighbouring resolution elements
        bin_edges = np.array([lam[0], (lam[1]+lam[2])/2])
        L_bin_int = binning._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )
        return

    def test_L_conservation(self):
        lam = np.linspace(1, 5, num=50)
        L_lam = np.random.random((4, len(lam)))
        bin_edges = np.array([lam[0], lam[-1]])
        L_bin_int = binning._binwise_trapz_sorted(lam, L_lam, bin_edges)
        for j, row in enumerate(L_lam):
            self.assertAlmostEqual(
                np.trapz(row, x=lam), L_bin_int[j, 0]
            )
        return

    def test_L_conservation_2(self):
        lam = np.linspace(1, 5, num=50)
        L_lam = np.random.random((1, len(lam)))
        bin_edges = np.array([lam[0], (lam[0]+lam[-1])/2, lam[-1]])
        L_bin_int = binning._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            np.trapz(L_lam[0, :], x=lam),
            np.sum(L_bin_int)
        )
        return

    def test_L_conservation_3(self):
        lam = np.linspace(1, 5, num=50)
        L_lam = np.random.random((1, len(lam)))
        bin_edges = np.array([lam[4], lam[20]])
        L_bin_int = binning._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            np.trapz(L_lam[0, 4:21], x=lam[4:21]),
            L_bin_int[0, 0]
        )
        return


class TestBinSpectra(TestCase):

    def test_std(self):
        wl = np.linspace(1, 1000, num=2000)
        SEDs = np.random.random((10, len(wl)))
        bins = np.linspace(100, 500, num=5)
        edges = False
        wl_new, SEDs_new = binning.bin_spectra(
            wl, SEDs, bins, edges=edges
        )

        self.assertTrue(
            np.allclose(bins, wl_new)
        )
        self.assertEqual(
            (SEDs.shape[0], len(bins)), SEDs_new.shape
        )
        self.assertTrue(
            np.all(SEDs_new >= 0)
        )
        return

    def test_L_conservation(self):
        wl = np.linspace(1, 1000, num=20000, endpoint=True)
        SEDs = np.random.random((20, len(wl)))
        bins = np.linspace(1, 1000, num=10, endpoint=True)
        edges = True
        wl_new, SEDs_new = binning.bin_spectra(
            wl, SEDs, bins, edges=edges
        )
        self.assertEqual(
            (SEDs.shape[0], len(bins)-1), SEDs_new.shape
        )
        self.assertTrue(
            np.all(SEDs_new >= 0)
        )
        self.assertTrue(
            np.allclose(
                np.trapz(SEDs, x=wl, axis=1),
                np.sum(SEDs_new*np.diff(bins), axis=1)
            )
        )
        return

    def test_L_conservation_desc(self):
        wl = np.linspace(1000, 1, num=20000, endpoint=True)
        SEDs = np.random.random((20, len(wl)))
        bins = np.linspace(1000, 1, num=10, endpoint=True)
        edges = True
        wl_new, SEDs_new = binning.bin_spectra(
            wl, SEDs, bins, edges=edges
        )
        self.assertEqual(
            (SEDs.shape[0], len(bins)-1), SEDs_new.shape
        )
        self.assertTrue(
            np.all(SEDs_new >= 0)
        )
        self.assertTrue(
            np.all(wl_new > 0)
        )
        self.assertTrue(
            np.allclose(
                np.trapz(SEDs, x=wl, axis=1),
                np.sum(SEDs_new*np.diff(bins), axis=1)
            )
        )
        return

    def test_exceptions(self):
        # wrong dimensionality
        wl = np.linspace(1, 100)
        sed = np.empty((len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            binning.bin_spectra(wl, sed, bin_edges)

        # incompatible shapes
        wl = np.linspace(1, 100)
        sed = np.empty((1, len(wl)-2))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            binning.bin_spectra(wl, sed, bin_edges)

        # identical values in wl
        wl = np.hstack((
            np.linspace(1, 100),
            np.linspace(1, 100)
        ))
        sed = np.empty((1, len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            binning.bin_spectra(wl, sed, bin_edges)

        # bins outside range
        wl = np.linspace(1, 100)
        sed = np.empty((1, len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            binning.bin_spectra(wl, sed, bin_edges, edges=True)
        bin_edges = np.array([2, 200])
        with self.assertRaises(ValueError):
            binning.bin_spectra(wl, sed, bin_edges, edges=True)
        wl = np.linspace(100, 1)
        sed = np.empty((1, len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            binning.bin_spectra(wl, sed, bin_edges, edges=True)
        return