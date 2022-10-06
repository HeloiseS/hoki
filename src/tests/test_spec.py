"""
Tests for the `hoki.spec` module.

Authors: Martin Glatzle
"""
from unittest import TestCase
from hoki import spec
import numpy as np


class TestBinwiseTrapz(TestCase):

    def test_std(self):
        x = np.linspace(1, 10)
        y = np.ones_like(x).reshape((1, -1))
        bin_edges = np.array([4, 5])
        y_new = spec._binwise_trapz_sorted(x, y, bin_edges)
        self.assertAlmostEqual(
            1.0, y_new[0, 0]
        )
        return

    def test_edge_cases(self):
        lam = np.linspace(1, 10)
        L_lam = np.ones_like(lam).reshape(1, -1)

        # bin edges at ends of lam
        bin_edges = np.array([lam[0], (lam[0]+lam[-1])/2, lam[-1]])
        L_bin_int = spec._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 1]/np.diff(bin_edges)[1]
        )

        # only one bin
        bin_edges = np.array([lam[0], lam[-1]])
        L_bin_int = spec._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )

        # one bin smaller than resolution
        bin_edges = np.array([lam[0], (lam[0]+lam[1])/2])
        L_bin_int = spec._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )

        # one bin covering neighbouring resolution elements
        bin_edges = np.array([lam[0], (lam[1]+lam[2])/2])
        L_bin_int = spec._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            1.0, L_bin_int[0, 0]/np.diff(bin_edges)[0]
        )
        return

    def test_L_conservation(self):
        lam = np.linspace(1, 5, num=50)
        L_lam = np.random.random((4, len(lam)))
        bin_edges = np.array([lam[0], lam[-1]])
        L_bin_int = spec._binwise_trapz_sorted(lam, L_lam, bin_edges)
        for j, row in enumerate(L_lam):
            self.assertAlmostEqual(
                np.trapz(row, x=lam), L_bin_int[j, 0]
            )
        return

    def test_L_conservation_2(self):
        lam = np.linspace(1, 5, num=50)
        L_lam = np.random.random((1, len(lam)))
        bin_edges = np.array([lam[0], (lam[0]+lam[-1])/2, lam[-1]])
        L_bin_int = spec._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            np.trapz(L_lam[0, :], x=lam),
            np.sum(L_bin_int)
        )
        return

    def test_L_conservation_3(self):
        lam = np.linspace(1, 5, num=50)
        L_lam = np.random.random((1, len(lam)))
        bin_edges = np.array([lam[4], lam[20]])
        L_bin_int = spec._binwise_trapz_sorted(lam, L_lam, bin_edges)
        self.assertAlmostEqual(
            np.trapz(L_lam[0, 4:21], x=lam[4:21]),
            L_bin_int[0, 0]
        )
        return


class TestBinSpectra(TestCase):

    def test_std(self):
        wl = np.linspace(1, 1000, num=2000)
        spectra = np.random.random((10, len(wl)))
        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra
        )

        self.assertEqual(
            (spectra.shape[0], 10), spectra_new.shape
        )
        self.assertTrue(
            np.all(spectra_new >= 0)
        )

        bin_edges = np.linspace(100, 500, num=500)
        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra, bin_edges
        )

        self.assertTrue(
            np.allclose(0.5*(bin_edges[1:]+bin_edges[:-1]), wl_new)
        )
        self.assertEqual(
            (spectra.shape[0], len(bin_edges)-1), spectra_new.shape
        )
        self.assertTrue(
            np.all(spectra_new >= 0)
        )
        return

    def test_L_conservation(self):
        wl = np.linspace(1, 1000, num=20000, endpoint=True)
        spectra = np.random.random((20, len(wl)))
        bin_edges = np.linspace(1, 1000, num=10, endpoint=True)

        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra, bin_edges
        )
        self.assertTrue(
            np.allclose(
                np.trapz(spectra, x=wl, axis=1),
                np.sum(spectra_new*np.diff(bin_edges), axis=1)
            )
        )

        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra
        )
        self.assertTrue(
            np.allclose(
                np.trapz(spectra, x=wl, axis=1),
                np.sum(spectra_new, axis=1)*(wl[-1]-wl[0])/10
            )
        )

        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra, 1
        )
        self.assertTrue(
            np.allclose(
                np.trapz(spectra, x=wl, axis=1),
                np.sum(spectra_new, axis=1)*(wl[-1]-wl[0])
            )
        )

        return

    def test_L_conservation_desc(self):
        wl = np.linspace(1000, 1, num=20000, endpoint=True)
        spectra = np.random.random((20, len(wl)))

        bin_edges = np.linspace(1000, 1, num=10, endpoint=True)
        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra, bin_edges
        )
        self.assertTrue(
            np.allclose(
                np.trapz(spectra, x=wl, axis=1),
                np.sum(spectra_new*np.diff(bin_edges), axis=1)
            )
        )

        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra
        )
        self.assertTrue(
            np.allclose(
                np.trapz(spectra, x=wl, axis=1),
                np.sum(spectra_new, axis=1)*(wl[-1]-wl[0])/10
            )
        )

        wl_new, spectra_new = spec.bin_luminosity(
            wl, spectra, 1
        )
        self.assertTrue(
            np.allclose(
                np.trapz(spectra, x=wl, axis=1),
                np.sum(spectra_new, axis=1)*(wl[-1]-wl[0])
            )
        )

        return

    def test_exceptions(self):
        # wrong dimensionality
        wl = np.linspace(1, 100)
        sed = np.empty((len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            spec.bin_luminosity(wl, sed, bin_edges)

        # incompatible shapes
        wl = np.linspace(1, 100)
        sed = np.empty((1, len(wl)-2))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            spec.bin_luminosity(wl, sed, bin_edges)

        # identical values in wl
        wl = np.hstack((
            np.linspace(1, 100),
            np.linspace(1, 100)
        ))
        sed = np.empty((1, len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            spec.bin_luminosity(wl, sed, bin_edges)

        # bins outside range
        wl = np.linspace(1, 100)
        sed = np.empty((1, len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            spec.bin_luminosity(wl, sed, bin_edges)
        bin_edges = np.array([2, 200])
        with self.assertRaises(ValueError):
            spec.bin_luminosity(wl, sed, bin_edges)
        wl = np.linspace(100, 1)
        sed = np.empty((1, len(wl)))
        bin_edges = np.array([0.5, 20])
        with self.assertRaises(ValueError):
            spec.bin_luminosity(wl, sed, bin_edges)
        return
