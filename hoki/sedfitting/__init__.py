# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys
from distutils.version import LooseVersion
from hoki.constants import *
from .kvn import KVN
from .lordcommander import LordCommander
from matplotlib import pyplot as plt

__minimum_python_version__ = "3.6"

__all__ = ['KVN','LordCommander', 'plot_voronoi', 'ppxf_plot_fixed', 'ppxf_manual']
__author__ = """Heloise F Stevance"""
__email__ = 'hfstevance@gmail.com'
__version__ = '0.1.0'


def ppxf_manual():
    "opens the ppxf manual page in the browser"
    import webbrowser
    webbrowser.open('https://pypi.org/project/ppxf/')


def ppxf_plot_fixed(ppxf, wl_fits, WL, F, offset_res=0.5):
    """
    The x axis of the ppxf results plot is messed up so this replots it correctly

    Parameters
    ----------
    ppxf: ppxf.ppxf.ppxf
        The ppxf object returned by the fit
    wl_fits: 1D array
         np.exp(loglam) where loglam is the logarithmically rebinned wl axis by ppxf.util.lo_rebin
    WL: 1D array
        the wavelength axis of the observations being fitted
    F: 1D array
        The flux of the osbervations being fitted
    offset_res: float
        Offset of residuals. Default +0.5

    Returns
    -------

    """
    # plot fit on top of data
    plt.plot(wl_fits, ppxf.bestfit,c='crimson', alpha=0.8, label='bestfit')
    plt.plot(WL, F, c='k', zorder=0.5, lw=1, label='data')
    # find residuals
    interp_fit = np.interp(WL, wl_fits, ppxf.bestfit)
    res = F-interp_fit
    # plot residuals
    plt.scatter(WL, res+offset_res, marker='h', s=2, c='grey', alpha=0.3, label='Residuals (+0.5)')
    plt.axhline(offset_res, ls='--', c='k', lw=1)
    plt.xlabel('Wavelength ')
    plt.legend(loc=(0,1.04), ncol=3)
    return res


def plot_voronoi(x, y, counts, pixelsize, ax, cmap='hot', origin=None, **kwargs):
    """
    Plots the voronoi bins

    Parameters
    ----------
    x:
    y:
    counts:
    pixelsize:
    ax:
    cmap:
    origin:

    Returns
    -------

    """

    # Bounds of the image
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # Number of X and Y pixels
    nx = int(round((xmax - xmin)/pixelsize) + 1)
    ny = int(round((ymax - ymin)/pixelsize) + 1)

    # container for the image -- to be filled
    img = np.full((nx, ny), np.nan)  # use nan for missing data

    # list of indeces with size the number of pixels
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)

    # fills the image with the "counts"
    img[j, k] = counts

    # plots the image and returns the mappable needed to make a colour bar if desired
    if origin is not None:
        for_color_bar = ax.imshow(np.rot90(img), interpolation='nearest', cmap=cmap,
               extent=[xmin - pixelsize/2, xmax + pixelsize/2,
                       ymin - pixelsize/2, ymax + pixelsize/2], origin=origin, **kwargs)

    for_color_bar = ax.imshow(np.rot90(img), interpolation='nearest', cmap=cmap,
               extent=[xmin - pixelsize/2, xmax + pixelsize/2,
                       ymin - pixelsize/2, ymax + pixelsize/2], **kwargs)

    return for_color_bar

class UnsupportedPythonError(Exception):
    pass


if LooseVersion(sys.version) < LooseVersion(__minimum_python_version__):
    raise UnsupportedPythonError("hoki does not support Python < {}"
                                 .format(__minimum_python_version__))

#if not _ASTROPY_SETUP_:   # noqa
    # For egg_info test builds to pass, put package imports here.
    #from .example_mod import *   # noqa
    # Then you can be explicit to control what ends up in the namespace,
    #__all__ += ['do_primes']   # noqa
    #__all__ += ['module_input', 'module_output']
    # or you can keep everything from the subpackage with the following instead
    # __all__ += example_mod.__all__
