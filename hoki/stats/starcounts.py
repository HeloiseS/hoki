import numpy as np
from ..utils.hoki_object import HokiObject
import emcee
import pandas as pd
import corner
import matplotlib.pyplot as plt
from hoki.utils.exceptions import HokiTypeError


def ratio_with_poisson_errs(n1, n2):
    """
    Returns Star Count Ratio and its error

    Notes
    -----
    We assume Poisson and independent errors

    Parameters
    ----------
    n1 : int
        Numerator of the star count ratio
    n2 : int
        Denominator of the star count ratio
    Returns
    -------
    R and dR (floats) - Star Count Ratio and its error

    """
    try:
        R = n1/n2
        dR = R*np.sqrt((1/n1)+(1/n2))
    except TypeError as e:
        raise HokiTypeError(f"Make sure you provide ints or floats.")

    return R, dR


class UnderlyingCountRatio(HokiObject):
    """
    Pipeline to calculate the underlying stellar number count ratio from observed stellar numbers

    Parameters
    ----------
    n1 : int
        Numerator of the star count ratio
    n2 : int
        Denominator of the star count ratio
    name : str, optional
        Name of the number count ratio -- will apear at the top of the corner plots. Default = Unnamed Ratio


    Attributes
    ----------
    self.name : str
        Name of the pipeline instance and title of the corner plot
    self.n1 : int
        **Observed** number of stars on the numerator
    self.n2 : int
        **Observed** number of stars on the denominator
    self.phi : float
        Hyperparameter in the prior function
    self.R_hat : tuple of 3 float
        **Underlying** number count ratio. The 3 values in the tuple correspond to the 50th percentile and the upper and
        lower percentile calculated using self.ci_width. For the default value of self.ci_width=68 percent, the 16th and
        84th percentile will be calculated.
    self.n2_hat : tuple of 3 float
        **Underlying** value of n2. The 3 values in the tuple correspond to the 50th percentile and the upper and
        lower percentile calculated using self.ci_width. For the default value of self.ci_width=68 percent, the 16th and
        84th percentile will be calculated.
    self.sampler : emcee.ensemble.EnsembleSampler
    self.samples : numpy.ndarray
        Samples from the MCMC with shape (nsteps*nwalkers, ndimensions)
    self.ci_width : int
        Width of the credible interval.
    self.results_summary : pandas.DataFrame
        Summary of the results in a dataframe.

    """
    def __init__(self, n1, n2, name="Unnamed Ratio"):
        """

        Parameters
        ----------
        n1 : int
            Numerator of the star count ratio
        n2 : int
            Denominator of the star count ratio
        name : str, optional
            Name of the number count ratio -- will apear at the top of the corner plots. Default = Unnamed Ratio

        """
        assert isinstance(n1, float) or isinstance(n1, int), "n1 should be a number"
        assert isinstance(n2, float) or isinstance(n2, int), "n2 should be a number"

        self.n1 = n1
        self.n2 = n2
        self.phi = None
        self.R_hat = None
        self.n2_hat = None
        self.sampler = None
        self.samples = None
        self.ci_width = None
        self.summary_df = None
        self.name=name

    def _lnprior(self, theta):
        """ Natural lof of the prior """
        R_param, n2_param = theta
        if (R_param <= 0) or (n2_param <= 0):
            return -np.inf
        return (self.phi - 1.0) * np.log(R_param) + (2.0 * self.phi - 1.0) * np.log(n2_param)

    def _lnlikelihood(self, theta, n1, n2):
        """ Natural log of the likelihood"""
        R_param, n2_param = theta

        lognumerator = n1 * np.log(R_param) + (n1 + n2) * np.log(n2_param) - n2_param * (R_param + 1.0)
        logdenominator = np.sum(np.log(np.arange(1, n1 + 1))) + np.sum(np.log(np.arange(1, n2 + 1)))

        return lognumerator - logdenominator

    def _lnposterior(self, theta, n1, n2):
        """ Natural log of the posterior """
        lnprior = self._lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf
        return lnprior + self._lnlikelihood(theta, n1, n2)

    def run_emcee(self, nwalkers=100, nburnin=500,
                  nsteps=3000, ci_width=68.0, phi=0.5):
        """
        Runs the MCMC using emcee

        Notes
        -----
        This fills the self.R_hat (the underlying ratio) and self.n2_hat (underlying number count of the denominator).
        Both attribute get filled by a tuple of 3 numbers: the 50th percentile and the upper and lower percentiles
        according to which ci_width was given. For the default ci_width=68, the 16th and 84th percentile are calculated.

        Parameters
        ----------
        nwalkers : int, optional
            Number of walkers in the MCMC. Default is 100.
        nburnin : int, optional
            Number of burn in steps. Default is 500.
        nsteps : int, optional
            Number of steps after the burn in phase. Default is 3000
        ci_width : int or float, optinal
            Width of the credible interval in percent. Default is 68 percent.
        phi : float, optional
            If you don't know what you're doing, don't touch it. See Dorn-Wallenstein & Levesque 2020 for more details.
            Default is 0.5.

        Returns
        -------
        None
        """

        assert phi>=0 and phi<=1, "Phi should be in the range [0,1]"

        self.phi = phi

        self.nwalkers = nwalkers

        self.nburnin = nburnin
        self.nsteps = nsteps
        self.ci_width = ci_width

        R = np.clip(self.n1 / self.n2, 1e-10, 1e5)  # if n1 or n2 are zero, clips to a very large or small number

        # Setting up initial walker positions
        pos = [np.array([R, self.n2]) + 1e-4 * np.random.randn(2) for i in range(nwalkers)]

        # Instanciating the MCMC Sampler
        self.sampler = emcee.EnsembleSampler(nwalkers, 2, self._lnposterior, args=(self.n1, self.n2))

        # Burn in phase
        self.sampler.run_mcmc(pos, nburnin, store=True)

        # Record position of the walkers after the burn in
        #p1 = self.sampler.chain[:, -1, :] #deprecated
        p1 = self.sampler.get_chain()[-1, :, :]

        # Resetting the MCMC sampler
        self.sampler.reset()

        # Sampling again starting Walkers at their positions at the end of the burn in
        self.sampler.run_mcmc(p1, nsteps)
        self.samples = self.sampler.get_chain(flat=True)  # emcee v2 not v3 be careful to updates

        self.R_hat, self.n2_hat = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                      zip(*np.percentile(self.samples,
                                                         [50 - self.ci_width / 2, 50, 50 + self.ci_width / 2],
                                                         axis=0)))

    @property
    def results_summary(self):
        """Creates a summary dataframe"""
        if not self.phi:
            print("You haven't run the MCMC yet.")
        if self.summary_df is not None:
            return self.summary_df

        columns = ['Variable', f'{int(50 - self.ci_width / 2)}th', '50th', f'{int(50 + self.ci_width / 2)}th']
        row1 = np.array(['R_hat', np.round(self.R_hat[2], 4),
                         np.round(self.R_hat[0], 4), np.round(self.R_hat[1], 4)])
        row2 = np.array(['n2_hat', np.round(self.n2_hat[2], 4),
                         np.round(self.n2_hat[1], 4), np.round(self.n2_hat[0], 4)])

        self.summary_df = pd.DataFrame(np.array([row1, row2]), columns=columns)
        return self.summary_df

    def corner_plot(self, output_file='corner_plot.png', show=True):
        """
        Makes a corner plot

        Parameters
        ----------
        output_file : str, optional
            Location of the output file. Default is './corner_plot.png'

        Returns
        -------
        plt.show()

        """
        if not self.phi:
            print("You haven't run the MCMC yet.")

        fig = corner.corner(self.samples, labels=[r'$\hat{R}$', r'$\hat{n}_{2}$'],
                            truths=[self.R_hat[0], self.n2_hat[0]], quantiles=[.16, .84])
        plt.suptitle(f'{self.name}')

        fig.dpi = 200

        for ax in fig.axes:
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(12)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(12)

        if not output_file and show:
            return plt.show()
        if not output_file and not show:
            return fig
        if isinstance(output_file, str):
            plt.savefig(output_file)
            if show:
                return plt.show()
            if not show:
                return fig