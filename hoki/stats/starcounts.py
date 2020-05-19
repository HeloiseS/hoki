import numpy as np
from ..utils.hoki_object import HokiObject
import emcee
import pandas as pd

#TODO: UNITEST
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
    R = n1/n2
    dR = R*np.sqrt((1/n1)+(1/n2))
    return R, dR

#TODO: UNITEST
class UnderlyingCountRatio(HokiObject):
    def __init__(self, n1, n2, name="Undefined Ratio"):
        self.n1 = n1
        self.n2 = n2
        self.phi = None
        self.R_hat = None
        self.n2_hat = None
        self.sampler = None
        self.samples = None
        self.ci_width = None
        self.summary_df = None

    def _lnprior(self, theta):
        R_param, n2_param = theta
        if (R_param <= 0) or (n2_param <= 0):
            return -np.inf
        return (self.phi - 1.0) * np.log(R_param) + (2.0 * self.phi - 1.0) * np.log(n2_param)

    def _lnlikelihood(self, theta, n1, n2):
        R_param, n2_param = theta

        lognumerator = n1 * np.log(R_param) + (n1 + n2) * np.log(n2_param) - n2_param * (R_param + 1.0)
        logdenominator = np.sum(np.log(np.arange(1, n1 + 1))) + np.sum(np.log(np.arange(1, n2 + 1)))

        return lognumerator - logdenominator

    def _lnposterior(self, theta, n1, n2):
        lnprior = self._lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf
        return lnprior + self._lnlikelihood(theta, n1, n2)

    def run_emcee(self, phi=0.5, nwalkers=100, nburnin=500,
                  nsteps=3000, ci_width=68.0):

        self.nwalkers = nwalkers
        self.phi = phi
        self.nburnin = nburnin
        self.nsteps = nsteps
        self.ci_width = ci_width

        R = np.clip(self.n1 / self.n2, 1e-10, 1e5)  # if n1 or n2 are zero, clips to a very large or small number

        # Setting up initial walker positions
        pos = [np.array([R, self.n2]) + 1e-4 * np.random.randn(2) for i in range(nwalkers)]

        # Instanciating the MCMC Sampler
        self.sampler = emcee.EnsembleSampler(nwalkers, 2, self._lnposterior, args=(self.n1, self.n2))

        # Burn in phase
        self.sampler.run_mcmc(pos, nburnin)

        # Record position of the walkers after the burn in
        p1 = self.sampler.chain[:, -1, :]

        # Resetting the MCMC sampler
        self.sampler.reset()

        # Sampling again starting Walkers at their positions at the end of the burn in
        self.sampler.run_mcmc(p1, nsteps)
        self.samples = self.sampler.flatchain  # emcee v2 not v3 be careful to updates

        self.R_hat, self.n2_hat = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                      zip(*np.percentile(self.samples,
                                                         [50 - self.ci_width / 2, 50, 50 + self.ci_width / 2],
                                                         axis=0)))

    @property
    def results_summary(self):
        if not self.phi:
            print("You haven't run the MCMC yet.")
        if self.summary_df:
            return self.summary_df

        columns = ['Variable', f'{int(50 - self.ci_width / 2)}th', '50th', f'{int(50 + self.ci_width / 2)}th']
        self.summary_df = pd.DataFrame(np.array([np.concatenate((np.array(['R_hat']), np.round(self.R_hat, 4))),
                                                 np.concatenate((np.array(['n2_hat']), np.round(self.n2_hat, 4)))]),
                                       columns=columns)
        return self.summary_df
