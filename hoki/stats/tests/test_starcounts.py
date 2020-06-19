from hoki.stats.starcounts import UnderlyingCountRatio, ratio_with_poisson_errs
import pandas as pd
import numpy as np
from hoki.utils.exceptions import HokiTypeError
import pytest


def test_ratio_with_poisson_errors():
    R, dR = ratio_with_poisson_errs(2,4)
    assert np.isclose(R, 0.50, atol=0.005), "Ratio is wrong"
    assert np.isclose(dR, 0.43, atol=0.005), "Errors are wrong"


def test_ratio_with_poisson_errors_fail():
    with pytest.raises(HokiTypeError):
        __, __ = ratio_with_poisson_errs('bla', 4), 'HokiTypeError should be raised'


class TestUnderlyingCountRatio(object):
    def test_init(self):
        ratio = UnderlyingCountRatio(2,4)
        assert int(ratio.n1) == 2, "instanciation messed up"

    ratio = UnderlyingCountRatio(2, 4)

    def test_run_emcee(self):
        self.ratio.run_emcee()

    def test_corner_plot(self):
        self.ratio.corner_plot(show=False)

    def test_summary_df(self):
        summary = self.ratio.results_summary
        assert isinstance(summary, pd.DataFrame)
        assert summary.columns[0]=='Variable'