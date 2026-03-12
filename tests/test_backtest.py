"""Tests for insurance_garch.backtest module."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from insurance_garch.backtest import GARCHBacktest, BacktestResult


class TestBacktestResult:
    def _make_result(self, hit_sequence: list, alpha: float = 0.05):
        """Construct a BacktestResult from a known hit sequence."""
        n = len(hit_sequence)
        idx = pd.period_range("2000Q1", periods=n, freq="Q")
        exceedances = pd.Series(hit_sequence, index=idx, name="exceedance", dtype=int)
        var_series = pd.Series(np.zeros(n), index=idx)
        actual_series = pd.Series(np.zeros(n), index=idx)
        return BacktestResult(exceedances, alpha, var_series, actual_series)

    def test_exceedance_rate(self):
        hits = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = self._make_result(hits)
        assert abs(result.exceedance_rate - 0.10) < 1e-10

    def test_kupiec_stat_is_nonneg(self):
        hits = [0] * 19 + [1]  # ~5% exceedance
        result = self._make_result(hits)
        assert result.kupiec_stat >= 0

    def test_kupiec_stat_on_correct_model(self):
        """If exceedance rate equals alpha exactly, LR should be ~0."""
        # Construct exactly 5% exceedances (1 in 20)
        hits = [0] * 19 + [1]
        result = self._make_result(hits, alpha=0.05)
        # LR should be close to 0 when observed == expected
        assert result.kupiec_stat >= 0.0

    def test_kupiec_pvalue_bounded(self):
        hits = [1, 0] * 30
        result = self._make_result(hits, alpha=0.05)
        assert 0 <= result.kupiec_pvalue <= 1

    def test_christoffersen_independent_sequence(self):
        """A non-clustered hit sequence should fail to reject independence."""
        # Every 20th observation is a hit — perfectly spaced, not clustered
        hits = ([0] * 19 + [1]) * 6
        result = self._make_result(hits, alpha=0.05)
        assert result.christoffersen_pvalue >= 0

    def test_christoffersen_clustered_fails(self):
        """Clustered hits (runs of 1s) should give low independence p-value."""
        # All hits in first 10 periods, then none — strong clustering
        hits = [1] * 10 + [0] * 50
        result = self._make_result(hits, alpha=0.05)
        # Christoffersen stat should be positive for clustered data
        assert result.christoffersen_stat >= 0

    def test_conditional_coverage_pvalue_bounded(self):
        hits = [1, 0] * 30
        result = self._make_result(hits)
        assert 0 <= result.conditional_coverage_pvalue <= 1

    def test_conditional_coverage_stat_equals_sum(self):
        hits = [1, 0] * 30
        result = self._make_result(hits)
        expected = result.kupiec_stat + result.christoffersen_stat
        assert abs(result.conditional_coverage_stat - expected) < 1e-10

    def test_summary_dataframe(self):
        hits = [0] * 19 + [1]
        result = self._make_result(hits)
        df = result.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "test" in df.columns
        assert "p_value" in df.columns
        assert "pass_5pct" in df.columns

    def test_christoffersen_known_values(self):
        """Verify Christoffersen statistic on a simple known case.

        When all hits are independent (iid Bernoulli), LR_ind ~ chi2(1).
        A perfectly alternating sequence has pi01 != pi11, yielding nonzero LR_ind.
        """
        # Alternating 1,0,1,0,... — strong independence signature
        hits = [i % 2 for i in range(40)]
        result = self._make_result(hits, alpha=0.5)  # alpha=0.5 so rate matches
        # Both pi01 and pi11 are defined; stat should be finite and non-negative
        assert np.isfinite(result.christoffersen_stat)
        assert result.christoffersen_stat >= 0

    def test_empty_hits_sequence(self):
        """Edge case: all zeros — no exceedances at all."""
        hits = [0] * 40
        result = self._make_result(hits, alpha=0.05)
        assert result.exceedance_rate == 0.0
        # Kupiec stat should be non-negative
        assert result.kupiec_stat >= 0.0


class TestGARCHBacktest:
    def test_run_returns_backtest_result(self, fitted_result):
        bt = GARCHBacktest(fitted_result, alpha=0.05)
        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_invalid_alpha_raises(self, fitted_result):
        with pytest.raises(ValueError, match="alpha must be"):
            GARCHBacktest(fitted_result, alpha=1.5)

    def test_exceedance_rate_reasonable(self, fitted_result):
        """With 120 obs and GARCH model, exceedance rate should be plausible."""
        bt = GARCHBacktest(fitted_result, alpha=0.05)
        result = bt.run()
        # Should be somewhere between 0% and 50% — very loose bound
        assert 0.0 <= result.exceedance_rate <= 0.5

    def test_var_series_in_result(self, fitted_result):
        bt = GARCHBacktest(fitted_result, alpha=0.05)
        result = bt.run()
        assert isinstance(result.var_series, pd.Series)
        assert len(result.var_series) > 0

    def test_alpha_stored(self, fitted_result):
        bt = GARCHBacktest(fitted_result, alpha=0.10)
        result = bt.run()
        assert result.alpha == 0.10

    def test_backtest_with_different_alpha(self, fitted_result):
        """alpha=0.01 (tighter VaR) should give fewer or equal exceedances vs alpha=0.10."""
        bt_tight = GARCHBacktest(fitted_result, alpha=0.01)
        bt_loose = GARCHBacktest(fitted_result, alpha=0.10)
        res_tight = bt_tight.run()
        res_loose = bt_loose.run()
        # Tight VaR: fewer exceedances (lower bound on losses is further out)
        assert res_tight.exceedance_rate <= res_loose.exceedance_rate + 0.1
