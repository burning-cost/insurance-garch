"""
backtest.py — VaR-style backtesting for GARCH volatility models.

We implement two standard tests:

1. Kupiec (1995) unconditional coverage: checks that the observed exceedance
   rate matches the nominal alpha. A model that predicts 5% VaR should see
   exceedances ~5% of the time.

2. Christoffersen (1998) conditional coverage: additionally checks that
   exceedances are serially independent. A GARCH model that clusters its
   VaR breaches has failed its primary job — clustering is exactly what
   it's supposed to explain.

The joint conditional coverage test (Christoffersen's LR_cc) combines both.
This is the test regulators care about for internal model validation.

Reference:
    Christoffersen, P.F. (1998). Evaluating interval forecasts.
    International Economic Review, 39(4), 841–862.

    Kupiec, P.H. (1995). Techniques for verifying the accuracy of risk
    measurement models. Journal of Derivatives, 3(2), 73–84.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from insurance_garch.model import GARCHResult, FREQ_ANNUALISE


class BacktestResult:
    """Results from VaR backtesting of a GARCH model.

    Attributes
    ----------
    exceedances : pd.Series
        Binary series (1 = exceedance, 0 = no exceedance).
    alpha : float
        Nominal coverage level used for testing.
    kupiec_stat : float
        Kupiec LR statistic (chi-squared 1 d.f. under H0).
    kupiec_pvalue : float
        p-value for Kupiec test. p < 0.05 indicates model failure.
    christoffersen_stat : float
        Christoffersen independence LR statistic.
    christoffersen_pvalue : float
        p-value for Christoffersen independence test.
    conditional_coverage_stat : float
        Joint conditional coverage LR statistic (2 d.f. under H0).
    conditional_coverage_pvalue : float
        p-value for joint conditional coverage test.
    exceedance_rate : float
        Observed proportion of exceedances.
    """

    def __init__(
        self,
        exceedances: pd.Series,
        alpha: float,
        var_series: pd.Series,
        actual_series: pd.Series,
    ) -> None:
        self.exceedances = exceedances
        self.alpha = alpha
        self.var_series = var_series
        self.actual_series = actual_series

        self.exceedance_rate = float(exceedances.mean())
        n = len(exceedances)
        n1 = int(exceedances.sum())
        n0 = n - n1

        # Kupiec unconditional coverage LR
        # H0: pi = alpha
        # LR_uc = -2 * log[alpha^n1 * (1-alpha)^n0 / pi_hat^n1 * (1-pi_hat)^n0]
        self.kupiec_stat, self.kupiec_pvalue = self._kupiec_test(
            n1, n0, alpha
        )

        # Christoffersen independence LR
        self.christoffersen_stat, self.christoffersen_pvalue = \
            self._christoffersen_independence(exceedances.values)

        # Joint conditional coverage: LR_cc = LR_uc + LR_ind
        self.conditional_coverage_stat = (
            self.kupiec_stat + self.christoffersen_stat
        )
        # LR_cc ~ chi-squared with 2 degrees of freedom under H0
        self.conditional_coverage_pvalue = float(
            1 - stats.chi2.cdf(self.conditional_coverage_stat, df=2)
        )

    @staticmethod
    def _kupiec_test(n1: int, n0: int, alpha: float) -> tuple[float, float]:
        """Compute Kupiec unconditional coverage LR statistic."""
        n = n1 + n0
        if n == 0:
            return 0.0, 1.0

        pi_hat = n1 / n if n > 0 else alpha

        # Avoid log(0)
        eps = 1e-10
        pi_hat = np.clip(pi_hat, eps, 1 - eps)
        alpha_clip = np.clip(alpha, eps, 1 - eps)

        ll_null = n1 * np.log(alpha_clip) + n0 * np.log(1 - alpha_clip)
        ll_alt = n1 * np.log(pi_hat) + n0 * np.log(1 - pi_hat)
        lr_stat = -2 * (ll_null - ll_alt)
        lr_stat = max(0.0, lr_stat)  # numerical stability
        pvalue = float(1 - stats.chi2.cdf(lr_stat, df=1))
        return float(lr_stat), pvalue

    @staticmethod
    def _christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
        """Compute Christoffersen (1998) independence LR statistic.

        Counts transitions in the binary exceedance sequence and tests
        whether P(hit | prev hit) == P(hit | prev no-hit).
        """
        if len(hits) < 2:
            return 0.0, 1.0

        # Count transitions
        n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
        n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
        n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
        n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

        # Conditional probabilities
        eps = 1e-10
        pi01 = n01 / (n00 + n01 + eps)
        pi11 = n11 / (n10 + n11 + eps)
        pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11 + eps)

        pi01 = np.clip(pi01, eps, 1 - eps)
        pi11 = np.clip(pi11, eps, 1 - eps)
        pi_hat = np.clip(pi_hat, eps, 1 - eps)

        # LR independence
        ll_null = (n01 + n11) * np.log(pi_hat) + (n00 + n10) * np.log(1 - pi_hat)
        ll_alt = (
            n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
            n10 * np.log(1 - pi11) + n11 * np.log(pi11)
        )
        lr_stat = -2 * (ll_null - ll_alt)
        lr_stat = max(0.0, lr_stat)
        pvalue = float(1 - stats.chi2.cdf(lr_stat, df=1))
        return float(lr_stat), pvalue

    def summary(self) -> pd.DataFrame:
        """Return backtest statistics as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Rows: test name. Columns: statistic, p_value, pass_5pct.
        """
        rows = [
            {
                "test": "Unconditional coverage (Kupiec 1995)",
                "observed_rate": self.exceedance_rate,
                "nominal_alpha": self.alpha,
                "statistic": self.kupiec_stat,
                "p_value": self.kupiec_pvalue,
                "pass_5pct": self.kupiec_pvalue >= 0.05,
            },
            {
                "test": "Independence (Christoffersen 1998)",
                "observed_rate": self.exceedance_rate,
                "nominal_alpha": self.alpha,
                "statistic": self.christoffersen_stat,
                "p_value": self.christoffersen_pvalue,
                "pass_5pct": self.christoffersen_pvalue >= 0.05,
            },
            {
                "test": "Conditional coverage (Christoffersen 1998)",
                "observed_rate": self.exceedance_rate,
                "nominal_alpha": self.alpha,
                "statistic": self.conditional_coverage_stat,
                "p_value": self.conditional_coverage_pvalue,
                "pass_5pct": self.conditional_coverage_pvalue >= 0.05,
            },
        ]
        return pd.DataFrame(rows)


class GARCHBacktest:
    """Rolling-window VaR exceedance backtesting for a fitted GARCH model.

    For each time point t in the out-of-sample window, we use GARCH conditional
    volatility to compute a one-step-ahead VaR and test whether the realised
    return exceeds it.

    Parameters
    ----------
    garch_result : GARCHResult
        A fitted GARCHResult.
    alpha : float, default 0.05
        VaR coverage level. alpha=0.05 means we test the 5% lower tail
        (exceedance = realised return falls below -VaR).

    Notes
    -----
    The backtest uses in-sample conditional volatility (not rolling re-estimation)
    for computational tractability. For a rigorous out-of-sample test, use
    rolling estimation — this requires the Databricks notebook workflow.
    """

    def __init__(
        self,
        garch_result: GARCHResult,
        alpha: float = 0.05,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.garch_result = garch_result
        self.alpha = alpha

    def run(self, burn_in: int = 10) -> BacktestResult:
        """Run the backtest and return results.

        Parameters
        ----------
        burn_in : int, default 10
            Number of initial observations to skip (GARCH needs warm-up
            observations to produce reliable volatility estimates).

        Returns
        -------
        BacktestResult
        """
        series = self.garch_result.series
        cond_vol = self.garch_result.arch_result.conditional_volatility

        # Align indices
        common_idx = series.index.intersection(cond_vol.index)
        if len(common_idx) < burn_in + 5:
            raise ValueError(
                f"Insufficient observations for backtesting. "
                f"Need at least {burn_in + 5} in common between series and "
                f"conditional volatility, got {len(common_idx)}."
            )

        series_aligned = series.loc[common_idx]
        vol_aligned = cond_vol.loc[common_idx]

        # Use post-burn-in window
        test_idx = common_idx[burn_in:]
        series_test = series_aligned.loc[test_idx]
        vol_test = vol_aligned.loc[test_idx]

        # Compute VaR: use normal quantile as approximation even for t-dist
        # (conservative: t-dist has heavier tails, so normal VaR understates risk)
        z_alpha = float(stats.norm.ppf(self.alpha))
        mean_ret = float(series_test.mean())

        # Demeaned VaR
        var_series = pd.Series(
            mean_ret + z_alpha * vol_test.values,
            index=test_idx,
            name=f"var_{int(self.alpha * 100)}pct",
        )

        # Exceedances: actual return below VaR
        exceedances = (series_test < var_series).astype(int)
        exceedances.name = "exceedance"

        n_exc = int(exceedances.sum())
        n_total = len(exceedances)

        if n_total < 20:
            warnings.warn(
                f"Only {n_total} observations in backtest window. "
                "LR test statistics will be unreliable with small samples.",
                UserWarning,
                stacklevel=2,
            )

        return BacktestResult(
            exceedances=exceedances,
            alpha=self.alpha,
            var_series=var_series,
            actual_series=series_test,
        )
