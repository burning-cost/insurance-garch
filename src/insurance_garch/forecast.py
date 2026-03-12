"""
forecast.py — Scenario generation for pricing review committees.

The output language matters here. We don't say "90th percentile volatility path" —
we say "stressed scenario" and "shocked scenario". Pricing committees and
reserving committees understand that language. The numbers are identical; the
framing makes them actionable.

Design choice: bootstrap simulation by default, not analytical forecasts.
For fat-tailed distributions (Student-t), the analytical approximation
degrades quickly at long horizons. Bootstrap preserves the empirical
innovation distribution.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from insurance_garch.model import GARCHResult, FREQ_ANNUALISE


class ScenarioSet:
    """Collection of volatility scenarios from simulation.

    Attributes
    ----------
    base : pd.Series
        Median simulated volatility path.
    stressed : pd.Series
        90th percentile simulated volatility path.
    shocked : pd.Series
        99th percentile simulated volatility path.
    paths : np.ndarray
        Full simulation array, shape (n_sims, horizon).
    horizon : int
        Forecast horizon in periods.
    frequency : str
        Observation frequency.
    quantiles : dict
        Additional quantile paths (keys are float quantiles 0–1).
    """

    def __init__(
        self,
        paths: np.ndarray,
        horizon: int,
        frequency: str,
    ) -> None:
        if paths.ndim != 2:
            raise ValueError("paths must be 2-dimensional (n_sims, horizon)")
        self.paths = paths
        self.horizon = horizon
        self.frequency = frequency

        # Pre-compute standard percentiles
        self._pctiles = np.percentile(paths, [10, 25, 50, 75, 90, 99], axis=0)
        self.base = pd.Series(self._pctiles[2], name="base_median")
        self.stressed = pd.Series(self._pctiles[4], name="stressed_p90")
        self.shocked = pd.Series(self._pctiles[5], name="shocked_p99")

    def to_dataframe(self) -> pd.DataFrame:
        """Return scenario paths in wide format.

        Returns
        -------
        pd.DataFrame
            Columns: period, p10, p25, base_median, p75, stressed_p90, shocked_p99.
            'period' is 1-indexed forecast horizon.
        """
        return pd.DataFrame({
            "period": np.arange(1, self.horizon + 1),
            "p10": self._pctiles[0],
            "p25": self._pctiles[1],
            "base_median": self._pctiles[2],
            "p75": self._pctiles[3],
            "stressed_p90": self._pctiles[4],
            "shocked_p99": self._pctiles[5],
        })

    def fan_chart(self, ax=None, title: Optional[str] = None):
        """Bank of England-style fan chart of volatility scenarios.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        title : str, optional
            Chart title. Defaults to 'Volatility scenario fan chart'.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        periods = np.arange(1, self.horizon + 1)
        p10, p25, p50, p75, p90, p99 = self._pctiles

        # Fan bands — darker in centre, lighter at extremes
        ax.fill_between(periods, p10, p90, alpha=0.15, color="#2171b5",
                        label="10th–90th percentile")
        ax.fill_between(periods, p25, p75, alpha=0.30, color="#2171b5",
                        label="25th–75th percentile")
        ax.plot(periods, p50, color="#08519c", linewidth=2.0, label="Median (base)")
        ax.plot(periods, p90, color="#fc8d59", linewidth=1.5, linestyle="--",
                label="Stressed (90th pctile)")
        ax.plot(periods, p99, color="#d7301f", linewidth=1.5, linestyle=":",
                label="Shocked (99th pctile)")

        freq_label = {"Q": "quarters", "M": "months", "A": "years"}.get(
            self.frequency, "periods"
        )
        ax.set_xlabel(f"Forecast horizon ({freq_label})")
        ax.set_ylabel("Annualised volatility")
        ax.set_title(title or "Volatility scenario fan chart")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, self.horizon)

        return ax


class VolatilityScenarioGenerator:
    """Generate forward volatility scenarios from a fitted GARCH model.

    Parameters
    ----------
    garch_result : GARCHResult
        A fitted GARCHResult from ClaimsInflationGARCH.fit() or GARCHSelector.best().
    horizon : int, default 8
        Forecast horizon in periods. Default 8 quarters = 2 years, a typical
        UK pricing cycle review window.

    Notes
    -----
    Uses arch's bootstrap simulation, which re-samples from the empirical
    standardised residual distribution. This is more conservative than analytical
    forecasts for fat-tailed distributions and better captures the heavy tails
    observed in claims inflation data.
    """

    def __init__(
        self,
        garch_result: GARCHResult,
        horizon: int = 8,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.garch_result = garch_result
        self.horizon = horizon

    def generate(
        self,
        n_sims: int = 10_000,
        method: str = "bootstrap",
        seed: Optional[int] = None,
    ) -> ScenarioSet:
        """Generate simulated volatility paths.

        Parameters
        ----------
        n_sims : int, default 10_000
            Number of simulation paths.
        method : str, default 'bootstrap'
            Simulation method. 'bootstrap' resamples from empirical residuals.
            'simulation' draws from the fitted parametric distribution.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ScenarioSet
        """
        if method not in ("bootstrap", "simulation"):
            raise ValueError(f"method must be 'bootstrap' or 'simulation', got '{method}'")

        if seed is not None:
            np.random.seed(seed)

        arch_res = self.garch_result.arch_result
        annualise = FREQ_ANNUALISE.get(self.garch_result.frequency, 1) ** 0.5

        try:
            if method == "bootstrap":
                forecasts = arch_res.forecast(
                    horizon=self.horizon,
                    method="bootstrap",
                    simulations=n_sims,
                    reindex=False,
                )
            else:
                forecasts = arch_res.forecast(
                    horizon=self.horizon,
                    method="simulation",
                    simulations=n_sims,
                    reindex=False,
                )

            # forecasts.simulations.variances shape: (n_obs, n_sims, horizon)
            # We want the last observation's forecast
            sim_variances = forecasts.simulations.variances[-1]  # (n_sims, horizon)
            # Convert variance to volatility and annualise
            sim_vols = np.sqrt(np.abs(sim_variances)) * annualise

        except Exception as exc:
            warnings.warn(
                f"Bootstrap forecast failed ({exc}). Falling back to analytical forecast "
                "with perturbation for scenario spread.",
                UserWarning,
                stacklevel=2,
            )
            sim_vols = self._analytical_fallback(n_sims, annualise)

        return ScenarioSet(
            paths=sim_vols,
            horizon=self.horizon,
            frequency=self.garch_result.frequency,
        )

    def _analytical_fallback(self, n_sims: int, annualise: float) -> np.ndarray:
        """Analytical forecast with noise perturbation as a fallback."""
        arch_res = self.garch_result.arch_result
        try:
            forecasts = arch_res.forecast(horizon=self.horizon, reindex=False)
            base_var = forecasts.variance.iloc[-1].values  # (horizon,)
            base_vol = np.sqrt(np.abs(base_var)) * annualise
        except Exception:
            # Last resort: use mean conditional volatility
            base_vol = np.full(
                self.horizon,
                float(self.garch_result.conditional_volatility.mean()),
            )

        # Add cross-sectional noise to create scenario spread
        # Use log-normal perturbation so volatility stays positive
        log_base = np.log(base_vol + 1e-10)
        noise_scale = 0.2 * (1 + np.arange(self.horizon) / self.horizon)
        noise = np.random.randn(n_sims, self.horizon) * noise_scale
        sim_vols = np.exp(log_base + noise)
        return sim_vols
