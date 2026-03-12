"""
series.py — Inflation series construction for GARCH modelling.

The key insight: raw claims counts mean nothing without exposure. A motor insurer
with a growing book will see rising claims even in benign inflation conditions.
We always work in rates (claims per unit exposure), then take logs for stationarity.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd


MIN_OBSERVATIONS = 40


class ExposureWeightedSeries:
    """Construct a log claims-rate series suitable for GARCH estimation.

    Parameters
    ----------
    claims : pd.Series
        Claims counts or amounts, indexed by period.
    exposure : pd.Series
        Exposure measure (vehicle years, earned premium, etc.), indexed by period.
        Must share the same index as `claims`.
    period : {'Q', 'M', 'A'}
        Observation frequency. Used for annualisation and frequency warnings.
        'Q' = quarterly, 'M' = monthly, 'A' = annual.

    Notes
    -----
    We compute log(claims / exposure). Zero or negative exposure values raise
    ValueError. Zero claims periods are handled by adding a small continuity
    correction (0.5) before logging — this is less bad than dropping the observation.

    Examples
    --------
    >>> import pandas as pd
    >>> claims = pd.Series([100, 120, 95, 110], index=pd.period_range('2020Q1', periods=4, freq='Q'))
    >>> exposure = pd.Series([1000, 1050, 980, 1020], index=pd.period_range('2020Q1', periods=4, freq='Q'))
    >>> ews = ExposureWeightedSeries(claims, exposure, period='Q')
    >>> log_rates = ews.to_series()
    """

    def __init__(
        self,
        claims: pd.Series,
        exposure: pd.Series,
        period: str = "Q",
    ) -> None:
        if not isinstance(claims, pd.Series):
            raise TypeError("claims must be a pd.Series")
        if not isinstance(exposure, pd.Series):
            raise TypeError("exposure must be a pd.Series")
        if not claims.index.equals(exposure.index):
            raise ValueError("claims and exposure must share the same index")
        if (exposure <= 0).any():
            raise ValueError("All exposure values must be positive")

        valid_periods = {"Q", "M", "A"}
        if period not in valid_periods:
            raise ValueError(f"period must be one of {valid_periods}, got '{period}'")

        self.claims = claims.copy()
        self.exposure = exposure.copy()
        self.period = period
        self._series: Optional[pd.Series] = None

    def to_series(self) -> pd.Series:
        """Return the log claims-rate series.

        Returns
        -------
        pd.Series
            Log(claims / exposure), with the same index as the input data.
            Name is set to 'log_claims_rate'.
        """
        if self._series is not None:
            return self._series

        rate = self.claims / self.exposure

        # Zero-claims continuity correction — log(0) is undefined
        zero_mask = rate <= 0
        if zero_mask.any():
            n_zero = zero_mask.sum()
            warnings.warn(
                f"{n_zero} period(s) have zero or negative claims rate. "
                "Applying continuity correction (0.5 / exposure). "
                "Consider whether these are genuine zero periods or data quality issues.",
                UserWarning,
                stacklevel=2,
            )
            # Add 0.5 claims worth of continuity correction
            corrected_claims = self.claims.copy().astype(float)
            corrected_claims[zero_mask] += 0.5
            rate = corrected_claims / self.exposure

        log_rate = np.log(rate)
        log_rate.name = "log_claims_rate"

        n = len(log_rate)
        if n < MIN_OBSERVATIONS:
            warnings.warn(
                f"Series has {n} observations. GARCH estimation requires at least "
                f"{MIN_OBSERVATIONS} for reliable parameter estimates. "
                "Results should be treated with caution.",
                UserWarning,
                stacklevel=2,
            )

        self._series = log_rate
        return self._series

    @classmethod
    def from_trend_result(
        cls,
        trend_result: Any,
        exposure: pd.Series,
        period: str = "Q",
    ) -> "ExposureWeightedSeries":
        """Construct from an insurance-trend TrendResult.

        Extracts the trend-adjusted claims series from a TrendResult object.
        The trend result must have a `.residuals` attribute (pd.Series).

        Parameters
        ----------
        trend_result : object
            Any object with a `.residuals` attribute returning a pd.Series.
            Compatible with insurance-trend library TrendResult.
        exposure : pd.Series
            Exposure series, must share the index of trend_result.residuals.
        period : str
            Observation frequency.

        Returns
        -------
        ExposureWeightedSeries
        """
        if not hasattr(trend_result, "residuals"):
            raise TypeError(
                "trend_result must have a .residuals attribute. "
                "Expected an insurance-trend TrendResult or compatible object."
            )
        residuals = trend_result.residuals
        if not isinstance(residuals, pd.Series):
            raise TypeError("trend_result.residuals must return a pd.Series")

        # Residuals from trend models are already on a log scale typically;
        # exponentiate to get back to claims space, then pass as claims
        # with unit exposure so we model residual variance.
        residual_claims = np.exp(residuals)
        unit_exposure = pd.Series(
            np.ones(len(residuals)),
            index=residuals.index,
            name="exposure",
        )
        return cls(residual_claims, unit_exposure, period=period)


class CalendarYearInflationSeries:
    """Extract a calendar-year inflation series from a development triangle.

    Reads the latest diagonal of a claims development triangle and computes
    period-on-period log changes, which serve as the inflation input to GARCH.

    Parameters
    ----------
    triangle : pd.DataFrame
        Development triangle where rows are accident years (or periods) and
        columns are development periods (0, 1, 2, ...). Values are cumulative
        or incremental paid claims.
    incremental : bool, default True
        If True, triangle values are incremental. If False, values are
        cumulative — the method will difference them.

    Notes
    -----
    Calendar year diagonals correspond to columns where accident_year +
    development_period = constant. We extract the last complete diagonal
    and compute log-changes as the inflation proxy.

    This is a common actuarial construct — the calendar year diagonal
    captures the inflationary environment at each accident + development point
    in time.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        incremental: bool = True,
    ) -> None:
        if not isinstance(triangle, pd.DataFrame):
            raise TypeError("triangle must be a pd.DataFrame")
        if triangle.empty:
            raise ValueError("triangle must not be empty")

        self.triangle = triangle.copy()
        self.incremental = incremental
        self._series: Optional[pd.Series] = None

    def to_series(self) -> pd.Series:
        """Extract calendar-year diagonal values and return as log-change series.

        Returns
        -------
        pd.Series
            Log changes in calendar-year diagonal totals. Indexed by calendar
            year (accident_year + development_period). Name is 'cy_inflation'.
        """
        if self._series is not None:
            return self._series

        tri = self.triangle

        if not self.incremental:
            # Convert cumulative to incremental by differencing across columns
            tri = tri.diff(axis=1)
            tri.iloc[:, 0] = self.triangle.iloc[:, 0]

        # Extract calendar year diagonals
        # For each calendar year k: sum of tri[row, col] where row_idx + col_idx = k
        row_labels = np.arange(len(tri))
        col_labels = np.arange(len(tri.columns))

        calendar_years: dict[int, float] = {}
        for i, row_idx in enumerate(tri.index):
            for j, col_name in enumerate(tri.columns):
                cy = i + j
                val = tri.loc[row_idx, col_name]
                if pd.notna(val):
                    calendar_years[cy] = calendar_years.get(cy, 0.0) + val

        if not calendar_years:
            raise ValueError("No valid data found in triangle diagonals")

        cy_series = pd.Series(calendar_years, name="cy_totals").sort_index()

        # Log-change series
        log_vals = np.log(cy_series.clip(lower=1e-10))
        log_changes = log_vals.diff().dropna()
        log_changes.name = "cy_inflation"

        n = len(log_changes)
        if n < MIN_OBSERVATIONS:
            warnings.warn(
                f"Calendar year inflation series has {n} observations. "
                f"GARCH estimation needs at least {MIN_OBSERVATIONS}. "
                "Consider supplementing with market inflation data.",
                UserWarning,
                stacklevel=2,
            )

        self._series = log_changes
        return self._series
