"""Tests for insurance_garch.series module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_garch.series import (
    ExposureWeightedSeries,
    CalendarYearInflationSeries,
    MIN_OBSERVATIONS,
)
from tests.conftest import make_exposure_series


class TestExposureWeightedSeries:
    def test_basic_construction(self):
        claims, exposure = make_exposure_series(n=60)
        ews = ExposureWeightedSeries(claims, exposure, period="Q")
        s = ews.to_series()
        assert isinstance(s, pd.Series)
        assert len(s) == 60
        assert s.name == "log_claims_rate"

    def test_log_rate_is_log_of_claims_over_exposure(self):
        idx = pd.period_range("2020Q1", periods=4, freq="Q")
        claims = pd.Series([100.0, 200.0, 150.0, 120.0], index=idx)
        exposure = pd.Series([1000.0, 1000.0, 1000.0, 1000.0], index=idx)
        ews = ExposureWeightedSeries(claims, exposure, period="Q")
        s = ews.to_series()
        expected = np.log(claims / exposure)
        np.testing.assert_allclose(s.values, expected.values)

    def test_invalid_period_raises(self):
        claims, exposure = make_exposure_series(n=10)
        with pytest.raises(ValueError, match="period must be one of"):
            ExposureWeightedSeries(claims, exposure, period="W")

    def test_negative_exposure_raises(self):
        idx = pd.period_range("2020Q1", periods=3, freq="Q")
        claims = pd.Series([10.0, 20.0, 15.0], index=idx)
        exposure = pd.Series([100.0, -50.0, 200.0], index=idx)
        with pytest.raises(ValueError, match="positive"):
            ExposureWeightedSeries(claims, exposure, period="Q")

    def test_mismatched_index_raises(self):
        idx1 = pd.period_range("2020Q1", periods=4, freq="Q")
        idx2 = pd.period_range("2021Q1", periods=4, freq="Q")
        claims = pd.Series([10.0] * 4, index=idx1)
        exposure = pd.Series([100.0] * 4, index=idx2)
        with pytest.raises(ValueError, match="same index"):
            ExposureWeightedSeries(claims, exposure, period="Q")

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            ExposureWeightedSeries([1, 2, 3], [4, 5, 6])

    def test_short_series_warns(self):
        claims, exposure = make_exposure_series(n=20)
        ews = ExposureWeightedSeries(claims, exposure, period="Q")
        with pytest.warns(UserWarning, match=r"20 observations"):
            ews.to_series()

    def test_zero_claims_continuity_correction(self):
        idx = pd.period_range("2020Q1", periods=4, freq="Q")
        claims = pd.Series([0.0, 100.0, 150.0, 120.0], index=idx)
        exposure = pd.Series([1000.0] * 4, index=idx)
        ews = ExposureWeightedSeries(claims, exposure, period="Q")
        with pytest.warns(UserWarning, match="continuity correction"):
            s = ews.to_series()
        # Should not contain -inf
        assert np.all(np.isfinite(s.values))

    def test_caching(self):
        claims, exposure = make_exposure_series(n=60)
        ews = ExposureWeightedSeries(claims, exposure, period="Q")
        s1 = ews.to_series()
        s2 = ews.to_series()
        assert s1 is s2  # same object returned from cache

    def test_from_trend_result(self):
        claims, exposure = make_exposure_series(n=60)
        idx = claims.index

        class MockTrendResult:
            residuals = pd.Series(np.random.normal(0, 0.01, 60), index=idx)

        ews = ExposureWeightedSeries.from_trend_result(
            MockTrendResult(), exposure, period="Q"
        )
        assert isinstance(ews, ExposureWeightedSeries)

    def test_from_trend_result_no_residuals_raises(self):
        with pytest.raises(TypeError, match="residuals"):
            ExposureWeightedSeries.from_trend_result(object(), pd.Series())

    def test_from_trend_result_non_series_residuals_raises(self):
        class BadTrend:
            residuals = [1, 2, 3]

        with pytest.raises(TypeError, match="pd.Series"):
            ExposureWeightedSeries.from_trend_result(BadTrend(), pd.Series())


class TestCalendarYearInflationSeries:
    def _make_triangle(self, n_rows=10, n_cols=8) -> pd.DataFrame:
        """Small incremental triangle with positive values."""
        rng = np.random.default_rng(99)
        data = np.abs(rng.normal(100, 10, size=(n_rows, n_cols)))
        idx = range(2010, 2010 + n_rows)
        cols = range(0, n_cols)
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_basic_construction(self):
        tri = self._make_triangle()
        cys = CalendarYearInflationSeries(tri)
        s = cys.to_series()
        assert isinstance(s, pd.Series)
        assert s.name == "cy_inflation"
        assert len(s) > 0

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="pd.DataFrame"):
            CalendarYearInflationSeries([[1, 2], [3, 4]])

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            CalendarYearInflationSeries(pd.DataFrame())

    def test_cumulative_triangle(self):
        """Cumulative triangle should produce similar log-changes after differencing."""
        tri_inc = self._make_triangle()
        # Convert to cumulative
        tri_cum = tri_inc.cumsum(axis=1)
        cys_inc = CalendarYearInflationSeries(tri_inc, incremental=True)
        cys_cum = CalendarYearInflationSeries(tri_cum, incremental=False)
        s_inc = cys_inc.to_series()
        s_cum = cys_cum.to_series()
        # Both should return a pd.Series
        assert isinstance(s_inc, pd.Series)
        assert isinstance(s_cum, pd.Series)

    def test_caching(self):
        tri = self._make_triangle()
        cys = CalendarYearInflationSeries(tri)
        s1 = cys.to_series()
        s2 = cys.to_series()
        assert s1 is s2

    def test_short_triangle_warns(self):
        """A small triangle produces few calendar years — should warn."""
        tri = self._make_triangle(n_rows=5, n_cols=4)
        cys = CalendarYearInflationSeries(tri)
        with pytest.warns(UserWarning, match=r"\d+ observations"):
            cys.to_series()
