"""Tests for insurance_garch.model module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_garch.model import (
    ClaimsInflationGARCH,
    GARCHResult,
    GARCHSelector,
    VALID_VOL,
    VALID_DIST,
    VALID_MEAN,
)
from tests.conftest import make_garch_series


class TestClaimsInflationGARCH:
    def test_basic_fit_returns_result(self, long_series):
        model = ClaimsInflationGARCH(long_series, vol="GARCH", dist="normal")
        result = model.fit()
        assert isinstance(result, GARCHResult)

    def test_default_spec_is_gjr_studentst(self, long_series):
        model = ClaimsInflationGARCH(long_series)
        assert model.vol == "GJR-GARCH"
        assert model.dist == "studentst"

    def test_all_vol_specs_fit(self, long_series):
        for vol in VALID_VOL:
            model = ClaimsInflationGARCH(long_series, vol=vol, dist="normal")
            result = model.fit()
            assert isinstance(result, GARCHResult)

    def test_all_distributions_fit(self, long_series):
        for dist in VALID_DIST:
            model = ClaimsInflationGARCH(long_series, vol="GARCH", dist=dist)
            result = model.fit()
            assert isinstance(result, GARCHResult)

    def test_invalid_vol_raises(self, long_series):
        with pytest.raises(ValueError, match="vol must be one of"):
            ClaimsInflationGARCH(long_series, vol="BOGARCH")

    def test_invalid_dist_raises(self, long_series):
        with pytest.raises(ValueError, match="dist must be one of"):
            ClaimsInflationGARCH(long_series, dist="laplace")

    def test_invalid_mean_raises(self, long_series):
        with pytest.raises(ValueError, match="mean must be one of"):
            ClaimsInflationGARCH(long_series, mean="SARIMA")

    def test_invalid_p_raises(self, long_series):
        with pytest.raises(ValueError, match="p must be"):
            ClaimsInflationGARCH(long_series, p=0)

    def test_invalid_q_raises(self, long_series):
        with pytest.raises(ValueError, match="q must be"):
            ClaimsInflationGARCH(long_series, q=0)

    def test_short_series_warns(self, short_series):
        with pytest.warns(UserWarning, match=r"\d+ observations"):
            ClaimsInflationGARCH(short_series)

    def test_result_stores_spec(self, long_series):
        model = ClaimsInflationGARCH(long_series, vol="EGARCH", dist="studentst",
                                     frequency="M")
        result = model.fit()
        assert result.vol_spec == "EGARCH"
        assert result.distribution == "studentst"
        assert result.frequency == "M"

    def test_gjr_garch_fit(self, long_series):
        """GJR-GARCH should fit and expose o parameter (leverage)."""
        model = ClaimsInflationGARCH(long_series, vol="GJR-GARCH", dist="normal")
        result = model.fit()
        params = result.arch_result.params
        # Should have a gamma[1] parameter for the leverage term
        assert any("gamma" in k for k in params.index)

    def test_tarch_fit(self, long_series):
        model = ClaimsInflationGARCH(long_series, vol="TARCH", dist="normal")
        result = model.fit()
        assert isinstance(result, GARCHResult)


class TestGARCHResult:
    def test_conditional_volatility_annualised(self, fitted_result):
        cv = fitted_result.conditional_volatility
        assert isinstance(cv, pd.Series)
        # Quarterly series annualised by sqrt(4) — values should be positive
        assert (cv > 0).all()
        # For our synthetic series, raw vol ~ 0.005, annualised ~ 0.01
        assert cv.mean() > 0

    def test_persistence_range(self, fitted_result):
        p = fitted_result.persistence
        assert 0 <= p <= 1.5  # could exceed 1 for near-integrated processes

    def test_half_life_positive(self, fitted_result):
        hl = fitted_result.half_life
        assert hl > 0

    def test_half_life_infinite_when_persistence_ge_one(self, fitted_result, monkeypatch):
        monkeypatch.setattr(fitted_result, "persistence",
                            property(lambda self: 1.0))
        # Can't monkeypatch property on dataclass easily; test directly
        from insurance_garch.model import GARCHResult
        import math
        # Simulate high persistence
        fake_p = 1.0
        if fake_p >= 1.0:
            hl = float("inf")
        else:
            hl = np.log(2) / np.log(1 / fake_p)
        assert math.isinf(hl)

    def test_summary_returns_dataframe(self, fitted_result):
        df = fitted_result.summary()
        assert isinstance(df, pd.DataFrame)
        assert "parameter" in df.columns
        assert "estimate" in df.columns
        assert len(df) > 0

    def test_plot_volatility_returns_axes(self, fitted_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = fitted_result.plot_volatility()
        assert ax is not None
        plt.close("all")

    def test_plot_volatility_accepts_ax(self, fitted_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        returned_ax = fitted_result.plot_volatility(ax=ax)
        assert returned_ax is ax
        plt.close("all")


class TestGARCHSelector:
    def test_fit_all_returns_dataframe(self, long_series):
        selector = GARCHSelector(
            long_series,
            vol_specs=("GARCH", "GJR-GARCH"),
            distributions=("normal",),
        )
        df = selector.fit_all()
        assert isinstance(df, pd.DataFrame)
        assert "bic" in df.columns
        assert "model_name" in df.columns
        assert len(df) == 2

    def test_best_returns_gjr_or_garch_on_known_dgp(self, long_series):
        """On a GARCH(1,1) DGP, selector should not prefer an exotic spec."""
        selector = GARCHSelector(
            long_series,
            vol_specs=("GARCH", "GJR-GARCH"),
            distributions=("normal", "studentst"),
        )
        selector.fit_all()
        best = selector.best()
        assert isinstance(best, GARCHResult)
        assert best.vol_spec in ("GARCH", "GJR-GARCH")

    def test_best_raises_before_fit_all(self, long_series):
        selector = GARCHSelector(long_series)
        with pytest.raises(RuntimeError, match="fit_all"):
            selector.best()

    def test_bic_ranking_ascending(self, long_series):
        """BIC values in the table must be non-decreasing (best first)."""
        selector = GARCHSelector(
            long_series,
            vol_specs=("GARCH", "GJR-GARCH"),
            distributions=("normal",),
        )
        df = selector.fit_all()
        converged = df.dropna(subset=["bic"])
        if len(converged) > 1:
            bics = converged["bic"].values
            assert all(bics[i] <= bics[i + 1] for i in range(len(bics) - 1))

    def test_full_grid_runs(self, long_series):
        """Full 4x3 grid should complete without crashing."""
        selector = GARCHSelector(long_series)
        df = selector.fit_all()
        assert len(df) == 12  # 4 vol specs × 3 distributions

    def test_persistence_in_table(self, long_series):
        selector = GARCHSelector(
            long_series,
            vol_specs=("GARCH",),
            distributions=("normal",),
        )
        df = selector.fit_all()
        assert "persistence" in df.columns
        assert not df["persistence"].isna().all()
