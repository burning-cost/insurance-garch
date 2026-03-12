"""
Integration tests — end-to-end pipeline from series construction to report.

These tests mirror real usage: a pricing actuary takes a claims/exposure series,
fits a GARCH model, generates scenarios, backtests the model, and produces a
report for the pricing committee.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from insurance_garch.series import ExposureWeightedSeries
from insurance_garch.model import ClaimsInflationGARCH, GARCHSelector
from insurance_garch.forecast import VolatilityScenarioGenerator
from insurance_garch.backtest import GARCHBacktest
from insurance_garch.report import GARCHReport
from tests.conftest import make_garch_series, make_exposure_series


class TestEndToEndPipeline:
    def test_full_pipeline_from_exposure_series(self):
        """claims + exposure -> log series -> GARCH -> scenarios -> backtest -> report."""
        claims, exposure = make_exposure_series(n=80, seed=7)

        # Step 1: construct log series
        ews = ExposureWeightedSeries(claims, exposure, period="Q")
        log_series = ews.to_series()
        assert len(log_series) == 80

        # Step 2: fit GARCH
        model = ClaimsInflationGARCH(log_series, vol="GARCH", dist="normal")
        result = model.fit()
        assert result.persistence >= 0

        # Step 3: generate scenarios
        gen = VolatilityScenarioGenerator(result, horizon=8)
        scenarios = gen.generate(n_sims=500, seed=42)
        assert scenarios.base is not None
        assert (scenarios.shocked.values >= scenarios.base.values).all()

        # Step 4: backtest
        bt = GARCHBacktest(result, alpha=0.05)
        bt_result = bt.run()
        assert isinstance(bt_result.exceedance_rate, float)

        # Step 5: report
        report = GARCHReport(result, scenarios, bt_result)
        d = report.to_dict()
        assert d["model_spec"]["vol_spec"] == "GARCH"

        html = report.to_html()
        assert "<html>" in html

    def test_selector_best_pipeline(self):
        """GARCHSelector -> best model -> scenarios -> report."""
        series = make_garch_series(n=100, seed=17)

        selector = GARCHSelector(
            series,
            vol_specs=("GARCH", "GJR-GARCH"),
            distributions=("normal", "studentst"),
        )
        ranking = selector.fit_all()
        assert len(ranking) == 4

        best = selector.best()
        assert best.vol_spec in ("GARCH", "GJR-GARCH")

        gen = VolatilityScenarioGenerator(best, horizon=4)
        scenarios = gen.generate(n_sims=300, seed=1)

        report = GARCHReport(best, scenarios, bic_table=ranking)
        d = report.to_dict()
        assert "bic_ranking" in d
        assert len(d["bic_ranking"]) <= 5

    def test_egarch_pipeline(self):
        """EGARCH specification should run cleanly end to end."""
        series = make_garch_series(n=80, seed=13)
        model = ClaimsInflationGARCH(series, vol="EGARCH", dist="studentst")
        result = model.fit()

        gen = VolatilityScenarioGenerator(result, horizon=4)
        scenarios = gen.generate(n_sims=200, seed=42)

        bt = GARCHBacktest(result)
        bt_result = bt.run()

        report = GARCHReport(result, scenarios, bt_result)
        html = report.to_html()
        assert "EGARCH" in html

    def test_short_series_warns_but_runs(self):
        """Pipeline should warn (not crash) on a short series."""
        series = make_garch_series(n=30, seed=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = ClaimsInflationGARCH(series, vol="GARCH", dist="normal")
            result = model.fit()
            warning_messages = [str(x.message) for x in w]

        assert any("observations" in m.lower() for m in warning_messages)
        assert result is not None

    def test_monthly_series_annualised(self):
        """Monthly series should annualise volatility by sqrt(12)."""
        rng = np.random.default_rng(42)
        idx = pd.period_range("2015-01", periods=120, freq="M")
        series = pd.Series(rng.normal(0.002, 0.008, 120), index=idx)

        model = ClaimsInflationGARCH(series, vol="GARCH", dist="normal", frequency="M")
        result = model.fit()

        # Annualised vol should be sqrt(12) * raw vol approximately
        raw_vol = result.arch_result.conditional_volatility.mean()
        ann_vol = result.conditional_volatility.mean()
        ratio = ann_vol / raw_vol
        # Should be close to sqrt(12) ≈ 3.46
        assert 2.0 < ratio < 5.0
