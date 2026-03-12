"""Tests for insurance_garch.report module."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from insurance_garch.report import GARCHReport
from insurance_garch.forecast import VolatilityScenarioGenerator
from insurance_garch.backtest import GARCHBacktest


@pytest.fixture
def scenario_set(fitted_result):
    gen = VolatilityScenarioGenerator(fitted_result, horizon=4)
    return gen.generate(n_sims=200, seed=42)


@pytest.fixture
def backtest_result(fitted_result):
    bt = GARCHBacktest(fitted_result, alpha=0.05)
    return bt.run()


@pytest.fixture
def bic_table(long_series):
    from insurance_garch.model import GARCHSelector
    selector = GARCHSelector(
        long_series,
        vol_specs=("GARCH", "GJR-GARCH"),
        distributions=("normal",),
    )
    return selector.fit_all()


class TestGARCHReport:
    def test_to_dict_basic(self, fitted_result, scenario_set):
        report = GARCHReport(fitted_result, scenario_set)
        d = report.to_dict()
        assert "title" in d
        assert "model_spec" in d
        assert "scenarios" in d
        assert "parameter_summary" in d
        assert "conditional_volatility" in d

    def test_to_dict_model_spec_fields(self, fitted_result, scenario_set):
        report = GARCHReport(fitted_result, scenario_set)
        d = report.to_dict()
        spec = d["model_spec"]
        assert "vol_spec" in spec
        assert "distribution" in spec
        assert "persistence" in spec
        assert "half_life_periods" in spec
        assert "n_observations" in spec

    def test_to_dict_with_backtest(self, fitted_result, scenario_set, backtest_result):
        report = GARCHReport(fitted_result, scenario_set, backtest_result=backtest_result)
        d = report.to_dict()
        assert "backtest" in d
        assert "kupiec_pvalue" in d["backtest"]
        assert "conditional_coverage_pvalue" in d["backtest"]
        assert "passes_all_tests" in d["backtest"]

    def test_to_dict_no_backtest(self, fitted_result, scenario_set):
        report = GARCHReport(fitted_result, scenario_set, backtest_result=None)
        d = report.to_dict()
        assert "backtest" not in d

    def test_to_dict_with_bic_table(self, fitted_result, scenario_set, bic_table):
        report = GARCHReport(fitted_result, scenario_set, bic_table=bic_table)
        d = report.to_dict()
        assert "bic_ranking" in d
        assert len(d["bic_ranking"]) <= 5

    def test_to_html_returns_string(self, fitted_result, scenario_set):
        report = GARCHReport(fitted_result, scenario_set)
        html = report.to_html()
        assert isinstance(html, str)
        assert "<html>" in html
        assert "</html>" in html

    def test_to_html_contains_model_name(self, fitted_result, scenario_set):
        report = GARCHReport(fitted_result, scenario_set)
        html = report.to_html()
        assert fitted_result.vol_spec in html

    def test_to_html_contains_title(self, fitted_result, scenario_set):
        title = "Q4 2025 Motor Inflation Review"
        report = GARCHReport(fitted_result, scenario_set, title=title)
        html = report.to_html()
        assert title in html

    def test_to_html_with_full_data(
        self, fitted_result, scenario_set, backtest_result, bic_table
    ):
        """Full report with all optional components should render without error."""
        report = GARCHReport(
            fitted_result,
            scenario_set,
            backtest_result=backtest_result,
            bic_table=bic_table,
            title="Full Integration Test Report",
        )
        html = report.to_html()
        assert "Full Integration Test Report" in html
        assert "Backtest" in html
        assert "BIC" in html

    def test_scenarios_table_in_dict(self, fitted_result, scenario_set):
        report = GARCHReport(fitted_result, scenario_set)
        d = report.to_dict()
        table = d["scenarios"]["table"]
        assert isinstance(table, list)
        assert len(table) == scenario_set.horizon

    def test_cv_stats_in_dict(self, fitted_result, scenario_set):
        report = GARCHReport(fitted_result, scenario_set)
        d = report.to_dict()
        cv = d["conditional_volatility"]
        assert cv["mean"] > 0
        assert cv["current"] > 0
        assert cv["max"] >= cv["mean"]
