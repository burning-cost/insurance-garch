"""Tests for insurance_garch.forecast module."""

import numpy as np
import pandas as pd
import pytest

from insurance_garch.forecast import VolatilityScenarioGenerator, ScenarioSet


class TestScenarioSet:
    def _make_paths(self, n_sims=1000, horizon=8):
        rng = np.random.default_rng(42)
        return np.abs(rng.normal(0.05, 0.02, size=(n_sims, horizon)))

    def test_construction(self):
        paths = self._make_paths()
        ss = ScenarioSet(paths, horizon=8, frequency="Q")
        assert isinstance(ss.base, pd.Series)
        assert isinstance(ss.stressed, pd.Series)
        assert isinstance(ss.shocked, pd.Series)
        assert len(ss.base) == 8

    def test_ordering_base_lt_stressed_lt_shocked(self):
        paths = self._make_paths(n_sims=5000)
        ss = ScenarioSet(paths, horizon=8, frequency="Q")
        # Median < p90 < p99 at every horizon step
        assert (ss.base.values <= ss.stressed.values).all()
        assert (ss.stressed.values <= ss.shocked.values).all()

    def test_to_dataframe_shape(self):
        paths = self._make_paths()
        ss = ScenarioSet(paths, horizon=8, frequency="Q")
        df = ss.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8
        assert "base_median" in df.columns
        assert "stressed_p90" in df.columns
        assert "shocked_p99" in df.columns
        assert "period" in df.columns

    def test_invalid_paths_ndim_raises(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            ScenarioSet(np.zeros(8), horizon=8, frequency="Q")

    def test_fan_chart_renders(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        paths = self._make_paths()
        ss = ScenarioSet(paths, horizon=8, frequency="Q")
        ax = ss.fan_chart()
        assert ax is not None
        plt.close("all")

    def test_fan_chart_accepts_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        paths = self._make_paths()
        ss = ScenarioSet(paths, horizon=8, frequency="Q")
        fig, ax = plt.subplots()
        returned = ss.fan_chart(ax=ax)
        assert returned is ax
        plt.close("all")

    def test_fan_chart_custom_title(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        paths = self._make_paths()
        ss = ScenarioSet(paths, horizon=8, frequency="Q")
        ax = ss.fan_chart(title="Motor inflation stress test")
        assert ax.get_title() == "Motor inflation stress test"
        plt.close("all")


class TestVolatilityScenarioGenerator:
    def test_generate_returns_scenario_set(self, fitted_result):
        gen = VolatilityScenarioGenerator(fitted_result, horizon=4)
        ss = gen.generate(n_sims=200, seed=42)
        assert isinstance(ss, ScenarioSet)

    def test_horizon_respected(self, fitted_result):
        gen = VolatilityScenarioGenerator(fitted_result, horizon=6)
        ss = gen.generate(n_sims=200, seed=42)
        assert ss.horizon == 6
        assert len(ss.base) == 6

    def test_n_sims_respected(self, fitted_result):
        gen = VolatilityScenarioGenerator(fitted_result, horizon=4)
        ss = gen.generate(n_sims=500, seed=42)
        assert ss.paths.shape[0] == 500

    def test_simulation_method(self, fitted_result):
        gen = VolatilityScenarioGenerator(fitted_result, horizon=4)
        ss = gen.generate(n_sims=200, method="simulation", seed=42)
        assert isinstance(ss, ScenarioSet)

    def test_invalid_method_raises(self, fitted_result):
        gen = VolatilityScenarioGenerator(fitted_result, horizon=4)
        with pytest.raises(ValueError, match="method must be"):
            gen.generate(method="monte_carlo")

    def test_invalid_horizon_raises(self, fitted_result):
        with pytest.raises(ValueError, match="horizon must be"):
            VolatilityScenarioGenerator(fitted_result, horizon=0)

    def test_volatility_is_positive(self, fitted_result):
        gen = VolatilityScenarioGenerator(fitted_result, horizon=4)
        ss = gen.generate(n_sims=200, seed=42)
        assert (ss.paths >= 0).all()

    def test_seed_generates_finite_values(self, fitted_result):
        # Bootstrap simulation uses arch internal RNG; seed controls numpy only.
        # Test that we get finite, positive values in the right shape.
        gen = VolatilityScenarioGenerator(fitted_result, horizon=4)
        ss1 = gen.generate(n_sims=100, seed=99)
        assert ss1.paths.shape == (100, 4)
        assert np.all(np.isfinite(ss1.paths))
        assert np.all(ss1.paths >= 0)
