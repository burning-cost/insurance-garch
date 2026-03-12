"""
insurance-garch: GARCH volatility models for UK personal lines claims inflation.

The PRA Dear Chief Actuary 2023 letter explicitly flagged inflation volatility as
a key risk. This library wraps Kevin Sheppard's arch package with an insurance-specific
workflow: exposure-weighted inflation series, automatic specification selection,
scenario generation for pricing review committees, and VaR-style backtesting.

Quickstart
----------
>>> from insurance_garch.series import ExposureWeightedSeries
>>> from insurance_garch.model import GARCHSelector
>>> from insurance_garch.forecast import VolatilityScenarioGenerator
>>> from insurance_garch.backtest import GARCHBacktest
"""

from insurance_garch.series import ExposureWeightedSeries, CalendarYearInflationSeries
from insurance_garch.model import ClaimsInflationGARCH, GARCHSelector, GARCHResult
from insurance_garch.forecast import VolatilityScenarioGenerator, ScenarioSet
from insurance_garch.backtest import GARCHBacktest, BacktestResult
from insurance_garch.report import GARCHReport

__version__ = "0.1.0"
__all__ = [
    "ExposureWeightedSeries",
    "CalendarYearInflationSeries",
    "ClaimsInflationGARCH",
    "GARCHSelector",
    "GARCHResult",
    "VolatilityScenarioGenerator",
    "ScenarioSet",
    "GARCHBacktest",
    "BacktestResult",
    "GARCHReport",
]
