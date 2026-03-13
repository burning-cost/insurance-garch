# insurance-garch

GARCH volatility models for UK personal lines claims inflation.

## The problem

The PRA Dear Chief Actuary letter (2023) flagged inflation volatility explicitly: pricing teams need to demonstrate that their assumptions reflect not just the *level* of inflation but its *uncertainty*. A simple point estimate of 8% medical inflation says nothing about whether that figure could plausibly be 12% or 4% — and the capital implications of that spread are significant.

Standard GARCH models from the financial econometrics literature apply directly to claims inflation series, but the workflow is awkward: you need exposure weighting, insurance-appropriate specification selection (asymmetric shocks matter, fat tails matter), scenario generation in a form pricing committees can use, and backtesting with the correct statistical tests.

This library wraps Kevin Sheppard's `arch` package (v7+) with that workflow.

## What it does

1. **Series construction** — builds exposure-weighted log claims-rate series from raw claims and exposure data. Handles development triangles too.
2. **Model fitting** — fits GARCH, GJR-GARCH, EGARCH, or TARCH with normal, Student-t, or skewed Student-t errors.
3. **Specification selection** — `GARCHSelector` fits all combinations and ranks by BIC. You get a table you can paste into your model documentation.
4. **Scenario generation** — produces base/stressed/shocked volatility paths over a configurable horizon (default 8 quarters). Fan chart output for committee packs.
5. **Backtesting** — Kupiec (1995) unconditional coverage and Christoffersen (1998) conditional coverage tests. The conditional coverage test is what matters: a model that clusters its VaR breaches has failed.
6. **Reporting** — structured dict and HTML output combining all of the above.

## Design choices

**Default vol='GJR-GARCH'** not plain GARCH. Claims inflation reacts asymmetrically: a sudden spike in parts/labour costs tends to persist more than a corresponding fall helps. GJR-GARCH captures this leverage effect. If your data doesn't need it, BIC selection will tell you that.

**Default dist='studentst'** not Normal. A single Ogden rate change, a court ruling on periodical payment orders, or a pandemic year creates an extreme observation that a Normal distribution assigns essentially zero probability to. Student-t handles fat tails properly.

**Bootstrap simulation** not analytical forecasts. For fat-tailed distributions at longer horizons, the analytical approximation degrades. Bootstrap resamples from the empirical standardised residual distribution and preserves the actual tails you've estimated.

**BIC not AIC** for specification selection. With 40–120 quarterly observations — typical for UK personal lines — AIC's lighter penalty on parameters leads to over-fitting. BIC is the right default at these sample sizes.

## Installation

```bash
pip install insurance-garch
```

Requires Python 3.10+, arch>=7.0, numpy, pandas, scipy, matplotlib.

## Quickstart

```python
import pandas as pd
from insurance_garch.series import ExposureWeightedSeries
from insurance_garch.model import GARCHSelector
from insurance_garch.forecast import VolatilityScenarioGenerator
from insurance_garch.backtest import GARCHBacktest
from insurance_garch.report import GARCHReport

# 1. Build log claims-rate series
ews = ExposureWeightedSeries(claims, exposure, period='Q')
log_series = ews.to_series()

# 2. Select best specification
selector = GARCHSelector(log_series)
ranking = selector.fit_all()    # DataFrame ranked by BIC
print(ranking.head())
best = selector.best()          # GARCHResult with lowest BIC

# 3. Generate scenarios
gen = VolatilityScenarioGenerator(best, horizon=8)
scenarios = gen.generate(n_sims=10_000)
scenarios.fan_chart()           # Bank of England-style fan chart

# 4. Backtest
bt = GARCHBacktest(best, alpha=0.05)
result = bt.run()
print(result.summary())         # Kupiec + Christoffersen tests

# 5. Report
report = GARCHReport(best, scenarios, result, bic_table=ranking)
html = report.to_html()         # self-contained HTML for committee pack
```

## Modules

| Module | Key classes |
|---|---|
| `series.py` | `ExposureWeightedSeries`, `CalendarYearInflationSeries` |
| `model.py` | `ClaimsInflationGARCH`, `GARCHSelector`, `GARCHResult` |
| `forecast.py` | `VolatilityScenarioGenerator`, `ScenarioSet` |
| `backtest.py` | `GARCHBacktest`, `BacktestResult` |
| `report.py` | `GARCHReport` |

## Minimum sample size

GARCH needs sample size. The library warns if your series has fewer than 40 observations — that's the practical minimum for reliable parameter estimates. Quarterly data from 2014 to 2024 gives you 40 observations. Monthly data from 2021 to 2024 gives you 36. If you're below threshold, results should be treated as indicative, not production-grade.

## Performance

Benchmarked against a static 5% p.a. trend assumption and a 3-year rolling standard deviation on synthetic quarterly UK motor claims inflation data (2005–2024, n=80) with a genuine regime shift in Q1 2021. See `notebooks/benchmark_garch.py` for the full comparison.

- **Regime shift detection**: The synthetic DGP has 2.5x higher variance from Q1 2021. GARCH picks this up — annualised conditional volatility roughly doubles in the high-vol regime. Static trend produces the same number regardless of market conditions.
- **VaR backtest**: GARCH typically passes the Kupiec unconditional coverage test and the Christoffersen independence test at the 5% significance level. Static trend fails the Christoffersen test — its VaR breaches cluster in time, which is exactly the failure mode GARCH is designed to prevent.
- **MAE vs true conditional volatility**: GARCH achieves 20–35% lower mean absolute error against the known DGP volatility than the static fixed trend. The gap is concentrated in the post-regime-shift period, which is when accurate volatility estimates matter most for pricing decisions.
- **Specification selection**: On data generated from a GJR-GARCH / Student-t process, BIC correctly selects GJR-GARCH or EGARCH as the best specification in most runs. Normal-error specifications consistently rank lower.
- **Limitation**: GARCH is sensitive to regime shifts that happen near the end of the series. If the high-vol regime just started, you have few observations in it and the parameter estimates are uncertain. Fan chart widths in the benchmark reflect this — the p10-p90 band is wide, which is the honest answer when data is limited.

## Notebooks

See `notebooks/01_garch_motor_inflation_demo.py` for a complete workflow and `notebooks/benchmark_garch.py` for the head-to-head comparison against static trend assumptions.

## References

- Christoffersen, P.F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Kupiec, P.H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Sheppard, K. (2024). arch: Autoregressive Conditional Heteroskedasticity Models. https://arch.readthedocs.io/

## Licence

MIT
