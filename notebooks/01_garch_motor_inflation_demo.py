# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-garch: Motor Claims Inflation Volatility
# MAGIC
# MAGIC This notebook demonstrates the full `insurance-garch` workflow on synthetic
# MAGIC UK motor insurance data calibrated to the 2021–2023 inflation episode.
# MAGIC
# MAGIC **Audience:** Pricing actuaries preparing committee packs.
# MAGIC **What you get:** A fitted GARCH model, specification comparison table,
# MAGIC fan chart for stressed scenarios, and Kupiec/Christoffersen backtest output.

# COMMAND ----------

# MAGIC %pip install insurance-garch arch matplotlib scipy

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic data
# MAGIC
# MAGIC We generate quarterly motor claims inflation data calibrated to the 2005–2024
# MAGIC period. The DGP embeds:
# MAGIC - Low volatility regime pre-2020 (GFC period excluded for simplicity)
# MAGIC - High volatility 2021–2023 (supply chain + used car price inflation)
# MAGIC - Mean reversion towards ~3% annualised inflation

# COMMAND ----------

def make_motor_inflation_series(seed: int = 42) -> tuple:
    """
    Synthetic UK motor claims inflation: quarterly, 2005 Q1 – 2024 Q4.
    Returns (claims, exposure, log_rate) as pd.Series with PeriodIndex.
    """
    rng = np.random.default_rng(seed)
    n = 80  # Q1 2005 to Q4 2024
    idx = pd.period_range("2005Q1", periods=n, freq="Q")

    # Exposure: vehicle years, grows ~1% per quarter
    exposure_base = 500_000
    exposure = pd.Series(
        exposure_base * (1.01 ** np.arange(n)) * (1 + rng.normal(0, 0.01, n)),
        index=idx,
        name="vehicle_years",
    )

    # Claims inflation DGP: GJR-GARCH with regime shift in 2021
    omega = 0.00002
    alpha = 0.08
    beta = 0.82
    gamma = 0.05   # leverage: upward inflation shocks persist more
    mu = 0.008     # ~3.2% annualised mean inflation

    # Regime shift: volatility doubles from Q1 2021
    inflation_start = 64  # ~Q1 2021

    eps = rng.standard_normal(n)
    inflation = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)

    for t in range(1, n):
        regime_mult = 2.5 if t >= inflation_start else 1.0
        sigma2[t] = (
            omega * regime_mult
            + (alpha + gamma * (inflation[t-1] < 0)) * inflation[t-1]**2
            + beta * sigma2[t-1]
        )
        inflation[t] = mu + np.sqrt(sigma2[t]) * eps[t]

    # Convert log-inflation to claims: claims_t = exposure_t * base_rate * exp(cumsum(inflation))
    base_rate = 0.04  # 4 claims per 100 vehicle-years
    cumulative_inflation = np.exp(np.cumsum(inflation))
    rate = base_rate * cumulative_inflation
    claims_raw = exposure.values * rate * (1 + rng.normal(0, 0.02, n))
    claims = pd.Series(np.maximum(claims_raw, 1.0), index=idx, name="claims_paid_gbp")

    return claims, exposure


claims, exposure = make_motor_inflation_series()
print(f"Claims series: {len(claims)} quarterly observations (2005 Q1 – 2024 Q4)")
print(f"Exposure range: {exposure.min():,.0f} – {exposure.max():,.0f} vehicle-years")
print(f"Claims range: £{claims.min():,.0f} – £{claims.max():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the log claims-rate series

# COMMAND ----------

from insurance_garch.series import ExposureWeightedSeries

ews = ExposureWeightedSeries(claims, exposure, period="Q")
log_series = ews.to_series()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(log_series.index.to_timestamp(), log_series.values,
             color="#2171b5", linewidth=1.2)
axes[0].set_title("Log claims rate (quarterly)", fontweight="bold")
axes[0].set_ylabel("log(claims / exposure)")
axes[0].axvline(pd.Timestamp("2021-01-01"), color="#d7301f", linestyle="--",
                alpha=0.7, label="High-inflation regime start")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# First differences as proxy for inflation
diff_series = log_series.diff().dropna()
axes[1].bar(diff_series.index.to_timestamp(), diff_series.values,
            color=["#d7301f" if v > 0 else "#2171b5" for v in diff_series.values],
            alpha=0.7, width=80)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Quarter-on-quarter log change (inflation proxy)", fontweight="bold")
axes[1].set_ylabel("Δ log(claims rate)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Specification selection via BIC
# MAGIC
# MAGIC We fit all combinations of vol spec × error distribution.
# MAGIC GJR-GARCH with Student-t errors is expected to win on BIC given our DGP.

# COMMAND ----------

from insurance_garch.model import GARCHSelector

selector = GARCHSelector(log_series, frequency="Q")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ranking = selector.fit_all()

print("\nModel comparison (ranked by BIC):")
print(ranking[["model_name", "aic", "bic", "persistence", "half_life", "converged"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect the best-fitting model

# COMMAND ----------

best = selector.best()
print(f"\nSelected: {best.vol_spec} with {best.distribution} errors")
print(f"Persistence: {best.persistence:.4f}")
print(f"Volatility half-life: {best.half_life:.1f} quarters")
print(f"\nParameter estimates:")
print(best.summary().to_string(index=False))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 4))
best.plot_volatility(ax=ax)
ax.axvline(
    log_series.index[64].to_timestamp(),
    color="#d7301f", linestyle="--", alpha=0.7, label="High-inflation regime"
)
ax.legend()
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate volatility scenarios (8-quarter horizon)
# MAGIC
# MAGIC This is the output that goes into the pricing committee pack.
# MAGIC The "stressed" scenario (p90) and "shocked" scenario (p99) inform
# MAGIC sensitivity loadings and capital buffer discussions.

# COMMAND ----------

from insurance_garch.forecast import VolatilityScenarioGenerator

gen = VolatilityScenarioGenerator(best, horizon=8)
scenarios = gen.generate(n_sims=10_000, seed=42)

print("\nScenario table (annualised volatility):")
df_scenarios = scenarios.to_dataframe()
df_scenarios_pct = df_scenarios.copy()
for col in df_scenarios.columns:
    if col != "period":
        df_scenarios_pct[col] = df_scenarios_pct[col].map(lambda x: f"{x:.1%}")
print(df_scenarios_pct.to_string(index=False))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 5))
scenarios.fan_chart(
    ax=ax,
    title="Motor claims inflation volatility — 8-quarter forward scenarios"
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Backtest: Kupiec + Christoffersen

# COMMAND ----------

from insurance_garch.backtest import GARCHBacktest

bt = GARCHBacktest(best, alpha=0.05)
bt_result = bt.run()

print(f"\nObserved exceedance rate: {bt_result.exceedance_rate:.1%} (nominal: 5%)")
print("\nBacktest summary:")
print(bt_result.summary().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Full report

# COMMAND ----------

from insurance_garch.report import GARCHReport

report = GARCHReport(
    best,
    scenarios,
    backtest_result=bt_result,
    bic_table=ranking,
    title="Motor Claims Inflation Volatility — Q4 2024 Pricing Review",
)

d = report.to_dict()
print(f"\nReport summary:")
print(f"  Model: {d['model_spec']['vol_spec']} / {d['model_spec']['distribution']}")
print(f"  Current volatility (annualised): {d['conditional_volatility']['current']:.1%}")
print(f"  Persistence: {d['model_spec']['persistence']}")
print(f"  Backtest passes all tests: {d['backtest']['passes_all_tests']}")

# Save HTML report
html = report.to_html()
with open("/tmp/motor_garch_report.html", "w") as f:
    f.write(html)
print("\nHTML report saved to /tmp/motor_garch_report.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key findings for committee
# MAGIC
# MAGIC | Metric | Value | Interpretation |
# MAGIC |--------|-------|----------------|
# MAGIC | Best model | GJR-GARCH / Student-t | Asymmetric + fat-tailed, as expected |
# MAGIC | Current volatility | ~ from table above | Elevated vs pre-2020 baseline |
# MAGIC | Persistence | ~ from table above | Shocks dissipate slowly |
# MAGIC | Half-life | ~ from table above | Quarters to decay to 50% |
# MAGIC | Stressed p90 | ~ from scenarios | Inform sensitivity loading |
# MAGIC | Shocked p99 | ~ from scenarios | Inform capital buffer |
# MAGIC | Kupiec test | ~ from backtest | Unconditional coverage |
# MAGIC | Christoffersen | ~ from backtest | Clustering of breaches |
# MAGIC
# MAGIC **Recommendation:** Use stressed (p90) scenario as the base pricing assumption
# MAGIC when current conditional volatility exceeds the 10-year historical mean.
# MAGIC Apply shocked (p99) scenario for pricing adequacy stress tests.
