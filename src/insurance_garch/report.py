"""
report.py — Structured reporting for GARCH results.

The output from this module goes into pricing review committee packs.
We produce both HTML (for Jupyter notebooks and dashboards) and dict
(for JSON APIs or further processing). Both formats contain the same content:
model specification, parameter table, BIC ranking if available, conditional
volatility plot, fan chart, and backtest summary.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from insurance_garch.model import GARCHResult
from insurance_garch.forecast import ScenarioSet
from insurance_garch.backtest import BacktestResult


class GARCHReport:
    """Structured report combining model, scenarios, and backtest results.

    Parameters
    ----------
    garch_result : GARCHResult
        Fitted GARCH model result.
    scenario_set : ScenarioSet
        Generated volatility scenarios.
    backtest_result : BacktestResult, optional
        Backtest results. If None, backtest section is omitted.
    bic_table : pd.DataFrame, optional
        BIC ranking table from GARCHSelector.fit_all(). If provided,
        the report includes a model comparison section.
    title : str, optional
        Report title. Defaults to 'GARCH Volatility Report'.

    Examples
    --------
    >>> report = GARCHReport(result, scenarios, backtest, bic_table=rankings)
    >>> html = report.to_html()
    >>> data = report.to_dict()
    """

    def __init__(
        self,
        garch_result: GARCHResult,
        scenario_set: ScenarioSet,
        backtest_result: Optional[BacktestResult] = None,
        bic_table: Optional[pd.DataFrame] = None,
        title: str = "GARCH Volatility Report",
    ) -> None:
        self.garch_result = garch_result
        self.scenario_set = scenario_set
        self.backtest_result = backtest_result
        self.bic_table = bic_table
        self.title = title

    def to_dict(self) -> dict:
        """Return report contents as a structured dictionary.

        Returns
        -------
        dict
            Keys: title, model_spec, parameter_summary, scenarios,
            backtest (if available), bic_ranking (if available).
        """
        result = self.garch_result
        scenarios = self.scenario_set

        d: dict = {
            "title": self.title,
            "model_spec": {
                "vol_spec": result.vol_spec,
                "distribution": result.distribution,
                "mean_spec": result.mean_spec,
                "frequency": result.frequency,
                "n_observations": len(result.series),
                "persistence": round(result.persistence, 4),
                "half_life_periods": round(result.half_life, 1),
            },
            "parameter_summary": result.summary().to_dict(orient="records"),
            "conditional_volatility": {
                "mean": round(float(result.conditional_volatility.mean()), 4),
                "std": round(float(result.conditional_volatility.std()), 4),
                "min": round(float(result.conditional_volatility.min()), 4),
                "max": round(float(result.conditional_volatility.max()), 4),
                "current": round(float(result.conditional_volatility.iloc[-1]), 4),
            },
            "scenarios": {
                "horizon": scenarios.horizon,
                "frequency": scenarios.frequency,
                "n_simulations": scenarios.paths.shape[0],
                "table": scenarios.to_dataframe().round(4).to_dict(orient="records"),
            },
        }

        if self.backtest_result is not None:
            bt = self.backtest_result
            d["backtest"] = {
                "alpha": bt.alpha,
                "exceedance_rate": round(bt.exceedance_rate, 4),
                "kupiec_stat": round(bt.kupiec_stat, 4),
                "kupiec_pvalue": round(bt.kupiec_pvalue, 4),
                "christoffersen_stat": round(bt.christoffersen_stat, 4),
                "christoffersen_pvalue": round(bt.christoffersen_pvalue, 4),
                "conditional_coverage_stat": round(bt.conditional_coverage_stat, 4),
                "conditional_coverage_pvalue": round(bt.conditional_coverage_pvalue, 4),
                "passes_all_tests": (
                    bt.kupiec_pvalue >= 0.05
                    and bt.christoffersen_pvalue >= 0.05
                    and bt.conditional_coverage_pvalue >= 0.05
                ),
            }

        if self.bic_table is not None:
            d["bic_ranking"] = (
                self.bic_table.head(5)
                .round(4)
                .to_dict(orient="records")
            )

        return d

    def to_html(self) -> str:
        """Return a self-contained HTML report string.

        The report includes inline base64-encoded matplotlib figures so it
        can be saved as a single .html file or rendered in a Jupyter notebook.

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        d = self.to_dict()

        # Generate figures
        vol_img = self._encode_figure(self._make_vol_figure())
        fan_img = self._encode_figure(self._make_fan_figure())

        spec = d["model_spec"]
        cv = d["conditional_volatility"]

        html_parts = [
            f"<html><head><meta charset='utf-8'>",
            f"<style>",
            "body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #333; }",
            "h1 { color: #08519c; border-bottom: 2px solid #08519c; padding-bottom: 8px; }",
            "h2 { color: #2171b5; margin-top: 32px; }",
            "table { border-collapse: collapse; width: 100%; margin: 12px 0; }",
            "th { background: #2171b5; color: white; padding: 8px 12px; text-align: left; }",
            "td { padding: 6px 12px; border-bottom: 1px solid #ddd; }",
            "tr:nth-child(even) td { background: #f5f8fc; }",
            ".pass { color: #1a7f37; font-weight: bold; }",
            ".fail { color: #d7301f; font-weight: bold; }",
            ".metric { display: inline-block; background: #f0f4fa; border: 1px solid #c6d5e8; "
            "border-radius: 6px; padding: 12px 20px; margin: 6px; text-align: center; }",
            ".metric .value { font-size: 24px; font-weight: bold; color: #08519c; }",
            ".metric .label { font-size: 12px; color: #666; margin-top: 4px; }",
            "img { max-width: 100%; margin: 16px 0; }",
            "</style></head><body>",
            f"<h1>{self.title}</h1>",
            "<h2>Model Specification</h2>",
            f"<div class='metric'><div class='value'>{spec['vol_spec']}</div>"
            f"<div class='label'>Volatility model</div></div>",
            f"<div class='metric'><div class='value'>{spec['distribution']}</div>"
            f"<div class='label'>Error distribution</div></div>",
            f"<div class='metric'><div class='value'>{spec['persistence']}</div>"
            f"<div class='label'>Persistence</div></div>",
            f"<div class='metric'><div class='value'>{spec['half_life_periods']}</div>"
            f"<div class='label'>Half-life (periods)</div></div>",
            f"<div class='metric'><div class='value'>{spec['n_observations']}</div>"
            f"<div class='label'>Observations</div></div>",
            "<h2>Conditional Volatility</h2>",
            f"<div class='metric'><div class='value'>{cv['current']:.1%}</div>"
            f"<div class='label'>Current (annualised)</div></div>",
            f"<div class='metric'><div class='value'>{cv['mean']:.1%}</div>"
            f"<div class='label'>Historical mean</div></div>",
            f"<div class='metric'><div class='value'>{cv['max']:.1%}</div>"
            f"<div class='label'>Historical peak</div></div>",
            f"<img src='data:image/png;base64,{vol_img}' alt='Conditional volatility chart'>",
        ]

        # Parameter table
        html_parts.append("<h2>Parameter Estimates</h2>")
        html_parts.append(self._params_to_html(d["parameter_summary"]))

        # BIC ranking
        if "bic_ranking" in d:
            html_parts.append("<h2>Model Comparison (top 5 by BIC)</h2>")
            html_parts.append(self._bic_to_html(d["bic_ranking"]))

        # Fan chart
        html_parts.append(
            f"<h2>Volatility Scenarios (horizon: {d['scenarios']['horizon']} "
            f"{d['scenarios']['frequency']} periods, "
            f"n={d['scenarios']['n_simulations']:,})</h2>"
        )
        html_parts.append(
            f"<img src='data:image/png;base64,{fan_img}' alt='Scenario fan chart'>"
        )

        # Scenario table
        html_parts.append("<h2>Scenario Summary</h2>")
        scenario_df = pd.DataFrame(d["scenarios"]["table"])
        html_parts.append(scenario_df.to_html(index=False, border=0, classes=""))

        # Backtest
        if "backtest" in d:
            bt = d["backtest"]
            overall = "pass" if bt["passes_all_tests"] else "fail"
            html_parts.append(
                f"<h2>Backtest Results "
                f"<span class='{overall}'>({'PASS' if bt['passes_all_tests'] else 'FAIL'})</span>"
                f"</h2>"
            )
            html_parts.append(
                self.backtest_result.summary().to_html(index=False, border=0, classes="")
            )

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    def _make_vol_figure(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        self.garch_result.plot_volatility(ax=ax)
        return fig

    def _make_fan_figure(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        self.scenario_set.fan_chart(ax=ax)
        return fig

    @staticmethod
    def _encode_figure(fig) -> str:
        """Encode matplotlib figure as base64 PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def _params_to_html(params: list[dict]) -> str:
        """Render parameter list as HTML table."""
        if not params:
            return "<p>No parameters.</p>"
        headers = ["Parameter", "Estimate", "Std Error", "t-stat", "p-value"]
        rows = []
        for p in params:
            pval = p.get("p_value", "")
            try:
                pval_f = float(pval)
                sig = " *" if pval_f < 0.05 else ""
                pval_str = f"{pval_f:.4f}{sig}"
            except (ValueError, TypeError):
                pval_str = str(pval)

            rows.append(
                f"<tr><td>{p.get('parameter', '')}</td>"
                f"<td>{float(p.get('estimate', 0)):.6f}</td>"
                f"<td>{float(p.get('std_error', 0)):.6f}</td>"
                f"<td>{float(p.get('t_stat', 0)):.3f}</td>"
                f"<td>{pval_str}</td></tr>"
            )
        header_html = "".join(f"<th>{h}</th>" for h in headers)
        return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    @staticmethod
    def _bic_to_html(bic_rows: list[dict]) -> str:
        """Render BIC ranking as HTML table."""
        if not bic_rows:
            return "<p>No BIC ranking available.</p>"
        df = pd.DataFrame(bic_rows)
        return df.to_html(index=False, border=0, classes="")
