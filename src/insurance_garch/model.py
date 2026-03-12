"""
model.py — GARCH model fitting and specification selection.

Design choices worth explaining:

1. Default vol='GJR-GARCH', not plain GARCH. Claims inflation reacts
   asymmetrically to shocks: a sudden spike in parts/labour costs hits harder
   than a corresponding fall helps. GJR-GARCH captures this leverage effect.

2. Default dist='studentst'. Claims inflation has fat tails — a single
   Ogden rate change or court ruling can create an extreme observation that
   a Normal distribution would assign essentially zero probability to. The
   Student-t distribution handles this properly.

3. GARCHSelector fits all vol × dist combinations and ranks by BIC. BIC
   penalises complexity more than AIC, which matters in small samples
   (40–120 quarterly observations is typical for UK personal lines).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from arch import arch_model

from insurance_garch.series import MIN_OBSERVATIONS


# Frequency multipliers for annualisation
FREQ_ANNUALISE = {"Q": 4, "M": 12, "A": 1, "D": 252}

# Valid specification options
VALID_VOL = ("GARCH", "GJR-GARCH", "EGARCH", "TARCH")
VALID_DIST = ("normal", "studentst", "skewstudent")
VALID_MEAN = ("AR", "Constant", "Zero")


@dataclass
class GARCHResult:
    """Fitted GARCH model result with insurance-relevant properties.

    Attributes
    ----------
    arch_result : Any  # ARCHModelResult from arch.univariate
        The underlying arch library result. Accessible for advanced users
        who want residual diagnostics, parameter t-stats, etc.
    vol_spec : str
        Volatility specification used ('GARCH', 'GJR-GARCH', etc.).
    distribution : str
        Error distribution ('normal', 'studentst', 'skewstudent').
    mean_spec : str
        Mean specification ('AR', 'Constant', 'Zero').
    frequency : str
        Observation frequency, used for annualisation.
    series : pd.Series
        The original series used for fitting.
    """

    arch_result: Any  # ARCHModelResult from arch.univariate
    vol_spec: str
    distribution: str
    mean_spec: str
    frequency: str
    series: pd.Series

    @property
    def conditional_volatility(self) -> pd.Series:
        """Annualised conditional volatility series.

        Returns
        -------
        pd.Series
            Conditional standard deviation, annualised by multiplying by
            sqrt(frequency multiplier). Same index as the fitted series.
        """
        annualise = FREQ_ANNUALISE.get(self.frequency, 1) ** 0.5
        cv = self.arch_result.conditional_volatility * annualise
        cv.name = "conditional_volatility_annualised"
        return cv

    @property
    def persistence(self) -> float:
        """GARCH persistence: sum of alpha + beta coefficients.

        For GJR-GARCH, accounts for the leverage term. Persistence >= 1
        implies an integrated (non-stationary) volatility process.

        Returns
        -------
        float
            Persistence value. Values close to 1.0 indicate long memory.
        """
        params = self.arch_result.params
        # Different specs use different parameter names
        # Try to compute from arch result's model directly
        try:
            # Use the variance model's persistence property if available
            model = self.arch_result.model
            if hasattr(model.volatility, 'compute_persistence'):
                return float(model.volatility.compute_persistence(params))
        except Exception:
            pass

        # Manual computation based on parameter names
        alpha = sum(v for k, v in params.items() if k.startswith("alpha["))
        beta = sum(v for k, v in params.items() if k.startswith("beta["))
        gamma = sum(v for k, v in params.items() if k.startswith("gamma["))

        # For GJR-GARCH, persistence = alpha + 0.5*gamma + beta
        # (assuming innovations are symmetric, E[z^2 * I(z<0)] = 0.5)
        if self.vol_spec in ("GJR-GARCH", "TARCH"):
            return float(alpha + 0.5 * gamma + beta)
        return float(alpha + beta)

    @property
    def half_life(self) -> float:
        """Volatility shock half-life in periods.

        Returns the number of periods for a volatility shock to decay to
        half its original magnitude, given the estimated persistence.

        Returns
        -------
        float
            Half-life in observation periods. Returns inf if persistence >= 1.
        """
        p = self.persistence
        if p >= 1.0:
            return float("inf")
        if p <= 0.0:
            return 0.0
        return float(np.log(2) / np.log(1 / p))

    def summary(self) -> pd.DataFrame:
        """Key parameter estimates as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: parameter, estimate, std_error, t_stat, p_value.
        """
        res = self.arch_result
        params = res.params
        stderr = res.std_err
        tvals = res.tvalues
        pvals = res.pvalues

        df = pd.DataFrame({
            "parameter": params.index,
            "estimate": params.values,
            "std_error": stderr.values,
            "t_stat": tvals.values,
            "p_value": pvals.values,
        })
        df["persistence"] = ""
        df.loc[df.index[-1], "persistence"] = f"{self.persistence:.4f}"
        df["half_life_periods"] = ""
        df.loc[df.index[-1], "half_life_periods"] = f"{self.half_life:.1f}"
        return df

    def plot_volatility(self, ax=None):
        """Plot conditional volatility over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        cv = self.conditional_volatility
        # Convert PeriodIndex to datetime for matplotlib compatibility
        x_vals = cv.index.to_timestamp() if hasattr(cv.index, "to_timestamp") else cv.index
        ax.plot(x_vals, cv.values, color="#1f77b4", linewidth=1.5,
                label="Conditional volatility (annualised)")
        ax.fill_between(x_vals, 0, cv.values, alpha=0.15, color="#1f77b4")
        ax.set_xlabel("Period")
        ax.set_ylabel("Annualised volatility")
        ax.set_title(
            f"{self.vol_spec} conditional volatility — {self.distribution} errors\n"
            f"Persistence: {self.persistence:.4f}, Half-life: {self.half_life:.1f} periods"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax


class ClaimsInflationGARCH:
    """GARCH model for claims inflation series.

    Parameters
    ----------
    series : pd.Series
        Log claims-rate series or log-change inflation series.
    vol : str, default 'GJR-GARCH'
        Volatility specification. One of 'GARCH', 'GJR-GARCH', 'EGARCH', 'TARCH'.
        GJR-GARCH is the default because inflation shocks are asymmetric —
        upward cost surprises tend to persist more than downward ones.
    p : int, default 1
        GARCH lag order.
    q : int, default 1
        ARCH lag order.
    dist : str, default 'studentst'
        Error distribution. One of 'normal', 'studentst', 'skewstudent'.
        studentst is the default because claims inflation has fat tails.
    mean : str, default 'Constant'
        Mean model. One of 'AR', 'Constant', 'Zero'.
    frequency : str, default 'Q'
        Observation frequency for annualisation.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> series = pd.Series(np.random.normal(0.03, 0.05, 80),
    ...                    index=pd.period_range('2000Q1', periods=80, freq='Q'))
    >>> model = ClaimsInflationGARCH(series)
    >>> result = model.fit()
    """

    def __init__(
        self,
        series: pd.Series,
        vol: str = "GJR-GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "studentst",
        mean: str = "Constant",
        frequency: str = "Q",
    ) -> None:
        if vol not in VALID_VOL:
            raise ValueError(f"vol must be one of {VALID_VOL}, got '{vol}'")
        if dist not in VALID_DIST:
            raise ValueError(f"dist must be one of {VALID_DIST}, got '{dist}'")
        if mean not in VALID_MEAN:
            raise ValueError(f"mean must be one of {VALID_MEAN}, got '{mean}'")
        if p < 1:
            raise ValueError("p must be >= 1")
        if q < 1:
            raise ValueError("q must be >= 1")

        n = len(series)
        if n < MIN_OBSERVATIONS:
            warnings.warn(
                f"Series has {n} observations (minimum recommended: {MIN_OBSERVATIONS}). "
                "GARCH estimates will be unreliable. Proceed with caution.",
                UserWarning,
                stacklevel=2,
            )

        self.series = series.copy()
        self.vol = vol
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.frequency = frequency

    def _build_arch_model(self):
        """Construct the arch model object."""
        # arch uses 'gjr-garch' lowercase; map our canonical names
        vol_map = {
            "GJR-GARCH": "GARCH",  # arch uses o parameter for leverage
            "GARCH": "GARCH",
            "EGARCH": "EGARCH",
            "TARCH": "GARCH",  # TARCH is GARCH with power=1
        }
        arch_vol = vol_map[self.vol]

        kwargs: dict = {
            "y": self.series,
            "mean": self.mean,
            "vol": arch_vol,
            "dist": self.dist,
        }

        if self.vol == "GJR-GARCH":
            kwargs["p"] = self.p
            kwargs["o"] = 1  # leverage term
            kwargs["q"] = self.q
        elif self.vol == "TARCH":
            kwargs["p"] = self.p
            kwargs["o"] = 1
            kwargs["q"] = self.q
            kwargs["power"] = 1.0
        elif self.vol == "EGARCH":
            kwargs["p"] = self.p
            kwargs["q"] = self.q
        else:
            kwargs["p"] = self.p
            kwargs["q"] = self.q

        return arch_model(**kwargs)

    def fit(
        self,
        disp: bool = False,
        show_warning: bool = True,
    ) -> GARCHResult:
        """Fit the GARCH model.

        Parameters
        ----------
        disp : bool, default False
            Whether to display optimisation output.
        show_warning : bool, default True
            Whether to surface convergence warnings.

        Returns
        -------
        GARCHResult
        """
        am = self._build_arch_model()

        fit_kwargs: dict = {
            "disp": disp,
            "show_warning": show_warning,
        }

        try:
            result = am.fit(**fit_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"GARCH fitting failed for {self.vol}({self.p},{self.q}) "
                f"with {self.dist} distribution. "
                f"Original error: {exc}"
            ) from exc

        return GARCHResult(
            arch_result=result,
            vol_spec=self.vol,
            distribution=self.dist,
            mean_spec=self.mean,
            frequency=self.frequency,
            series=self.series,
        )


class GARCHSelector:
    """Automatic GARCH specification selection by BIC.

    Fits all combinations of volatility specifications and error distributions,
    ranks by BIC (Bayesian Information Criterion), and returns the best-fitting
    model. BIC is preferred over AIC for insurance time series because it penalises
    additional parameters more heavily — important when sample sizes are modest.

    Parameters
    ----------
    series : pd.Series
        Log claims-rate or inflation series.
    vol_specs : tuple, default ('GARCH', 'GJR-GARCH', 'EGARCH', 'TARCH')
        Volatility specifications to compare.
    distributions : tuple, default ('normal', 'studentst', 'skewstudent')
        Error distributions to compare.
    p : int, default 1
        GARCH lag order (held constant across specifications).
    q : int, default 1
        ARCH lag order (held constant across specifications).
    mean : str, default 'Constant'
        Mean model.
    frequency : str, default 'Q'
        Observation frequency.

    Examples
    --------
    >>> selector = GARCHSelector(series)
    >>> rankings = selector.fit_all()
    >>> best_result = selector.best()
    """

    def __init__(
        self,
        series: pd.Series,
        vol_specs: tuple = ("GARCH", "GJR-GARCH", "EGARCH", "TARCH"),
        distributions: tuple = ("normal", "studentst", "skewstudent"),
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        frequency: str = "Q",
    ) -> None:
        self.series = series.copy()
        self.vol_specs = vol_specs
        self.distributions = distributions
        self.p = p
        self.q = q
        self.mean = mean
        self.frequency = frequency
        self._results: list[dict] = []
        self._fitted_models: dict[str, GARCHResult] = {}

    def fit_all(self) -> pd.DataFrame:
        """Fit all specification combinations and return ranked comparison table.

        Returns
        -------
        pd.DataFrame
            Columns: model_name, vol_spec, distribution, aic, bic,
            persistence, half_life, converged. Sorted by BIC ascending.
        """
        self._results = []
        self._fitted_models = {}

        combos = list(itertools.product(self.vol_specs, self.distributions))
        n_combos = len(combos)

        for i, (vol, dist) in enumerate(combos):
            model_name = f"{vol}({self.p},{self.q})/{dist}"
            try:
                model = ClaimsInflationGARCH(
                    series=self.series,
                    vol=vol,
                    p=self.p,
                    q=self.q,
                    dist=dist,
                    mean=self.mean,
                    frequency=self.frequency,
                )
                result = model.fit(disp=False, show_warning=False)
                arch_res = result.arch_result

                self._results.append({
                    "model_name": model_name,
                    "vol_spec": vol,
                    "distribution": dist,
                    "aic": float(arch_res.aic),
                    "bic": float(arch_res.bic),
                    "persistence": result.persistence,
                    "half_life": result.half_life,
                    "converged": arch_res.convergence_flag == 0,
                })
                self._fitted_models[model_name] = result

            except Exception as exc:
                warnings.warn(
                    f"Failed to fit {model_name}: {exc}. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                self._results.append({
                    "model_name": model_name,
                    "vol_spec": vol,
                    "distribution": dist,
                    "aic": np.nan,
                    "bic": np.nan,
                    "persistence": np.nan,
                    "half_life": np.nan,
                    "converged": False,
                })

        df = pd.DataFrame(self._results)
        df = df.sort_values("bic", na_position="last").reset_index(drop=True)
        df.index += 1  # 1-based rank
        df.index.name = "bic_rank"
        return df

    def best(self) -> GARCHResult:
        """Return the GARCHResult with the lowest BIC among converged models.

        Returns
        -------
        GARCHResult

        Raises
        ------
        RuntimeError
            If fit_all() has not been called or no models converged.
        """
        if not self._results:
            raise RuntimeError(
                "Call fit_all() before best(). No models have been fitted yet."
            )

        # Filter to converged models only
        converged = [r for r in self._results if r["converged"] and not np.isnan(r["bic"])]
        if not converged:
            raise RuntimeError(
                "No specifications converged. Check the series for near-constant "
                "variance or very short sample length."
            )

        best_row = min(converged, key=lambda x: x["bic"])
        return self._fitted_models[best_row["model_name"]]
