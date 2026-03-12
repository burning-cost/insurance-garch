"""
Shared test fixtures for insurance-garch tests.

All synthetic data is generated with fixed seeds for reproducibility.
We use a GARCH(1,1) DGP with known parameters to test that the estimator
recovers approximately correct parameter values.
"""

import numpy as np
import pandas as pd
import pytest


RNG = np.random.default_rng(42)


def make_garch_series(
    n: int = 120,
    omega: float = 0.0001,
    alpha: float = 0.10,
    beta: float = 0.80,
    mu: float = 0.005,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic GARCH(1,1) series with known parameters."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        eps = rng.standard_normal()
        returns[t] = mu + np.sqrt(sigma2[t - 1]) * eps
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    idx = pd.period_range("2000Q1", periods=n, freq="Q")
    return pd.Series(returns, index=idx, name="log_claims_rate")


def make_exposure_series(n: int = 120, seed: int = 42) -> tuple:
    """Generate synthetic claims and exposure series."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000Q1", periods=n, freq="Q")
    exposure = pd.Series(
        1000 + rng.integers(-100, 100, size=n).astype(float),
        index=idx,
        name="exposure",
    )
    # Claims rate ~ 0.05 with some noise
    rate = 0.05 * np.exp(rng.normal(0, 0.05, size=n))
    claims = pd.Series(exposure.values * rate, index=idx, name="claims")
    return claims, exposure


@pytest.fixture
def long_series():
    """120-observation GARCH(1,1) series with known DGP."""
    return make_garch_series(n=120)


@pytest.fixture
def short_series():
    """25-observation series — deliberately below minimum threshold."""
    return make_garch_series(n=25)


@pytest.fixture
def claims_exposure():
    """Tuple of (claims, exposure) pd.Series."""
    return make_exposure_series(n=80)


@pytest.fixture
def fitted_result(long_series):
    """A fitted GARCHResult for use in downstream tests."""
    from insurance_garch.model import ClaimsInflationGARCH
    model = ClaimsInflationGARCH(long_series, vol="GARCH", dist="normal", frequency="Q")
    return model.fit()
