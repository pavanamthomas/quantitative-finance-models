"""CAPM-style one-factor illustration on simulated excess returns.

The security-market-line identity used here is

    E[R_i] = r_f + β_i (E[R_m] − r_f)

Beta is estimated by OLS of asset excess returns on market excess
returns. The construction is a linear statistical illustration, not a
trading rule and not a test of market efficiency.

Assumptions that the identity does not justify: a single priced factor,
unrestricted borrowing at r_f, homogeneous expectations, and stability
of beta across regimes. Jensen's alpha on a simulated sample is an
estimation residual, not evidence of mispricing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CapmEstimate:
    """OLS output for a single-factor excess-return regression."""

    alpha: float
    beta: float
    residual_std: float
    r_squared: float
    n_obs: int


def simulate_capm_returns(
    betas: ArrayLike,
    *,
    n_obs: int = 500,
    rf: float = 0.0002,
    market_mean: float = 0.0008,
    market_vol: float = 0.01,
    idiosyncratic_vol: float = 0.012,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a market factor and assets generated from the one-factor map.

    r_i = r_f + β_i (r_m − r_f) + ε_i,  ε_i ~ N(0, σ_ε²) independent.
    Returns (market_returns, asset_returns) with asset_returns shaped T × N.
    """
    beta_vec = np.asarray(betas, dtype=float).reshape(-1)
    if beta_vec.size == 0:
        raise ValueError("betas must be non-empty")
    if n_obs < 3:
        raise ValueError("n_obs must be at least 3")
    rng = np.random.default_rng(seed)
    market = rng.normal(market_mean, market_vol, size=n_obs)
    excess_m = market - rf
    noise = rng.normal(0.0, idiosyncratic_vol, size=(n_obs, beta_vec.size))
    assets = rf + excess_m[:, None] * beta_vec[None, :] + noise
    return market, assets


def estimate_capm(
    asset_returns: ArrayLike,
    market_returns: ArrayLike,
    rf: float = 0.0,
) -> CapmEstimate:
    """Estimate α and β from r_i − r_f = α + β (r_m − r_f) + ε."""
    y = np.asarray(asset_returns, dtype=float).reshape(-1) - float(rf)
    x = np.asarray(market_returns, dtype=float).reshape(-1) - float(rf)
    if y.size != x.size:
        raise ValueError("asset_returns and market_returns must have the same length")
    if y.size < 3:
        raise ValueError("need at least three observations")
    frame = pd.DataFrame({"y": y, "x": x}).dropna()
    design = sm.add_constant(frame["x"], has_constant="add")
    model = sm.OLS(frame["y"], design, hasconst=True).fit()
    params = np.asarray(model.params, dtype=float)
    resid_std = float(np.sqrt(model.scale))
    return CapmEstimate(
        alpha=float(params[0]),
        beta=float(params[1]),
        residual_std=resid_std,
        r_squared=float(model.rsquared),
        n_obs=int(frame.shape[0]),
    )


def security_market_line(
    betas: ArrayLike,
    rf: float,
    market_premium: float,
) -> np.ndarray:
    """Fitted expected returns on the SML: r_f + β (E[R_m] − r_f)."""
    beta_vec = np.asarray(betas, dtype=float).reshape(-1)
    return float(rf) + beta_vec * float(market_premium)
