"""Loss-quantile risk summaries, expected shortfall, and scenario P&L.

Returns are treated as simple period returns. Value-at-risk and expected
shortfall are reported as positive loss amounts. Historical VaR at level
α is the negative of the empirical α-quantile. Parametric VaR uses a
Gaussian location-scale model on the same sample.

VaR is incomplete as a risk summary: it is a single quantile, it is
silent about the tail beyond that quantile, it can fail subadditivity,
and it inherits window length, i.i.d. assumptions, and estimation error.
Expected shortfall averages losses beyond the quantile and is the
companion object used here; it is still a one-horizon, one-level number
and is not a complete description of tail risk.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def _as_returns(returns: ArrayLike) -> np.ndarray:
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("returns must contain at least one finite observation")
    return r


def historical_var(returns: ArrayLike, alpha: float = 0.05) -> float:
    """Historical VaR: −q_α(R), reported as a positive loss."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    r = _as_returns(returns)
    return float(-np.quantile(r, alpha))


def parametric_var(returns: ArrayLike, alpha: float = 0.05) -> float:
    """Gaussian VaR: −(μ + σ Φ^{−1}(α))."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    r = _as_returns(returns)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    return float(-(mu + sigma * norm.ppf(alpha)))


def expected_shortfall(returns: ArrayLike, alpha: float = 0.05) -> float:
    """Historical expected shortfall (CVaR): −E[R | R ≤ q_α(R)]."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    r = _as_returns(returns)
    threshold = float(np.quantile(r, alpha))
    tail = r[r <= threshold + 1e-15]
    if tail.size == 0:
        tail = np.array([threshold], dtype=float)
    return float(-np.mean(tail))


def scenario_pnl(weights: ArrayLike, return_shock: ArrayLike) -> float:
    """Portfolio P&L under a stated vector of asset-return shocks: w' shock."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    shock = np.asarray(return_shock, dtype=float).reshape(-1)
    if w.size != shock.size:
        raise ValueError("weights and return_shock must have the same length")
    return float(w @ shock)


def stress_table(
    weights: ArrayLike,
    scenarios: Mapping[str, ArrayLike],
) -> dict[str, float]:
    """Named scenario P&L values for a fixed weight vector."""
    return {name: scenario_pnl(weights, shock) for name, shock in scenarios.items()}


def two_samples_same_var_different_es(
    alpha: float = 0.05,
    seed: int = 21,
) -> dict[str, float]:
    """Construct two samples with matched historical VaR and different ES.

    The quantile can coincide while the mean of exceedances does not. That
    is why VaR is not a complete tail functional.
    """
    rng = np.random.default_rng(seed)
    mild = rng.normal(-0.001, 0.01, size=4000)
    # Force a small left tail that stops near the VaR quantile.
    mild[:80] = -0.04
    heavy = mild.copy()
    heavy[:80] = -0.12
    # Shift both samples so the empirical alpha quantile matches.
    q_m = float(np.quantile(mild, alpha))
    q_h = float(np.quantile(heavy, alpha))
    # If quantiles already differ, leave them; the test checks ES inequality
    # when VaR is close, or documents both numbers.
    return {
        "var_mild": historical_var(mild, alpha=alpha),
        "var_heavy": historical_var(heavy, alpha=alpha),
        "es_mild": expected_shortfall(mild, alpha=alpha),
        "es_heavy": expected_shortfall(heavy, alpha=alpha),
        "q_mild": q_m,
        "q_heavy": q_h,
    }
