"""Mean-variance portfolio algebra on a finite set of assets.

Expected returns and covariances are treated as known primitives when
passed in, or as sample moments when estimated from a matrix of
simulated returns. The global-minimum-variance (GMV) portfolio minimises
w' Σ w subject to 1'w = 1. Short positions are allowed unless
`long_only=True`, in which case 0 ≤ w_i ≤ 1 is imposed numerically.

The long-only restriction is a constraint on the programme, not a
statement about mandates, transaction costs, or investability.
Diversification here means that portfolio standard deviation can lie
below the weighted average of component standard deviations when
correlations are below one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize


def expected_returns(returns: ArrayLike) -> np.ndarray:
    """Column-wise sample means of a T × N return matrix."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if r.ndim != 2 or r.shape[0] < 1:
        raise ValueError("returns must be a T×N array")
    return np.mean(r, axis=0)


def sample_covariance(returns: ArrayLike, ddof: int = 1) -> np.ndarray:
    """Sample covariance of a T × N return matrix."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError("covariance requires at least two observations")
    return np.cov(r, rowvar=False, ddof=ddof)


def _as_weights(weights: ArrayLike, n_assets: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n_assets:
        raise ValueError("weights must match the number of assets")
    return w


def portfolio_return(weights: ArrayLike, mean_returns: ArrayLike) -> float:
    """w' μ."""
    mu = np.asarray(mean_returns, dtype=float).reshape(-1)
    w = _as_weights(weights, mu.size)
    return float(w @ mu)


def portfolio_variance(weights: ArrayLike, covariance: ArrayLike) -> float:
    """w' Σ w."""
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    w = _as_weights(weights, cov.shape[0])
    return float(w @ cov @ w)


def portfolio_volatility(weights: ArrayLike, covariance: ArrayLike) -> float:
    """sqrt(w' Σ w)."""
    var = portfolio_variance(weights, covariance)
    if var < 0 and abs(var) < 1e-14:
        var = 0.0
    if var < 0:
        raise ValueError("portfolio variance is negative; check the covariance")
    return float(np.sqrt(var))


def weighted_average_volatility(weights: ArrayLike, covariance: ArrayLike) -> float:
    """sum |w_i| σ_i, a no-diversification benchmark when weights are long-only."""
    cov = np.asarray(covariance, dtype=float)
    w = _as_weights(weights, cov.shape[0])
    vols = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return float(np.sum(np.abs(w) * vols))


def global_minimum_variance(
    covariance: ArrayLike,
    *,
    long_only: bool = False,
) -> np.ndarray:
    """Weights of the global-minimum-variance portfolio.

    Unconstrained solution: Σ w = 1, then normalise so that 1'w = 1.
    Long-only solution: sequential least squares with bounds [0, 1] and
    a budget constraint. If the covariance is ill-conditioned the
    unconstrained weights can be extreme; that is a feature of the
    linear algebra, not a trading recommendation.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    n = cov.shape[0]
    if n == 0:
        raise ValueError("covariance must be non-empty")

    ones = np.ones(n)
    if not long_only:
        raw = np.linalg.solve(cov, ones)
        weights = raw / float(raw.sum())
        return weights

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    cons = {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}
    bounds = [(0.0, 1.0)] * n
    w0 = ones / n
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 200, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"long-only GMV optimiser did not converge: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("long-only GMV produced a zero-weight vector")
    return weights / total


def efficient_frontier(
    mean_returns: ArrayLike,
    covariance: ArrayLike,
    n_points: int = 25,
    *,
    long_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean–volatility coordinates for a two- or three-asset illustration.

    For two assets the frontier is parameterised by the first weight in
    [0, 1] if `long_only`, otherwise on a slightly wider interval. For
    three assets, weights on the simplex (or a relaxed simplex allowing
    modest shorts) are enumerated. Returns (volatility, mean, weights).
    """
    mu = np.asarray(mean_returns, dtype=float).reshape(-1)
    cov = np.asarray(covariance, dtype=float)
    n = mu.size
    if n not in (2, 3):
        raise ValueError("efficient_frontier illustration supports two or three assets")
    if cov.shape != (n, n):
        raise ValueError("covariance shape must match mean_returns")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")

    records: list[tuple[float, float, np.ndarray]] = []
    if n == 2:
        grid = np.linspace(0.0, 1.0, n_points) if long_only else np.linspace(-0.2, 1.2, n_points)
        for w0 in grid:
            w = np.array([w0, 1.0 - w0], dtype=float)
            records.append((portfolio_volatility(w, cov), portfolio_return(w, mu), w))
    else:
        k = max(int(np.ceil(np.sqrt(n_points))), 3)
        axis = np.linspace(0.0, 1.0, k) if long_only else np.linspace(-0.15, 1.15, k)
        for a in axis:
            for b in axis:
                c = 1.0 - a - b
                if long_only and (c < -1e-12 or c > 1.0 + 1e-12):
                    continue
                w = np.array([a, b, max(c, 0.0) if long_only else c], dtype=float)
                if long_only:
                    if w.sum() <= 0:
                        continue
                    w = w / w.sum()
                records.append((portfolio_volatility(w, cov), portfolio_return(w, mu), w))

    vols = np.array([item[0] for item in records], dtype=float)
    means = np.array([item[1] for item in records], dtype=float)
    weights = np.vstack([item[2] for item in records])
    order = np.argsort(vols)
    return vols[order], means[order], weights[order]
