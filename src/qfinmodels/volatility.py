"""Realized volatility, rolling volatility, and a Gaussian GARCH(1,1).

Realized and rolling estimators are sample second-moment maps. The
GARCH(1,1) recursion is

    σ_t² = ω + α r_{t-1}² + β σ_{t-1}²

with ω > 0, α ≥ 0, β ≥ 0, and α + β < 1 for covariance stationarity.
Parameters are estimated by Gaussian quasi-maximum likelihood using
`scipy.optimize`. The Gaussian likelihood is a computational device; it
does not assert that returns are Gaussian.

A fitted GARCH path on simulated data is a numerical illustration. It is
not a forecast product and it is not estimated on market series in this
repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize


def _as_returns(returns: ArrayLike) -> np.ndarray:
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("returns must contain at least two finite observations")
    return r


def realized_volatility(
    returns: ArrayLike,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """Annualised realized volatility: sqrt(sum r_t²) * sqrt(periods_per_year / T)."""
    r = _as_returns(returns)
    t = float(r.size)
    return float(np.sqrt(np.sum(r * r) * (float(periods_per_year) / t)))


def rolling_volatility(
    returns: ArrayLike,
    window: int,
    *,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """Annualised rolling standard deviation with a trailing window."""
    if window < 2:
        raise ValueError("window must be at least 2")
    r = _as_returns(returns)
    if r.size < window:
        raise ValueError("returns shorter than the rolling window")
    scale = np.sqrt(float(periods_per_year))
    out = np.full(r.size, np.nan, dtype=float)
    for t in range(window - 1, r.size):
        chunk = r[t - window + 1 : t + 1]
        out[t] = float(np.std(chunk, ddof=1) * scale)
    return out


@dataclass(frozen=True)
class Garch11Result:
    """Gaussian QML estimates and the filtered variance path."""

    omega: float
    alpha: float
    beta: float
    variance: np.ndarray
    loglik: float
    n_obs: int

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def simulate_garch11(
    n_obs: int,
    omega: float,
    alpha: float,
    beta: float,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Gaussian GARCH(1,1) returns and the latent variance path."""
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("GARCH parameters must satisfy ω > 0, α ≥ 0, β ≥ 0")
    if alpha + beta >= 1.0:
        raise ValueError("α + β must be strictly less than 1")
    rng = np.random.default_rng(seed)
    variance = np.empty(n_obs, dtype=float)
    returns = np.empty(n_obs, dtype=float)
    variance[0] = omega / (1.0 - alpha - beta)
    returns[0] = rng.normal(0.0, np.sqrt(variance[0]))
    for t in range(1, n_obs):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = rng.normal(0.0, np.sqrt(variance[t]))
    return returns, variance


def _garch_variance(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    var = np.empty(returns.size, dtype=float)
    var[0] = float(np.var(returns))
    if var[0] <= 0:
        var[0] = omega / max(1.0 - alpha - beta, 1e-8)
    for t in range(1, returns.size):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
    return var


def _neg_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = (float(params[0]), float(params[1]), float(params[2]))
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12
    var = _garch_variance(returns, omega, alpha, beta)
    if np.any(var <= 1e-16) or np.any(~np.isfinite(var)):
        return 1e12
    ll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(var) + returns ** 2 / var)
    if not np.isfinite(ll):
        return 1e12
    return float(ll)


def fit_garch11(
    returns: ArrayLike,
    *,
    x0: tuple[float, float, float] | None = None,
) -> Garch11Result:
    """Estimate Gaussian GARCH(1,1) by quasi-maximum likelihood."""
    r = _as_returns(returns)
    sample_var = float(np.var(r))
    if x0 is None:
        start = np.array([sample_var * 0.05, 0.05, 0.80], dtype=float)
    else:
        start = np.asarray(x0, dtype=float)
    bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999)]
    result = minimize(
        _neg_loglik,
        start,
        args=(r,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-10},
    )
    omega, alpha, beta = (float(result.x[0]), float(result.x[1]), float(result.x[2]))
    if alpha + beta >= 1.0:
        scale = 0.99 / (alpha + beta)
        alpha *= scale
        beta *= scale
    variance = _garch_variance(r, omega, alpha, beta)
    loglik = -float(_neg_loglik(np.array([omega, alpha, beta]), r))
    return Garch11Result(
        omega=omega,
        alpha=alpha,
        beta=beta,
        variance=variance,
        loglik=loglik,
        n_obs=int(r.size),
    )
