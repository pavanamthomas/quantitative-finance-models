"""Realized volatility, rolling windows, and a reproducible GARCH(1,1)."""

from __future__ import annotations

import numpy as np

from qfinmodels.volatility import (
    fit_garch11,
    realized_volatility,
    rolling_volatility,
    simulate_garch11,
)


def test_realized_volatility_known_path():
    returns = np.array([0.01, -0.02, 0.015, -0.005], dtype=float)
    t = returns.size
    independent = float(np.sqrt(np.sum(returns ** 2) * (252.0 / t)))
    assert abs(realized_volatility(returns) - independent) < 1e-12


def test_rolling_volatility_length_and_scale():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, size=120)
    window = 20
    roll = rolling_volatility(r, window=window, periods_per_year=252.0)
    assert roll.shape == r.shape
    assert np.all(np.isnan(roll[: window - 1]))
    assert np.all(np.isfinite(roll[window - 1 :]))
    assert np.all(roll[window - 1 :] > 0)


def test_garch11_fit_on_simulated_path():
    omega, alpha, beta = 0.00003, 0.10, 0.85
    returns, variance = simulate_garch11(2500, omega, alpha, beta, seed=5)
    assert np.all(variance > 0)
    fitted = fit_garch11(returns)
    assert fitted.omega > 0
    assert fitted.alpha >= 0
    assert fitted.beta >= 0
    assert fitted.persistence < 1.0
    # Loose recovery: simulated identification, not a market calibration.
    assert abs(fitted.alpha - alpha) < 0.08
    assert abs(fitted.beta - beta) < 0.10
    assert np.all(fitted.variance > 0)
