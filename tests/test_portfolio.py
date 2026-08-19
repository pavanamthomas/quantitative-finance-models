"""Mean-variance identities and GMV constraints."""

from __future__ import annotations

import numpy as np

from qfinmodels.portfolio import (
    efficient_frontier,
    expected_returns,
    global_minimum_variance,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    sample_covariance,
    weighted_average_volatility,
)


def _three_asset_moments():
    means = np.array([0.10, 0.06, 0.08])
    vols = np.array([0.20, 0.10, 0.15])
    corr = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.3],
            [0.4, 0.3, 1.0],
        ]
    )
    cov = np.outer(vols, vols) * corr
    return means, cov


def test_portfolio_weights_sum_to_one():
    _, cov = _three_asset_moments()
    w = global_minimum_variance(cov, long_only=False)
    assert abs(float(w.sum()) - 1.0) < 1e-10
    w_long = global_minimum_variance(cov, long_only=True)
    assert abs(float(w_long.sum()) - 1.0) < 1e-8


def test_gmv_weights_satisfy_constraints():
    _, cov = _three_asset_moments()
    w = global_minimum_variance(cov, long_only=False)
    ones = np.ones(cov.shape[0])
    # First-order condition: Σ w is proportional to 1.
    residual = cov @ w
    residual = residual / residual.mean() - ones
    assert np.max(np.abs(residual)) < 1e-8

    w_long = global_minimum_variance(cov, long_only=True)
    assert np.all(w_long >= -1e-10)
    assert np.all(w_long <= 1.0 + 1e-10)


def test_diversification_beats_weighted_average_vol():
    _, cov = _three_asset_moments()
    w = np.array([1.0, 1.0, 1.0]) / 3.0
    port_vol = portfolio_volatility(w, cov)
    avg_vol = weighted_average_volatility(w, cov)
    assert port_vol < avg_vol


def test_portfolio_variance_identity():
    means, cov = _three_asset_moments()
    w = np.array([0.2, 0.5, 0.3])
    assert abs(portfolio_variance(w, cov) - float(w @ cov @ w)) < 1e-15
    assert abs(portfolio_return(w, means) - float(w @ means)) < 1e-15


def test_sample_moments_and_frontier():
    rng = np.random.default_rng(0)
    data = rng.normal(0.001, 0.01, size=(200, 2))
    mu = expected_returns(data)
    cov = sample_covariance(data)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    vols, means, weights = efficient_frontier(mu, cov, n_points=12, long_only=True)
    assert vols.size == means.size
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(weights >= -1e-10)
