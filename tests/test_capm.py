"""CAPM regression and security-market-line checks on simulated returns."""

from __future__ import annotations

import numpy as np

from qfinmodels.capm import estimate_capm, security_market_line, simulate_capm_returns


def test_sml_identity():
    betas = np.array([0.5, 1.0, 1.5])
    rf = 0.02
    premium = 0.06
    expected = rf + betas * premium
    assert np.allclose(security_market_line(betas, rf, premium), expected)


def test_estimated_beta_recovers_simulated_factor():
    true_beta = np.array([1.25])
    rf = 0.0001
    market, assets = simulate_capm_returns(
        true_beta,
        n_obs=4000,
        rf=rf,
        market_vol=0.01,
        idiosyncratic_vol=0.004,
        seed=11,
    )
    fit = estimate_capm(assets[:, 0], market, rf=rf)
    assert abs(fit.beta - 1.25) < 0.05
    assert fit.n_obs == 4000
    assert 0.0 <= fit.r_squared <= 1.0
