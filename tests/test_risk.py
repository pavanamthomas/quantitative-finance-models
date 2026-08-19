"""Historical VaR, parametric VaR, expected shortfall, and stress tables."""

from __future__ import annotations

import numpy as np

from qfinmodels.risk import (
    expected_shortfall,
    historical_var,
    parametric_var,
    scenario_pnl,
    stress_table,
)


def test_var_does_not_exceed_expected_shortfall():
    rng = np.random.default_rng(21)
    # Mixture: mostly Gaussian with an occasional left-tail shock so the
    # tail mean sits beyond the quantile.
    core = rng.normal(-0.001, 0.01, size=2000)
    tail = rng.normal(-0.08, 0.02, size=80)
    sample = np.concatenate([core, tail])
    alpha = 0.05
    var = historical_var(sample, alpha=alpha)
    es = expected_shortfall(sample, alpha=alpha)
    assert es >= var - 1e-12
    # ES is the more extreme loss number on this sample.
    assert es > var


def test_gaussian_parametric_var_matches_closed_form():
    sample = np.array([0.02, 0.01, 0.00, -0.01, -0.02, 0.015, -0.005], dtype=float)
    mu = float(np.mean(sample))
    sigma = float(np.std(sample, ddof=1))
    from scipy.stats import norm

    alpha = 0.05
    expected = -(mu + sigma * float(norm.ppf(alpha)))
    assert abs(parametric_var(sample, alpha=alpha) - expected) < 1e-12


def test_scenario_and_stress_tables():
    weights = np.array([0.4, 0.6])
    shock = np.array([-0.10, 0.02])
    pnl = scenario_pnl(weights, shock)
    assert abs(pnl - (0.4 * -0.10 + 0.6 * 0.02)) < 1e-15
    table = stress_table(weights, {"down": shock, "flat": np.zeros(2)})
    assert abs(table["down"] - pnl) < 1e-15
    assert abs(table["flat"]) < 1e-15


def test_var_does_not_determine_expected_shortfall():
    from qfinmodels.risk import two_samples_same_var_different_es

    out = two_samples_same_var_different_es(alpha=0.05, seed=21)
    assert out["es_heavy"] > out["es_mild"]
    assert out["es_mild"] >= out["var_mild"] - 1e-12
    assert out["es_heavy"] >= out["var_heavy"] - 1e-12
