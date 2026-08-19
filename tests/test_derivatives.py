"""Payoffs, put-call parity, binomial convergence, and Black-Scholes numbers."""

from __future__ import annotations

import math

from scipy.special import erf

from qfinmodels.derivatives import (
    binomial_european,
    black_scholes,
    call_payoff,
    forward_payoff,
    forward_price,
    futures_payoff,
    put_call_parity_gap,
    put_payoff,
)


def _norm_cdf(x: float) -> float:
    """Independent standard-normal CDF via the error function."""
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def test_payoffs_at_known_spots():
    assert abs(float(call_payoff(120.0, 100.0)) - 20.0) < 1e-12
    assert abs(float(call_payoff(80.0, 100.0)) - 0.0) < 1e-12
    assert abs(float(put_payoff(80.0, 100.0)) - 20.0) < 1e-12
    assert abs(float(forward_payoff(110.0, 100.0)) - 10.0) < 1e-12
    assert abs(float(futures_payoff(90.0, 100.0)) + 10.0) < 1e-12


def test_forward_price_closed_form():
    s, r, t = 100.0, 0.05, 1.0
    assert abs(forward_price(s, r, t) - s * math.exp(r * t)) < 1e-12


def test_black_scholes_matches_independent_cdf():
    s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    disc = math.exp(-r * t)
    call = s * _norm_cdf(d1) - k * disc * _norm_cdf(d2)
    put = k * disc * _norm_cdf(-d2) - s * _norm_cdf(-d1)
    assert abs(d1 - 0.35) < 1e-12
    assert abs(d2 - 0.15) < 1e-12
    assert abs(call - 10.450583572185565) < 5e-10
    assert abs(put - 5.573526022256971) < 5e-10
    assert abs(black_scholes(s, k, t, r, sigma, option="call") - call) < 1e-9
    assert abs(black_scholes(s, k, t, r, sigma, option="put") - put) < 1e-9


def test_put_call_parity_holds_within_tolerance():
    s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    call = black_scholes(s, k, t, r, sigma, option="call")
    put = black_scholes(s, k, t, r, sigma, option="put")
    gap = put_call_parity_gap(call, put, s, k, r, t)
    assert abs(gap) < 1e-10
    disc = math.exp(-r * t)
    assert abs((call - put) - (s - k * disc)) < 1e-10


def test_binomial_european_call_converges_toward_black_scholes():
    s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = black_scholes(s, k, t, r, sigma, option="call")
    coarse = binomial_european(s, k, t, r, sigma, n_steps=20, option="call")
    fine = binomial_european(s, k, t, r, sigma, n_steps=200, option="call")
    assert abs(fine - bs) < abs(coarse - bs)
    assert abs(fine - bs) < 0.05
