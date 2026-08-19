"""Elementary forwards, option payoffs, binomial prices, and Black–Scholes.

Contracts are European and the underlying pays no dividend unless a
dividend yield `q` is supplied to Black–Scholes. The binomial tree is
the Cox–Ross–Rubinstein parameterisation. Black–Scholes assumes
geometric Brownian motion, constant r and σ, continuous frictionless
trading, and no jumps.

Futures and forwards share the same terminal payoff diagram in this
frictionless, deterministic-rate setting. Daily mark-to-market of
futures is not modelled. Put–call parity is an identity among European
prices, the discount factor, and the spot; it is not a trading signal.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def forward_price(spot: float, rate: float, maturity: float, dividend_yield: float = 0.0) -> float:
    """F = S exp((r − q) T) for a continuous dividend yield q."""
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    return float(spot) * math.exp((float(rate) - float(dividend_yield)) * float(maturity))


def forward_payoff(spot_at_maturity: ArrayLike, delivery_price: float) -> np.ndarray:
    """Long forward payoff: S_T − K."""
    s = np.asarray(spot_at_maturity, dtype=float)
    return s - float(delivery_price)


def futures_payoff(spot_at_maturity: ArrayLike, futures_price: float) -> np.ndarray:
    """Terminal futures payoff under deterministic rates: S_T − F.

    Mark-to-market financing is omitted. The diagram coincides with the
    forward payoff when r is deterministic and there is no basis risk.
    """
    return forward_payoff(spot_at_maturity, futures_price)


def call_payoff(spot_at_maturity: ArrayLike, strike: float) -> np.ndarray:
    """max(S_T − K, 0)."""
    s = np.asarray(spot_at_maturity, dtype=float)
    return np.maximum(s - float(strike), 0.0)


def put_payoff(spot_at_maturity: ArrayLike, strike: float) -> np.ndarray:
    """max(K − S_T, 0)."""
    s = np.asarray(spot_at_maturity, dtype=float)
    return np.maximum(float(strike) - s, 0.0)


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    dividend_yield: float = 0.0,
) -> float:
    """C − P − (S e^{−qT} − K e^{−rT}). Zero when parity holds."""
    forwardish = float(spot) * math.exp(-float(dividend_yield) * float(maturity))
    discounted_strike = float(strike) * math.exp(-float(rate) * float(maturity))
    return float(call_price) - float(put_price) - (forwardish - discounted_strike)


def black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    *,
    option: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Black–Scholes–Merton European call or put."""
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    s = float(spot)
    k = float(strike)
    t = float(maturity)
    r = float(rate)
    q = float(dividend_yield)
    sigma = float(volatility)
    if t == 0.0 or sigma == 0.0:
        forward_s = s * math.exp((r - q) * t)
        intrinsic_call = math.exp(-r * t) * max(forward_s - k, 0.0)
        if option == "call":
            return float(intrinsic_call)
        intrinsic_put = math.exp(-r * t) * max(k - forward_s, 0.0)
        return float(intrinsic_put)
    root_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    df_q = math.exp(-q * t)
    df_r = math.exp(-r * t)
    if option == "call":
        return float(s * df_q * norm.cdf(d1) - k * df_r * norm.cdf(d2))
    return float(k * df_r * norm.cdf(-d2) - s * df_q * norm.cdf(-d1))


def binomial_european(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_steps: int,
    *,
    option: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Cox–Ross–Rubinstein European price on an n-step tree."""
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if maturity <= 0:
        return black_scholes(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            option=option,
            dividend_yield=dividend_yield,
        )
    dt = float(maturity) / float(n_steps)
    sigma = float(volatility)
    r = float(rate)
    q = float(dividend_yield)
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((r - q) * dt)
    risk_neutral = (growth - down) / (up - down)
    if not (0.0 < risk_neutral < 1.0):
        raise ValueError("risk-neutral probability is outside (0, 1); check inputs")
    discount = math.exp(-r * dt)
    j = np.arange(n_steps + 1, dtype=float)
    terminal = float(spot) * up ** (n_steps - j) * down ** j
    if option == "call":
        values = np.maximum(terminal - float(strike), 0.0)
    else:
        values = np.maximum(float(strike) - terminal, 0.0)
    p = risk_neutral
    for _ in range(n_steps):
        values = discount * (p * values[:-1] + (1.0 - p) * values[1:])
    return float(values[0])
