"""Time value of money and project-evaluation identities.

Cash-flow arrays are ordered from date t = 0. Rates are decimal periodic
rates commensurate with the spacing of the cash-flow index. A zero rate
is allowed; annuity formulae then collapse to a linear sum.

These maps are discounting identities. They do not encode credit risk,
reinvestment risk, or a stochastic term structure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, newton


def discount_factor(rate: float, periods: float) -> float:
    """Return (1 + rate) ** (-periods)."""
    if periods < 0:
        raise ValueError("periods must be non-negative")
    if rate <= -1.0:
        raise ValueError("rate must be greater than -1")
    return float((1.0 + rate) ** (-periods))


def present_value(future_amount: float, rate: float, periods: float) -> float:
    """Discount a single future amount to t = 0."""
    return float(future_amount) * discount_factor(rate, periods)


def future_value(present_amount: float, rate: float, periods: float) -> float:
    """Compound a single present amount to a future date."""
    if periods < 0:
        raise ValueError("periods must be non-negative")
    if rate <= -1.0:
        raise ValueError("rate must be greater than -1")
    return float(present_amount) * float((1.0 + rate) ** periods)


def annuity_present_value(
    payment: float,
    rate: float,
    n_periods: int,
    *,
    due: bool = False,
) -> float:
    """Closed-form present value of a level annuity.

    Ordinary annuity payments occur at t = 1, ..., n. An annuity-due
    (`due=True`) pays at t = 0, ..., n-1, which multiplies the ordinary
    value by (1 + rate).
    """
    if n_periods < 0:
        raise ValueError("n_periods must be non-negative")
    n = int(n_periods)
    pmt = float(payment)
    r = float(rate)
    if n == 0:
        return 0.0
    if abs(r) < 1e-15:
        value = pmt * n
    else:
        if r <= -1.0:
            raise ValueError("rate must be greater than -1")
        value = pmt * (1.0 - (1.0 + r) ** (-n)) / r
    if due:
        value *= 1.0 + r
    return float(value)


def annuity_future_value(
    payment: float,
    rate: float,
    n_periods: int,
    *,
    due: bool = False,
) -> float:
    """Closed-form future value of a level annuity at t = n."""
    if n_periods < 0:
        raise ValueError("n_periods must be non-negative")
    n = int(n_periods)
    pmt = float(payment)
    r = float(rate)
    if n == 0:
        return 0.0
    if abs(r) < 1e-15:
        value = pmt * n
    else:
        if r <= -1.0:
            raise ValueError("rate must be greater than -1")
        value = pmt * ((1.0 + r) ** n - 1.0) / r
    if due:
        value *= 1.0 + r
    return float(value)


def net_present_value(cashflows: ArrayLike, rate: float) -> float:
    """Sum discounted cash flows, with cashflows[0] undiscounted at t = 0."""
    cf = np.asarray(cashflows, dtype=float).reshape(-1)
    if cf.size == 0:
        raise ValueError("cashflows must be non-empty")
    if rate <= -1.0:
        raise ValueError("rate must be greater than -1")
    times = np.arange(cf.size, dtype=float)
    return float(np.sum(cf / (1.0 + rate) ** times))


def _npv_and_derivative(cashflows: np.ndarray, rate: float) -> tuple[float, float]:
    times = np.arange(cashflows.size, dtype=float)
    disc = (1.0 + rate) ** times
    npv = float(np.sum(cashflows / disc))
    d_npv = float(np.sum(-times * cashflows / ((1.0 + rate) ** (times + 1.0))))
    return npv, d_npv


def internal_rate_of_return(
    cashflows: ArrayLike,
    *,
    guess: float = 0.1,
) -> float:
    """Solve NPV(r) = 0 for r > -1.

    Newton iteration on the NPV polynomial is tried first. If it fails to
    converge, a bracketed Brent search is used on (-1 + epsilon, a large
    upper bound). Multiple real roots are possible when the cash-flow
    sign pattern is irregular; the procedure returns one economically
    conventional root, not a complete root set.
    """
    cf = np.asarray(cashflows, dtype=float).reshape(-1)
    if cf.size < 2:
        raise ValueError("cashflows must contain at least two dates")
    if not np.any(cf > 0) or not np.any(cf < 0):
        raise ValueError("IRR requires at least one positive and one negative cash flow")

    def objective(rate: float) -> float:
        return _npv_and_derivative(cf, rate)[0]

    def derivative(rate: float) -> float:
        return _npv_and_derivative(cf, rate)[1]

    try:
        root = float(newton(objective, guess, fprime=derivative, maxiter=80, tol=1e-12))
        if root > -1.0 and abs(objective(root)) < 1e-8:
            return root
    except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
        root = float("nan")

    lower = -1.0 + 1e-8
    upper = 10.0
    f_lo = objective(lower)
    f_hi = objective(upper)
    expand = 0
    while f_lo * f_hi > 0 and expand < 8:
        upper *= 2.0
        f_hi = objective(upper)
        expand += 1
    if f_lo * f_hi > 0:
        raise ValueError("could not bracket an IRR")
    return float(brentq(objective, lower, upper, xtol=1e-12))
