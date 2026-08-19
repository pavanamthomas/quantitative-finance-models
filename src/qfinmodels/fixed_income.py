"""Coupon-bond valuation, yield, duration, and convexity.

Cash flows are a level coupon plus redemption of face at maturity. The
yield is a constant per-period discount rate (bond-equivalent, divided
by payment frequency). Default, optionality, and non-parallel curve
moves are outside the model.

Duration and convexity are local expansions of the present-value map in
the yield. They are compared with a full reprice in `duration_convexity_price`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _periods_and_coupon(
    face: float,
    coupon_rate: float,
    years: float,
    frequency: int,
) -> tuple[int, float]:
    if years <= 0:
        raise ValueError("years must be positive")
    if frequency not in (1, 2, 4, 12):
        raise ValueError("frequency must be 1, 2, 4, or 12")
    if face <= 0:
        raise ValueError("face must be positive")
    n_periods = int(round(years * frequency))
    if n_periods <= 0:
        raise ValueError("years and frequency must yield at least one period")
    coupon = float(face) * float(coupon_rate) / float(frequency)
    return n_periods, coupon


def bond_cashflows(
    face: float,
    coupon_rate: float,
    years: float,
    frequency: int = 1,
) -> np.ndarray:
    """Return the coupon-and-principal schedule from t = 1 to maturity."""
    n_periods, coupon = _periods_and_coupon(face, coupon_rate, years, frequency)
    flows = np.full(n_periods, coupon, dtype=float)
    flows[-1] += float(face)
    return flows


def bond_price(
    face: float,
    coupon_rate: float,
    years: float,
    yield_rate: float,
    frequency: int = 1,
) -> float:
    """Price a coupon bond by discounting scheduled cash flows at a flat yield."""
    flows = bond_cashflows(face, coupon_rate, years, frequency)
    y = float(yield_rate) / float(frequency)
    if y <= -1.0:
        raise ValueError("periodic yield must be greater than -1")
    times = np.arange(1, flows.size + 1, dtype=float)
    return float(np.sum(flows / (1.0 + y) ** times))


def bond_yield(
    price: float,
    face: float,
    coupon_rate: float,
    years: float,
    frequency: int = 1,
) -> float:
    """Solve for the annualised yield that reprices the bond."""
    if price <= 0:
        raise ValueError("price must be positive")
    flows = bond_cashflows(face, coupon_rate, years, frequency)

    def objective(annual_yield: float) -> float:
        y = annual_yield / float(frequency)
        times = np.arange(1, flows.size + 1, dtype=float)
        return float(np.sum(flows / (1.0 + y) ** times) - price)

    return float(brentq(objective, -0.9, 5.0, xtol=1e-12))


def macaulay_duration(
    face: float,
    coupon_rate: float,
    years: float,
    yield_rate: float,
    frequency: int = 1,
) -> float:
    """Macaulay duration in years: present-value-weighted average receipt time."""
    flows = bond_cashflows(face, coupon_rate, years, frequency)
    y = float(yield_rate) / float(frequency)
    times = np.arange(1, flows.size + 1, dtype=float)
    pv_flows = flows / (1.0 + y) ** times
    price = float(np.sum(pv_flows))
    if price <= 0:
        raise ValueError("bond price must be positive to define duration")
    duration_periods = float(np.sum(times * pv_flows) / price)
    return duration_periods / float(frequency)


def modified_duration(
    face: float,
    coupon_rate: float,
    years: float,
    yield_rate: float,
    frequency: int = 1,
) -> float:
    """Modified duration in years: -dP/P / dy for a parallel yield shift."""
    mac = macaulay_duration(face, coupon_rate, years, yield_rate, frequency)
    y = float(yield_rate) / float(frequency)
    return mac / (1.0 + y)


def bond_convexity(
    face: float,
    coupon_rate: float,
    years: float,
    yield_rate: float,
    frequency: int = 1,
) -> float:
    """Convexity in years squared, consistent with a yield shock in annual units.

    C = (1 / P) * sum_t t(t + 1) CF_t / (1 + y_p)^{t+2} / frequency^2
    where y_p is the periodic yield. Then
    dP / P ≈ −D_mod dy + 0.5 C (dy)^2
    for a change dy in the annual yield.
    """
    flows = bond_cashflows(face, coupon_rate, years, frequency)
    freq = float(frequency)
    y = float(yield_rate) / freq
    times = np.arange(1, flows.size + 1, dtype=float)
    pv_flows = flows / (1.0 + y) ** times
    price = float(np.sum(pv_flows))
    if price <= 0:
        raise ValueError("bond price must be positive to define convexity")
    weighted = np.sum(times * (times + 1.0) * flows / (1.0 + y) ** (times + 2.0))
    return float(weighted / price / (freq ** 2))


def duration_convexity_price(
    price: float,
    modified_dur: float,
    convexity: float,
    yield_shock: float,
) -> float:
    """First-plus-second-order yield expansion of a bond price."""
    dy = float(yield_shock)
    return float(price) * (
        1.0 - float(modified_dur) * dy + 0.5 * float(convexity) * dy * dy
    )


def price_after_yield_shift(
    face: float,
    coupon_rate: float,
    years: float,
    yield_rate: float,
    yield_shock: float,
    *,
    frequency: int = 1,
    method: str = "reprice",
) -> float:
    """Map a parallel yield shock to a price.

    ``reprice`` discounts cash flows at the new yield. ``duration_convexity``
    uses the local expansion. For large shocks the reprice is the correction.
    """
    if method == "reprice":
        return bond_price(
            face, coupon_rate, years, yield_rate + yield_shock, frequency=frequency
        )
    if method == "duration_convexity":
        price0 = bond_price(face, coupon_rate, years, yield_rate, frequency=frequency)
        mod = modified_duration(
            face, coupon_rate, years, yield_rate, frequency=frequency
        )
        conv = bond_convexity(face, coupon_rate, years, yield_rate, frequency=frequency)
        return duration_convexity_price(price0, mod, conv, yield_shock)
    raise ValueError("method must be 'reprice' or 'duration_convexity'")
