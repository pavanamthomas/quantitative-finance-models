"""Bond pricing, yield, duration, and convexity checks."""

from __future__ import annotations

from qfinmodels.fixed_income import (
    bond_convexity,
    bond_price,
    bond_yield,
    duration_convexity_price,
    macaulay_duration,
    modified_duration,
)


def test_bond_price_falls_when_yield_rises():
    face, coupon, years, freq = 100.0, 0.05, 10.0, 1
    low = bond_price(face, coupon, years, 0.04, frequency=freq)
    mid = bond_price(face, coupon, years, 0.06, frequency=freq)
    high = bond_price(face, coupon, years, 0.08, frequency=freq)
    assert low > mid > high


def test_bond_price_matches_independent_sum():
    face, coupon_pmt, y, n = 100.0, 5.0, 0.06, 10
    independent = sum(coupon_pmt / (1.0 + y) ** t for t in range(1, n + 1)) + face / (1.0 + y) ** n
    assert abs(independent - 92.63991294858526) < 1e-10
    priced = bond_price(100.0, 0.05, 10.0, 0.06, frequency=1)
    assert abs(priced - independent) < 1e-10


def test_bond_yield_recovers_input():
    price = bond_price(100.0, 0.05, 10.0, 0.06, frequency=2)
    recovered = bond_yield(price, 100.0, 0.05, 10.0, frequency=2)
    assert abs(recovered - 0.06) < 1e-8


def test_par_bond_prices_at_face():
    assert abs(bond_price(100.0, 0.07, 8.0, 0.07, frequency=1) - 100.0) < 1e-9


def test_duration_convexity_error_shrinks_for_small_shocks():
    face, coupon, years, ytm, freq = 100.0, 0.05, 10.0, 0.06, 1
    price = bond_price(face, coupon, years, ytm, frequency=freq)
    mod = modified_duration(face, coupon, years, ytm, frequency=freq)
    conv = bond_convexity(face, coupon, years, ytm, frequency=freq)
    mac = macaulay_duration(face, coupon, years, ytm, frequency=freq)
    assert mac > 0
    assert abs(mod - mac / (1.0 + ytm / freq)) < 1e-12

    small = 0.001
    large = 0.025
    err_small = abs(
        duration_convexity_price(price, mod, conv, small)
        - bond_price(face, coupon, years, ytm + small, frequency=freq)
    )
    err_large = abs(
        duration_convexity_price(price, mod, conv, large)
        - bond_price(face, coupon, years, ytm + large, frequency=freq)
    )
    assert err_small < err_large
    # The second-order map should be closer than a few cents on a 10bp move
    # of a 100-face bond.
    assert err_small < 0.01
