"""Closed-form and independently computed checks for time-value identities."""

from __future__ import annotations

import math

from qfinmodels.tvm import (
    annuity_future_value,
    annuity_present_value,
    discount_factor,
    future_value,
    internal_rate_of_return,
    net_present_value,
    present_value,
)


def test_present_and_future_value_invert():
    amount = 250.0
    rate = 0.04
    n = 7
    fv = future_value(amount, rate, n)
    pv = present_value(fv, rate, n)
    assert abs(pv - amount) < 1e-10
    assert abs(discount_factor(rate, n) - (1.0 + rate) ** (-n)) < 1e-15


def test_annuity_present_value_matches_closed_form():
    payment = 100.0
    rate = 0.05
    n = 10
    closed = payment * (1.0 - (1.0 + rate) ** (-n)) / rate
    # Independent evaluation of the same geometric sum used as the textbook identity.
    assert abs(closed - 772.1734929184818) < 1e-10
    assert abs(annuity_present_value(payment, rate, n) - closed) < 1e-12


def test_annuity_zero_rate_and_future_value():
    assert abs(annuity_present_value(10.0, 0.0, 8) - 80.0) < 1e-12
    rate = 0.06
    n = 5
    pmt = 20.0
    closed_fv = pmt * ((1.0 + rate) ** n - 1.0) / rate
    assert abs(annuity_future_value(pmt, rate, n) - closed_fv) < 1e-12


def test_npv_known_simple_project():
    # -100 at t=0 and +110 at t=1 is NPV-zero at 10 percent.
    assert abs(net_present_value([-100.0, 110.0], 0.10)) < 1e-12


def test_irr_quadratic_cashflows():
    # Cash flows [-100, 60, 60] satisfy 5 r^2 + 7 r - 1 = 0.
    expected = (-7.0 + math.sqrt(69.0)) / 10.0
    assert abs(expected - 0.1306623862918075) < 1e-12
    estimated = internal_rate_of_return([-100.0, 60.0, 60.0])
    assert abs(estimated - expected) < 1e-8
    assert abs(net_present_value([-100.0, 60.0, 60.0], estimated)) < 1e-8
