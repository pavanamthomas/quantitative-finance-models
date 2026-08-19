"""Educational quantitative-finance identities and estimators.

The public surface is organised by subject: time value of money, fixed
income, mean-variance portfolios, CAPM-style regressions, elementary
derivatives, loss-quantile risk, and volatility. Simulated series used
in demonstrations are generated in scripts and tests; they are not
market observations.
"""

from qfinmodels.capm import (
    estimate_capm,
    security_market_line,
    simulate_capm_returns,
)
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
from qfinmodels.fixed_income import (
    bond_convexity,
    bond_price,
    bond_yield,
    duration_convexity_price,
    macaulay_duration,
    modified_duration,
)
from qfinmodels.portfolio import (
    efficient_frontier,
    expected_returns,
    global_minimum_variance,
    portfolio_return,
    portfolio_variance,
    sample_covariance,
)
from qfinmodels.risk import (
    expected_shortfall,
    historical_var,
    parametric_var,
    scenario_pnl,
    stress_table,
)
from qfinmodels.tvm import (
    annuity_future_value,
    annuity_present_value,
    discount_factor,
    future_value,
    internal_rate_of_return,
    net_present_value,
    present_value,
)
from qfinmodels.volatility import (
    fit_garch11,
    realized_volatility,
    rolling_volatility,
    simulate_garch11,
)

__version__ = "0.1.0"

__all__ = [
    "annuity_future_value",
    "annuity_present_value",
    "binomial_european",
    "black_scholes",
    "bond_convexity",
    "bond_price",
    "bond_yield",
    "call_payoff",
    "discount_factor",
    "duration_convexity_price",
    "efficient_frontier",
    "estimate_capm",
    "expected_returns",
    "expected_shortfall",
    "fit_garch11",
    "forward_payoff",
    "forward_price",
    "future_value",
    "futures_payoff",
    "global_minimum_variance",
    "historical_var",
    "internal_rate_of_return",
    "macaulay_duration",
    "modified_duration",
    "net_present_value",
    "parametric_var",
    "portfolio_return",
    "portfolio_variance",
    "present_value",
    "put_call_parity_gap",
    "put_payoff",
    "realized_volatility",
    "rolling_volatility",
    "sample_covariance",
    "scenario_pnl",
    "security_market_line",
    "simulate_capm_returns",
    "simulate_garch11",
    "stress_table",
]
