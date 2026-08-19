"""Run the educational demonstrations and write figures.

Market-like series are simulated. Closed-form identities are evaluated
directly. This script is a reproducibility entry point, not a live
valuation service.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qfinmodels.capm import estimate_capm, security_market_line, simulate_capm_returns
from qfinmodels.derivatives import binomial_european, black_scholes
from qfinmodels.fixed_income import (
    bond_convexity,
    bond_price,
    duration_convexity_price,
    macaulay_duration,
    modified_duration,
)
from qfinmodels.plots import (
    plot_duration_convexity,
    plot_efficient_frontier,
    plot_option_payoffs,
    plot_rolling_volatility,
    plot_security_market_line,
    plot_var_es,
)
from qfinmodels.portfolio import (
    efficient_frontier,
    expected_returns,
    global_minimum_variance,
    portfolio_return,
    portfolio_volatility,
    sample_covariance,
)
from qfinmodels.risk import expected_shortfall, historical_var, parametric_var, stress_table
from qfinmodels.tvm import annuity_present_value, internal_rate_of_return, net_present_value
from qfinmodels.volatility import fit_garch11, realized_volatility, rolling_volatility, simulate_garch11


def main() -> None:
    rng = np.random.default_rng(42)
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("=== Time value of money ===")
    annuity = annuity_present_value(100.0, 0.05, 10)
    npv = net_present_value([-100.0, 40.0, 50.0, 40.0], 0.08)
    irr = internal_rate_of_return([-100.0, 60.0, 60.0])
    print(f"annuity PV (100 per period, 5%, 10 periods): {annuity:.6f}")
    print(f"NPV at 8%: {npv:.6f}")
    print(f"IRR of [-100, 60, 60]: {irr:.6f}")

    print("=== Fixed income ===")
    face, coupon, years, ytm = 100.0, 0.05, 10.0, 0.06
    price = bond_price(face, coupon, years, ytm, frequency=1)
    mac = macaulay_duration(face, coupon, years, ytm, frequency=1)
    mod = modified_duration(face, coupon, years, ytm, frequency=1)
    conv = bond_convexity(face, coupon, years, ytm, frequency=1)
    print(f"bond price: {price:.6f}")
    print(f"Macaulay duration (years): {mac:.6f}")
    print(f"modified duration: {mod:.6f}")
    print(f"convexity: {conv:.6f}")
    shocks = np.linspace(-0.02, 0.02, 21)
    full = np.array([bond_price(face, coupon, years, ytm + dy, frequency=1) for dy in shocks])
    approx = np.array([duration_convexity_price(price, mod, conv, dy) for dy in shocks])
    fig = plot_duration_convexity(shocks, full, approx, path=figures / "duration_convexity.png")
    plt_close(fig)

    print("=== Portfolio ===")
    means = np.array([0.08, 0.05, 0.065])
    vols = np.array([0.18, 0.09, 0.14])
    corr = np.array(
        [
            [1.00, 0.35, 0.45],
            [0.35, 1.00, 0.30],
            [0.45, 0.30, 1.00],
        ]
    )
    cov = np.outer(vols, vols) * corr
    w_gmv = global_minimum_variance(cov, long_only=False)
    w_long = global_minimum_variance(cov, long_only=True)
    print(f"unconstrained GMV weights: {w_gmv}")
    print(f"long-only GMV weights:     {w_long}")
    print(f"unconstrained GMV vol: {portfolio_volatility(w_gmv, cov):.6f}")
    frontier_vol, frontier_mu, _ = efficient_frontier(means, cov, n_points=30, long_only=False)
    fig = plot_efficient_frontier(
        frontier_vol,
        frontier_mu,
        gmv=(portfolio_volatility(w_gmv, cov), portfolio_return(w_gmv, means)),
        path=figures / "efficient_frontier.png",
    )
    plt_close(fig)

    simulated = rng.multivariate_normal(means / 252.0, cov / 252.0, size=750)
    print(f"sample means (daily): {expected_returns(simulated)}")
    print(f"sample covariance diagonal: {np.diag(sample_covariance(simulated))}")

    print("=== CAPM (simulated) ===")
    true_betas = np.array([0.6, 1.0, 1.4])
    rf = 0.0002
    market, assets = simulate_capm_returns(true_betas, n_obs=600, rf=rf, seed=42)
    est_betas = []
    avg_ret = []
    for i in range(true_betas.size):
        fit = estimate_capm(assets[:, i], market, rf=rf)
        est_betas.append(fit.beta)
        avg_ret.append(float(np.mean(assets[:, i])))
        print(
            f"asset {i}: true beta={true_betas[i]:.2f}, "
            f"OLS beta={fit.beta:.3f}, alpha={fit.alpha:.6f}, R^2={fit.r_squared:.3f}"
        )
    est_betas = np.array(est_betas)
    mkt_premium = float(np.mean(market) - rf)
    grid = np.linspace(0.4, 1.6, 20)
    fig = plot_security_market_line(
        est_betas,
        np.array(avg_ret),
        grid,
        security_market_line(grid, rf, mkt_premium),
        path=figures / "security_market_line.png",
    )
    plt_close(fig)

    print("=== Derivatives ===")
    s0, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_call = black_scholes(s0, k, t, r, sigma, option="call")
    bs_put = black_scholes(s0, k, t, r, sigma, option="put")
    bin_call = binomial_european(s0, k, t, r, sigma, n_steps=100, option="call")
    print(f"Black-Scholes call: {bs_call:.6f}")
    print(f"Black-Scholes put:  {bs_put:.6f}")
    print(f"binomial call (n=100): {bin_call:.6f}")
    spots = np.linspace(50.0, 150.0, 201)
    fig = plot_option_payoffs(spots, strike=k, path=figures / "option_payoffs.png")
    plt_close(fig)

    print("=== Risk ===")
    garch_ret, _ = simulate_garch11(800, omega=0.00002, alpha=0.08, beta=0.88, seed=7)
    hvar = historical_var(garch_ret, alpha=0.05)
    pvar = parametric_var(garch_ret, alpha=0.05)
    es = expected_shortfall(garch_ret, alpha=0.05)
    print(f"historical VaR 5%: {hvar:.6f}")
    print(f"parametric VaR 5%: {pvar:.6f}")
    print(f"expected shortfall 5%: {es:.6f}")
    stresses = stress_table(
        np.array([0.5, 0.3, 0.2]),
        {
            "parallel_down": np.array([-0.08, -0.03, -0.05]),
            "equity_crash": np.array([-0.20, -0.02, -0.10]),
            "flight_to_quality": np.array([-0.12, 0.04, -0.04]),
        },
    )
    print(f"stress P&L: {stresses}")
    fig = plot_var_es(garch_ret, hvar, es, path=figures / "var_es.png")
    plt_close(fig)

    print("=== Volatility ===")
    rv = realized_volatility(garch_ret)
    roll = rolling_volatility(garch_ret, window=40)
    fitted = fit_garch11(garch_ret)
    print(f"realized vol (ann.): {rv:.6f}")
    print(
        f"GARCH(1,1) QML: omega={fitted.omega:.6g}, alpha={fitted.alpha:.4f}, "
        f"beta={fitted.beta:.4f}, persistence={fitted.persistence:.4f}"
    )
    fig = plot_rolling_volatility(roll, path=figures / "rolling_volatility.png")
    plt_close(fig)

    print("Wrote figures to", figures)


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    main()
