"""Figure helpers for the educational demonstrations.

Each function returns a Matplotlib figure. Callers may display or save.
The routines do not fetch data; they plot arrays supplied by the caller.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def _finish(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
    return fig


def plot_efficient_frontier(
    volatility: ArrayLike,
    mean_return: ArrayLike,
    *,
    gmv: tuple[float, float] | None = None,
    path: str | Path | None = None,
) -> plt.Figure:
    """Scatter or curve of mean versus volatility for enumerated portfolios."""
    vol = np.asarray(volatility, dtype=float)
    mu = np.asarray(mean_return, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(vol, mu, color="#1f4e79", lw=1.8, label="Enumerated portfolios")
    if gmv is not None:
        ax.scatter([gmv[0]], [gmv[1]], color="#c0392b", zorder=3, label="GMV")
    ax.set_xlabel("Portfolio volatility")
    ax.set_ylabel("Expected return")
    ax.set_title("Mean–variance illustration (simulated moments)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return _finish(fig, path)


def plot_security_market_line(
    betas: ArrayLike,
    average_returns: ArrayLike,
    sml_betas: ArrayLike,
    sml_returns: ArrayLike,
    *,
    path: str | Path | None = None,
) -> plt.Figure:
    """Average returns against estimated betas, with the theoretical SML overlay."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(
        np.asarray(betas, dtype=float),
        np.asarray(average_returns, dtype=float),
        color="#1f4e79",
        label="Sample average returns",
        zorder=3,
    )
    ax.plot(
        np.asarray(sml_betas, dtype=float),
        np.asarray(sml_returns, dtype=float),
        color="#c0392b",
        lw=1.8,
        label="Security market line",
    )
    ax.set_xlabel("Beta")
    ax.set_ylabel("Average return")
    ax.set_title("Security-market-line illustration on simulated returns")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return _finish(fig, path)


def plot_option_payoffs(
    spots: ArrayLike,
    *,
    strike: float,
    path: str | Path | None = None,
) -> plt.Figure:
    """Call, put, and long-forward payoffs against terminal spot."""
    from qfinmodels.derivatives import call_payoff, forward_payoff, put_payoff

    s = np.asarray(spots, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(s, call_payoff(s, strike), label="Long call", color="#1f4e79")
    ax.plot(s, put_payoff(s, strike), label="Long put", color="#c0392b")
    ax.plot(s, forward_payoff(s, strike), label="Long forward", color="#2e7d32", ls="--")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Terminal spot")
    ax.set_ylabel("Payoff")
    ax.set_title("Terminal payoffs (not present values)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return _finish(fig, path)


def plot_rolling_volatility(
    values: ArrayLike,
    *,
    path: str | Path | None = None,
) -> plt.Figure:
    """Plot a rolling-volatility series, omitting the initial unfilled window."""
    series = np.asarray(values, dtype=float)
    idx = np.arange(series.size)
    mask = np.isfinite(series)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(idx[mask], series[mask], color="#1f4e79", lw=1.4)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Annualised rolling volatility")
    ax.set_title("Rolling volatility on a simulated return path")
    ax.grid(True, alpha=0.3)
    return _finish(fig, path)


def plot_duration_convexity(
    shocks: ArrayLike,
    full_reprice: ArrayLike,
    approximation: ArrayLike,
    *,
    path: str | Path | None = None,
) -> plt.Figure:
    """Full reprice versus duration-plus-convexity approximation."""
    dy = np.asarray(shocks, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dy, np.asarray(full_reprice, dtype=float), color="#1f4e79", lw=1.8, label="Full reprice")
    ax.plot(
        dy,
        np.asarray(approximation, dtype=float),
        color="#c0392b",
        lw=1.6,
        ls="--",
        label="Duration + convexity",
    )
    ax.set_xlabel("Yield shock")
    ax.set_ylabel("Bond price")
    ax.set_title("Local expansion versus exact present value")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return _finish(fig, path)


def plot_var_es(
    returns: ArrayLike,
    var_level: float,
    es_level: float,
    *,
    path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of simulated returns with VaR and expected-shortfall markers."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(r, bins=40, color="#9bb4cc", edgecolor="white", density=True)
    ax.axvline(-var_level, color="#c0392b", lw=1.8, label=f"VaR quantile ({var_level:.4f} loss)")
    ax.axvline(-es_level, color="#1f4e79", lw=1.8, ls="--", label=f"ES ({es_level:.4f} loss)")
    ax.set_xlabel("Simulated return")
    ax.set_ylabel("Density")
    ax.set_title("VaR is a quantile; ES averages the tail beyond it")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return _finish(fig, path)
