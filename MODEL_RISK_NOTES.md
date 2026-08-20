# Model risk notes

These notes sit beside the code. The library evaluates identities and
estimators under stated assumptions. A number that prints cleanly is not, by
itself, evidence that a model is adequate for a decision.

Model risk enters as soon as the formula is treated as a description of a
market rather than as an identity under its assumptions.

## Model assumptions

Every routine encodes a mapping from primitives to an output. Examples:

- Discounting at a single constant rate treats the term structure as flat and
  known.
- Mean–variance portfolio algebra treats the covariance matrix as the complete
  description of joint risk.
- Black–Scholes treats the underlying as a geometric Brownian motion with
  constant volatility, frictionless trading, and a constant risk-free rate.
- Historical value-at-risk treats the recent empirical distribution as the
  relevant loss law over the chosen horizon.

If an assumption is false, the output remains well-defined as a number and
undefined as an answer to the original problem. Checking that code matches a
formula does not check that the formula matches the phenomenon.

## Parameter uncertainty

Estimated quantities (yield, beta, GARCH coefficients, covariance entries)
are functions of a finite sample. Sampling variation, estimator choice, and
numerical solver settings all move the reported value. Point estimates used
as if they were known constants understate uncertainty. Where the code
returns a single vector of weights or a single volatility path, that
sharpness is a property of the estimator, not of the world.

A related failure mode is over-conditioning: a covariance matrix estimated
on a short window can be nearly singular, so global-minimum-variance weights
become unstable even when the algebra is correct.

## Distributional assumptions

Parametric value-at-risk and Black–Scholes both lean on Gaussian structure
(returns, or log-returns after Itô). Many financial series exhibit heavier
tails, skewness, and volatility clustering. Under a thin-tailed law, extreme
quantiles and expected shortfall are too optimistic. Under a misspecified
innovation distribution, GARCH variance forecasts can be tight and still
wrong in the tails.

Non-parametric historical estimators avoid a named density, but they replace
that assumption with a stationarity and i.i.d.-window assumption that is
itself a distributional claim about the sample.

## Tail risk

Value-at-risk at level α is a quantile. It is silent about the shape of the
loss distribution beyond that quantile. Two loss laws can share a VaR and
differ arbitrarily in expected shortfall. That is why the tests insist that
expected shortfall is at least as large as VaR on the same sample and level:
the inequality is a coherence check, not a claim that either number is a
sufficient risk summary.

Duration is a local linear map. Convexity extends the map by one quadratic
term. Neither object describes default, liquidity gaps, or discontinuous
yield-curve moves. Tail events in fixed income often live outside the
duration–convexity chart.

## Calibration risk

A model that is fitted to one object (an implied volatility, a historical
covariance, an annuity rate) need not be reliable for another object (a
stress P&L, an out-of-sample path, a different tenor). Binomial prices
approach Black–Scholes as the step count grows when both models share the
same continuous-time limit; that numerical convergence is not a calibration
to market prices.

GARCH(1,1) estimated by Gaussian quasi-maximum likelihood can produce
parameters inside the stationarity region and still be a poor description of
the next month if the likelihood surface is flat or if the sample is short.

## Data dependence

Estimators inherit the dependence structure of the sample. Overlapping
returns, stale quotes, and shared factors induce correlation across
observations and across names. Standard errors that assume independence are
then too small. This repository avoids third-party market files, so the
dependence that appears in demonstrations is the dependence that the
simulator was told to produce. On real series the dependence is unknown and
must be argued, not assumed.

See `docs/data_policy.md` for what this repository will and will not store.

## Regime changes

Stationary estimators average across whatever regimes are present in the
window. A covariance estimated through a low-volatility period will understate
risk if a high-volatility regime arrives. A beta estimated in one policy
environment need not travel to another. Nothing in the algebra of CAPM or
mean–variance optimisation detects a break; detection is an extra statistical
exercise, and even a detected break does not automatically yield the correct
post-break parameters.

## Backtesting concepts

A risk or pricing model is not confirmed by a single in-sample fit. Useful
backtesting questions include:

- Unconditional coverage: does the hit rate of VaR exceptions match the
  nominal level?
- Independence of exceptions: do violations cluster, as they will under
  omitted volatility dynamics?
- Sensitivity to window length and to the estimation scheme.
- Comparison against a simpler baseline (historical versus parametric VaR;
  duration versus full reprice).

A clean backtest on simulated data generated from the same model is a
software test. A clean backtest on a short market sample is still compatible
with substantial model risk, especially if the sample contains few tail
events.

## Why model output is not self-validating

Internal consistency is necessary and not sufficient. Put–call parity holding
inside a numerical tolerance shows that the call, put, discount factor, and
spot were combined as the identity requires. It does not show that the
Black–Scholes inputs are the right inputs, or that European exercise is the
right contract, or that the absence of dividends is true.

Likewise, a bond price that falls when yield rises confirms monotonicity of
the present-value map for fixed cash flows. It does not confirm that the
cash flows are fixed, that the yield is the relevant discount rate, or that
the duration approximation will remain accurate under a large, non-parallel
curve shift.

Validation in this repository therefore has two layers:

1. Numerical fidelity to a stated formula or inequality (the test suite).
2. Explicit limitations of the mapping from formula to decision (these notes,
   module docstrings, and the notebook).

Layer 1 can be automated. Layer 2 cannot be replaced by a green test run.
