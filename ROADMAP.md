# Roadmap

Current as of August 2026.

This repository is an analytical and educational quantitative-finance portfolio. It does not represent trading advice or professional trading performance.

## In scope now

- TVM identities, bonds (price, yield, duration, convexity), mean–variance algebra, CAPM illustration on simulated excess returns, binomial and Black–Scholes, historical and parametric VaR, expected shortfall, Gaussian GARCH-style QMLE.
- Tests against closed forms and monotonicity identities.
- `MODEL_RISK_NOTES.md`.

## Failures that are part of the design

- Duration–convexity approximation error grows with the yield shock.
- VaR is not a complete tail functional: expected shortfall is more extreme on the mixed sample used in tests.
- Black–Scholes and binomial prices agree only under the stated assumptions and, for the tree, in the large-step limit within a loose tolerance.

Details: `docs/failures_and_corrections.md`.

## Open (issues)

1. No third-party market dataset is committed. A download script for a public series would need an explicit redistribution check before any file is stored.
2. Historical VaR/ES are iid sample functionals. They are not a backtest against a trading book.
3. GARCH QMLE is Gaussian. t-innovations and variance targeting are not implemented.
4. Multi-factor models beyond a single simulated beta illustration are out of scope.

## Explicitly not in scope

- Live P&L, capacity, or “strategy performance”.
- yfinance or other market-data clients.
- Treating a green test suite as model validation in the sense of a bank’s model-risk policy.

Close an issue only with a test, a note in `MODEL_RISK_NOTES.md`, or an explicit limitation.
