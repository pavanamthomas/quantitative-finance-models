# quantitative-finance-models

[![CI](https://github.com/pavanamthomas/quantitative-finance-models/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/quantitative-finance-models/actions)

Educational quantitative-finance models covering valuation, portfolios, fixed income, derivatives, risk, and model validation.

This repository is an analytical and educational quantitative-finance portfolio. It does not represent trading advice or professional trading performance.

The library implements textbook identities and estimators used in valuation, mean–variance algebra, linear factor illustrations, elementary derivatives pricing, and loss-quantile risk summaries. Market-like series in scripts, tests, and the notebook are **simulated** unless a quantity is obtained from a known closed form. Third-party market datasets are not included; see `docs/data_policy.md`.

Related work:

- Time-series estimation and forecast evaluation: [time-series-forecasting-lab](https://github.com/pavanamthomas/time-series-forecasting-lab)
- Constrained optimisation and decision formulations: [optimization-decision-models](https://github.com/pavanamthomas/optimization-decision-models)

## Methodology

Work in this repository follows a single sequence:

**Problem → formalization → assumptions → computation/estimation → validation → interpretation → limitations**

The test suite addresses the validation step for identities that admit a known numerical check. `MODEL_RISK_NOTES.md` addresses the limitations step: a correct implementation of a formula does not establish that the formula is an adequate description of a market.

## Contents

| Area | Module | What is implemented |
| --- | --- | --- |
| Time value of money | `qfinmodels.tvm` | Present and future value, discount factors, level annuities, NPV, IRR |
| Fixed income | `qfinmodels.fixed_income` | Bond price and yield, Macaulay and modified duration, convexity, duration–convexity price approximation |
| Portfolios | `qfinmodels.portfolio` | Means, covariances, portfolio variance, diversification comparison, global-minimum-variance weights, efficient-frontier coordinates |
| CAPM / factor illustration | `qfinmodels.capm` | OLS beta on simulated excess returns, security-market-line coordinates |
| Derivatives | `qfinmodels.derivatives` | Forward and option payoffs, put–call parity, Cox–Ross–Rubinstein binomial prices, Black–Scholes |
| Risk | `qfinmodels.risk` | Historical VaR, parametric VaR, expected shortfall (CVaR), scenario P&L, stress tables |
| Volatility | `qfinmodels.volatility` | Realized volatility, rolling volatility, Gaussian GARCH(1,1) by quasi-maximum likelihood |
| Figures | `qfinmodels.plots` | Efficient frontier, SML, payoffs, rolling volatility, duration–convexity error, VaR versus ES |

Long-only portfolio constraints are optional. The unconstrained global-minimum-variance solution is the closed-form mean–variance object; non-negative weights are obtained by a bounded quadratic programme and are documented as a restriction, not as a claim about implementable mandates.

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Dependencies are `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, and `pytest`. There is no market-data client.

## Reproducing the demonstrations

```bash
python scripts/run_all.py
pytest
```

`scripts/run_all.py` writes figures under `figures/` using a fixed random seed. The figures directory is an output location; PNG files are not source data.

The notebook `notebooks/01_fixed_income_and_risk.ipynb` walks through bond valuation, duration–convexity approximation, and loss quantiles on simulated returns, using the same methodology sequence.

## Tests

Tests compare implementations with independently computed textbook-style numbers and with inequalities that must hold for the objects as defined:

- Bond prices fall when yield rises for a fixed cash-flow schedule.
- Annuity present value matches the closed-form geometric sum.
- European put–call parity holds within tolerance.
- Binomial European call prices move toward Black–Scholes as the step count increases (loose tolerance).
- Portfolio weights sum to one; GMV weights satisfy the stated constraint set.
- Expected shortfall is at least as large as VaR on the same sample and level.
- Duration–convexity approximation error is smaller for a small yield shock than for a large shock.

## What this repository is not

- Not a trading system, signal library, or performance track record.
- Not a substitute for a term-structure model, a default model, or a market-risk engine.
- Not calibrated to quoted prices. Black–Scholes and binomial values are functions of inputs you supply.

## Author

Dr. Pavanam Thomas  
GitHub: [pavanamthomas](https://github.com/pavanamthomas)  
Email: thomaspavanam@gmail.com

## Citation

See `CITATION.cff`.

## License

Copyright 2026 Dr. Pavanam Thomas. MIT License; see `LICENSE`.
