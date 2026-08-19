# Failures and corrections

A correct implementation of a formula can still be the wrong description of a market. The rows below are numerical or conceptual failures retained in the laboratory.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| Duration–convexity price map for a large yield shock | Absolute error larger than for a 10 bp shock | Reprice the bond at the new yield | Use full reprice for large moves; treat duration as local | `tests/test_fixed_income.py::test_duration_convexity_error_shrinks_for_small_shocks` | Key-rate duration; this is a parallel yield shift |
| Quote VaR as the tail risk number | ES exceeds VaR on a mixed Gaussian-plus-shock sample | ES is the mean of losses beyond the VaR quantile | Report ES (and scenarios); state that VaR ignores loss given exceedance | `tests/test_risk.py::test_var_does_not_exceed_expected_shortfall` | Spectral risk measures; elicitability debates are not implemented |
| Treat CAPM beta on simulated excess returns as a trading rule | The DGP is an illustration | SML plot is labelled simulated | Keep the README sentence on educational use | `tests/test_capm.py`; README disclaimer | Empirical factor models |
| European call from a coarse binomial tree | Need not sit on the Black–Scholes price | Increase steps; loose tolerance in tests | State discrete-time versus continuous-time assumptions | `tests/test_derivatives.py` | American exercise; local vol |
| Put–call identity ignored | Prices would then be internally inconsistent | Parity residual | Enforce or test parity under the same discounting | `tests/test_derivatives.py` | Dividends if the identity is extended |

Model-risk discussion: `MODEL_RISK_NOTES.md`. Process: `docs/lab_process.md`.
