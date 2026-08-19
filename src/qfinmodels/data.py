"""Refuse committed market series.

The laboratory does not download or store third-party prices. Estimators
accept arrays the caller supplies; those arrays must not be committed here.
"""


def load_committed_market_series() -> None:
    raise FileNotFoundError(
        "This laboratory does not commit third-party market series. "
        "Use the simulate_* functions, or pass your own array without "
        "storing it in the repository. See docs/data_policy.md."
    )
