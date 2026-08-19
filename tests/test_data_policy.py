"""Data-policy enforcement."""

from __future__ import annotations

import pytest

from qfinmodels.data import load_committed_market_series


def test_committed_market_series_are_refused():
    with pytest.raises(FileNotFoundError, match="does not commit"):
        load_committed_market_series()
