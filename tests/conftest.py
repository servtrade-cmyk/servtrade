"""Shared pytest fixtures for the backtest test suite."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is on sys.path when running pytest from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_bars(
    n: int,
    start: datetime,
    open_prices: list[float] | np.ndarray,
    high_offsets: list[float] | np.ndarray | None = None,
    low_offsets: list[float] | np.ndarray | None = None,
    close_prices: list[float] | np.ndarray | None = None,
    freq: str = "h",
) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame for tests."""
    if len(open_prices) != n:
        raise AssertionError("open_prices length mismatch")
    open_arr = np.asarray(open_prices, dtype=float)
    if close_prices is None:
        close_arr = open_arr.copy()
    else:
        close_arr = np.asarray(close_prices, dtype=float)
    if high_offsets is None:
        high_arr = np.maximum(open_arr, close_arr)
    else:
        high_arr = np.maximum(open_arr, close_arr) + np.asarray(
            high_offsets, dtype=float
        )
    if low_offsets is None:
        low_arr = np.minimum(open_arr, close_arr)
    else:
        low_arr = np.minimum(open_arr, close_arr) - np.asarray(
            low_offsets, dtype=float
        )
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": np.ones(n) * 100.0,
        },
        index=idx,
    )


@pytest.fixture
def make_bars():
    return _make_bars


@pytest.fixture
def trending_bars(make_bars):
    """200 1h bars trending up linearly from 100 to 130."""
    n = 200
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    closes = np.linspace(100.0, 130.0, n)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n) * 100.0,
        },
        index=idx,
    )


@pytest.fixture
def oscillating_bars(make_bars):
    """200 1h bars oscillating in a sine wave around 100, range +/- 10."""
    n = 200
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t = np.linspace(0, 8 * np.pi, n)
    closes = 100.0 + 10.0 * np.sin(t)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.3
    lows = np.minimum(opens, closes) - 0.3
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n) * 100.0,
        },
        index=idx,
    )


@pytest.fixture
def epoch():
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def hour():
    return timedelta(hours=1)
