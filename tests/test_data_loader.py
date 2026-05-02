"""Tests for the OHLCV data loader.

These tests stub out CCXT so they can run offline and deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from backtest import data_loader


class _FakeExchange:
    rateLimit = 0  # no sleeping in tests

    def __init__(self, rows_by_since: dict[int, list[list[float]]]):
        self._rows = rows_by_since

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        # Find the bucket whose key is closest to the requested 'since'.
        if not self._rows:
            return []
        key = max(k for k in self._rows if k <= since) if any(
            k <= since for k in self._rows
        ) else None
        if key is None:
            return []
        rows = [r for r in self._rows[key] if r[0] >= since]
        return rows[:limit]


@pytest.fixture
def fake_exchange_rows():
    """Two pages worth of synthetic 1h BTC bars."""
    base = int(
        datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    )
    hour_ms = 3_600_000
    page1 = [
        [base + i * hour_ms, 100 + i, 101 + i, 99 + i, 100 + i, 10]
        for i in range(0, 50)
    ]
    page2 = [
        [base + i * hour_ms, 100 + i, 101 + i, 99 + i, 100 + i, 10]
        for i in range(50, 100)
    ]
    return {base: page1, base + 50 * hour_ms: page2}


def test_load_ohlcv_writes_and_reads_cache(
    monkeypatch, tmp_path, fake_exchange_rows
):
    fake = _FakeExchange(fake_exchange_rows)
    monkeypatch.setattr(data_loader, "_build_exchange", lambda *a, **k: fake)

    bars = data_loader.load_ohlcv(
        exchange="testex",
        symbol="BTC/USDT",
        timeframe="1h",
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 1, 5, tzinfo=timezone.utc),
        limit=50,
        cache_dir=tmp_path,
    )
    assert not bars.empty
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(bars.index, pd.DatetimeIndex)
    cache_path = tmp_path / "testex" / "1h" / "BTC_USDT.parquet"
    assert cache_path.exists()


def test_load_ohlcv_returns_cached_on_second_call(
    monkeypatch, tmp_path, fake_exchange_rows
):
    fake = _FakeExchange(fake_exchange_rows)
    calls = {"n": 0}

    def factory(*a, **k):
        calls["n"] += 1
        return fake

    monkeypatch.setattr(data_loader, "_build_exchange", factory)
    common = dict(
        exchange="testex",
        symbol="BTC/USDT",
        timeframe="1h",
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 1, 3, tzinfo=timezone.utc),
        limit=200,
        cache_dir=tmp_path,
    )
    first = data_loader.load_ohlcv(**common)
    n_after_first = calls["n"]
    second = data_loader.load_ohlcv(**common)
    # Second call should use cache and not re-instantiate the exchange
    # because the bounded 'until' is fully covered.
    assert calls["n"] == n_after_first
    pd.testing.assert_frame_equal(first, second)


def test_unsupported_timeframe_raises():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        data_loader._timeframe_ms("17m")


def test_safe_filename():
    assert (
        data_loader._safe_filename("BTC/USDT:USDT") == "BTC_USDT_USDT"
    )


def test_to_utc_ms_handles_iso_string():
    ms = data_loader._to_utc_ms("2024-01-01T00:00:00+00:00")
    assert ms == int(
        datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    )


def test_to_utc_ms_treats_naive_as_utc():
    ms = data_loader._to_utc_ms(datetime(2024, 1, 1))
    assert ms == int(
        datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    )


def test_load_ohlcv_with_refresh_ignores_cache(
    monkeypatch, tmp_path, fake_exchange_rows
):
    fake = _FakeExchange(fake_exchange_rows)
    calls = {"n": 0}

    def factory(*a, **k):
        calls["n"] += 1
        return fake

    monkeypatch.setattr(data_loader, "_build_exchange", factory)
    common = dict(
        exchange="testex",
        symbol="BTC/USDT",
        timeframe="1h",
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 1, 3, tzinfo=timezone.utc),
        limit=200,
        cache_dir=tmp_path,
    )
    data_loader.load_ohlcv(**common)
    n_after_first = calls["n"]
    data_loader.load_ohlcv(refresh=True, **common)
    assert calls["n"] > n_after_first


def test_load_ohlcv_cache_disabled(
    monkeypatch, tmp_path, fake_exchange_rows
):
    fake = _FakeExchange(fake_exchange_rows)
    monkeypatch.setattr(data_loader, "_build_exchange", lambda *a, **k: fake)
    bars = data_loader.load_ohlcv(
        exchange="testex",
        symbol="BTC/USDT",
        timeframe="1h",
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 1, 2, tzinfo=timezone.utc),
        limit=200,
        cache_dir=tmp_path,
        use_cache=False,
    )
    assert not bars.empty
    cache_path = (
        tmp_path / "testex" / "1h" / "BTC_USDT.parquet"
    )
    assert not cache_path.exists()
