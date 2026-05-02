"""Tests for the Signal dataclass."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest.signal import Signal


def test_long_signal_basic():
    s = Signal(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="BTC/USDT",
        direction="long",
        entry_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
    )
    assert s.direction == "LONG"
    assert s.is_long is True
    assert s.risk_pct == pytest.approx(0.05)
    assert s.reward_pct == pytest.approx(0.10)
    assert s.risk_reward == pytest.approx(2.0)


def test_short_signal_basic():
    s = Signal(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="ETH/USDT",
        direction="SHORT",
        entry_price=100.0,
        stop_loss=105.0,
        target_1=90.0,
    )
    assert s.is_long is False
    assert s.risk_pct == pytest.approx(0.05)
    assert s.reward_pct == pytest.approx(0.10)


@pytest.mark.parametrize(
    "kwargs,err",
    [
        # Bad direction
        (
            {
                "direction": "FLAT",
                "entry_price": 100,
                "stop_loss": 95,
                "target_1": 110,
            },
            "direction must be",
        ),
        # LONG with stop above entry
        (
            {
                "direction": "LONG",
                "entry_price": 100,
                "stop_loss": 105,
                "target_1": 110,
            },
            "LONG stop_loss",
        ),
        # LONG with target below entry
        (
            {
                "direction": "LONG",
                "entry_price": 100,
                "stop_loss": 95,
                "target_1": 90,
            },
            "LONG target_1",
        ),
        # SHORT with stop below entry
        (
            {
                "direction": "SHORT",
                "entry_price": 100,
                "stop_loss": 95,
                "target_1": 90,
            },
            "SHORT stop_loss",
        ),
        # SHORT with target above entry
        (
            {
                "direction": "SHORT",
                "entry_price": 100,
                "stop_loss": 105,
                "target_1": 110,
            },
            "SHORT target_1",
        ),
        # Negative entry
        (
            {
                "direction": "LONG",
                "entry_price": -1,
                "stop_loss": 0.5,
                "target_1": 2,
            },
            "entry_price",
        ),
    ],
)
def test_signal_validation(kwargs, err):
    full = {
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "symbol": "BTC/USDT",
        **kwargs,
    }
    with pytest.raises(ValueError, match=err):
        Signal(**full)


def test_signal_from_dict_live_bot_shape():
    raw = {
        "symbol": "BTC/USDT",
        "direction": "🚀 LONG",
        "price": 100.0,
        "target_1": 110.0,
        "target_2": 120.0,
        "stop_loss": 95.0,
        "signal_power": "strong",
        "signal_strength": 75,
        "reasons": ["EMA touch", "RSI bounce"],
        "created_at": "2024-01-01T00:00:00",
    }
    s = Signal.from_dict(raw)
    assert s.symbol == "BTC/USDT"
    assert s.direction == "LONG"
    assert s.entry_price == 100.0
    assert s.target_2 == 120.0
    assert s.signal_strength == 75.0
    assert s.reasons == ("EMA touch", "RSI bounce")


def test_signal_from_dict_missing_target_2_ok():
    raw = {
        "symbol": "ETH/USDT",
        "direction": "SHORT",
        "entry_price": 100.0,
        "target_1": 90.0,
        "target_2": None,
        "stop_loss": 105.0,
        "timestamp": "2024-01-02T00:00:00",
    }
    s = Signal.from_dict(raw)
    assert s.target_2 is None
