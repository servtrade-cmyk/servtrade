"""Tests for the signal_replay loader."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from backtest import signal_replay


def _write_db(path, signals):
    db = {"signals": {f"sig_{i}": s for i, s in enumerate(signals)}}
    path.write_text(json.dumps(db), encoding="utf-8")


@pytest.fixture
def db_with_mixed_signals(tmp_path):
    path = tmp_path / "signals_database.json"
    sigs = [
        {
            "symbol": "BTC/USDT",
            "direction": "🚀 LONG",
            "price": 100.0,
            "target_1": 110.0,
            "target_2": 120.0,
            "stop_loss": 95.0,
            "signal_power": "strong",
            "signal_strength": 80,
            "reasons": ["EMA touch"],
            "created_at": datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "type": "regular",
        },
        {
            "symbol": "BTC/USDT",
            "direction": "⬇️ SHORT",
            "price": 200.0,
            "target_1": 190.0,
            "target_2": None,
            "stop_loss": 210.0,
            "signal_strength": 50,
            "created_at": datetime(2024, 1, 2, tzinfo=timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "type": "pump",
        },
        {
            "symbol": "ETH/USDT",
            "direction": "LONG",
            "price": 50.0,
            "target_1": 55.0,
            "stop_loss": 47.0,
            "signal_strength": 70,
            "created_at": datetime(2024, 1, 3, tzinfo=timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "type": "regular",
        },
        # Malformed: missing target_1 — should be skipped.
        {
            "symbol": "DOGE/USDT",
            "direction": "LONG",
            "price": 0.1,
            "stop_loss": 0.09,
            "created_at": datetime(2024, 1, 4, tzinfo=timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "type": "regular",
        },
    ]
    _write_db(path, sigs)
    return path


def test_load_signals_default_loads_all_valid(db_with_mixed_signals):
    out = signal_replay.load_signals_from_db(db_with_mixed_signals)
    # Malformed DOGE skipped.
    assert len(out) == 3
    # Sorted by timestamp.
    assert [s.symbol for s in out] == ["BTC/USDT", "BTC/USDT", "ETH/USDT"]


def test_load_signals_filter_by_symbol(db_with_mixed_signals):
    out = signal_replay.load_signals_from_db(
        db_with_mixed_signals, symbol="BTC/USDT"
    )
    assert all(s.symbol == "BTC/USDT" for s in out)
    assert len(out) == 2


def test_load_signals_filter_by_direction(db_with_mixed_signals):
    out = signal_replay.load_signals_from_db(
        db_with_mixed_signals, direction="long"
    )
    assert all(s.direction == "LONG" for s in out)


def test_load_signals_filter_by_type(db_with_mixed_signals):
    out = signal_replay.load_signals_from_db(
        db_with_mixed_signals, signal_type="pump"
    )
    assert len(out) == 1
    assert out[0].direction == "SHORT"


def test_load_signals_min_strength(db_with_mixed_signals):
    out = signal_replay.load_signals_from_db(
        db_with_mixed_signals, min_strength=70.0
    )
    # Only the strength-80 BTC long and strength-70 ETH long pass.
    assert len(out) == 2


def test_load_signals_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        signal_replay.load_signals_from_db(tmp_path / "nope.json")


def test_signals_to_dicts_round_trip(db_with_mixed_signals):
    sigs = signal_replay.load_signals_from_db(db_with_mixed_signals)
    dicts = signal_replay.signals_to_dicts(sigs)
    assert all("price" in d and "entry_price" in d for d in dicts)
    assert all(d["direction"] in {"LONG", "SHORT"} for d in dicts)
