"""Tests for the reference strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.strategies import (
    RSIMeanReversion,
    RSIMeanReversionParams,
    SMACrossover,
    SMACrossoverParams,
)


def test_sma_cross_emits_long_after_trend_change():
    """A regime change (down -> up) should produce a long crossover signal."""
    n = 120
    # First 60 bars trend down 100 -> 70, next 60 bars trend up 70 -> 130.
    closes = np.concatenate(
        [
            np.linspace(100.0, 70.0, 60),
            np.linspace(70.0, 130.0, 60),
        ]
    )
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    bars = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n),
        },
        index=pd.date_range(
            "2024-01-01", periods=n, freq="h", tz="UTC"
        ),
    )
    strat = SMACrossover(SMACrossoverParams(fast=5, slow=15, atr_period=10))
    sigs = strat.generate(bars, "BTC/USDT")
    longs = [s for s in sigs if s.is_long]
    assert longs, "expected at least one LONG cross-up after regime change"
    first_long = longs[0]
    assert first_long.target_1 > first_long.entry_price > first_long.stop_loss


def test_sma_cross_emits_signals_in_oscillating_market(oscillating_bars):
    strat = SMACrossover(SMACrossoverParams(fast=5, slow=15, atr_period=10))
    sigs = strat.generate(oscillating_bars, "ETH/USDT")
    longs = [s for s in sigs if s.is_long]
    shorts = [s for s in sigs if not s.is_long]
    assert longs and shorts


def test_sma_cross_long_only_excludes_shorts(oscillating_bars):
    strat = SMACrossover(
        SMACrossoverParams(fast=5, slow=15, atr_period=10, long_only=True)
    )
    sigs = strat.generate(oscillating_bars, "ETH/USDT")
    assert all(s.is_long for s in sigs)


def test_sma_params_validation():
    with pytest.raises(ValueError, match="fast must be"):
        SMACrossoverParams(fast=10, slow=5)
    with pytest.raises(ValueError):
        SMACrossoverParams(fast=0, slow=20)


def test_rsi_mean_reversion_emits_signals(oscillating_bars):
    strat = RSIMeanReversion(
        RSIMeanReversionParams(
            rsi_period=10, oversold=35, overbought=65, atr_period=10
        )
    )
    sigs = strat.generate(oscillating_bars, "ETH/USDT")
    assert sigs, "Oscillating market should produce some RSI signals"
    longs = [s for s in sigs if s.is_long]
    shorts = [s for s in sigs if not s.is_long]
    assert longs and shorts


def test_rsi_long_only_filter(oscillating_bars):
    strat = RSIMeanReversion(
        RSIMeanReversionParams(
            rsi_period=10,
            oversold=35,
            overbought=65,
            atr_period=10,
            long_only=True,
        )
    )
    sigs = strat.generate(oscillating_bars, "ETH/USDT")
    assert all(s.is_long for s in sigs)


def test_rsi_params_validation():
    with pytest.raises(ValueError):
        RSIMeanReversionParams(oversold=70, overbought=30)


def test_strategy_returns_empty_on_short_history():
    strat = SMACrossover()
    short = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
        },
        index=pd.date_range(
            "2024-01-01", periods=1, freq="h", tz="UTC"
        ),
    )
    assert strat.generate(short, "BTC/USDT") == []


def test_strategies_signals_are_sorted(oscillating_bars):
    strat = SMACrossover(SMACrossoverParams(fast=5, slow=15, atr_period=10))
    sigs = strat.generate(oscillating_bars, "BTC/USDT")
    timestamps = [s.timestamp for s in sigs]
    assert timestamps == sorted(timestamps)


def test_atr_helper_handles_constant_price():
    """No price movement -> ATR = 0 -> no signals (degenerate case)."""
    n = 60
    closes = np.full(n, 100.0)
    bars = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.ones(n),
        },
        index=pd.date_range(
            "2024-01-01", periods=n, freq="h", tz="UTC"
        ),
    )
    strat = SMACrossover(SMACrossoverParams(fast=5, slow=15, atr_period=10))
    assert strat.generate(bars, "BTC/USDT") == []
