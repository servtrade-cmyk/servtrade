"""Tests for the metrics module."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backtest.config import BacktestConfig
from backtest.engine import BacktestResult, TradeResult
from backtest.metrics import compute_metrics
from backtest.signal import Signal


def _equity(values):
    idx = pd.date_range(
        "2024-01-01", periods=len(values), freq="D", tz="UTC"
    )
    return pd.Series(values, index=idx, name="equity", dtype="float64")


def _trade(pnl_quote: float, pnl_pct: float, bars: int = 1) -> TradeResult:
    sig = Signal(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="BTC/USDT",
        direction="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
    )
    return TradeResult(
        signal=sig,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        entry_price=100.0,
        exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        exit_price=110.0 if pnl_quote > 0 else 95.0,
        exit_reason="target_1" if pnl_quote > 0 else "stop_loss",
        qty=1.0,
        pnl_quote=pnl_quote,
        pnl_pct=pnl_pct,
        fees_paid=0.0,
        bars_held=bars,
    )


def test_no_trades_returns_zero_metrics():
    cfg = BacktestConfig()
    result = BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=_equity([10_000.0] * 10),
        bars_processed=10,
    )
    m = compute_metrics(result)
    assert m.n_trades == 0
    assert m.win_rate_pct == 0.0
    assert m.total_return_pct == 0.0
    assert m.max_drawdown_pct == 0.0


def test_max_drawdown_basic():
    # Equity goes 100 -> 120 -> 80 -> 110.
    # Max DD = (120 - 80) / 120 = 33.33%.
    cfg = BacktestConfig(initial_capital=100.0)
    result = BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=_equity([100.0, 120.0, 80.0, 110.0]),
        bars_processed=4,
    )
    m = compute_metrics(result)
    assert m.max_drawdown_pct == pytest.approx(100 * (120 - 80) / 120)


def test_total_return_and_cagr():
    eq = _equity([100.0, 110.0, 121.0, 133.1])  # ~10% per bar
    cfg = BacktestConfig(initial_capital=100.0)
    result = BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=eq,
        bars_processed=4,
    )
    m = compute_metrics(result, periods_per_year=365.0)
    assert m.total_return_pct == pytest.approx(33.1, rel=1e-3)
    # CAGR = (1.331 ^ (365/3)) - 1 — huge number, just sanity-check positive.
    assert m.cagr_pct > 0


def test_win_rate_and_profit_factor():
    cfg = BacktestConfig()
    trades = [
        _trade(pnl_quote=200.0, pnl_pct=0.02),
        _trade(pnl_quote=150.0, pnl_pct=0.015),
        _trade(pnl_quote=-100.0, pnl_pct=-0.01),
        _trade(pnl_quote=-50.0, pnl_pct=-0.005),
    ]
    result = BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=_equity([10_000, 10_200, 10_350, 10_250, 10_200]),
        bars_processed=5,
    )
    m = compute_metrics(result)
    assert m.n_trades == 4
    assert m.win_rate_pct == pytest.approx(50.0)
    # Profit factor = (200 + 150) / (100 + 50) = 350 / 150 = 2.333
    assert m.profit_factor == pytest.approx(350 / 150)


def test_profit_factor_no_losses_is_inf():
    cfg = BacktestConfig()
    trades = [
        _trade(pnl_quote=100.0, pnl_pct=0.01),
        _trade(pnl_quote=200.0, pnl_pct=0.02),
    ]
    result = BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=_equity([10_000, 10_100, 10_300]),
        bars_processed=3,
    )
    m = compute_metrics(result)
    assert m.profit_factor == float("inf")


def test_sharpe_zero_volatility_zero():
    cfg = BacktestConfig(initial_capital=10_000.0)
    eq = _equity([10_000.0] * 30)
    result = BacktestResult(
        config=cfg, trades=[], equity_curve=eq, bars_processed=30
    )
    m = compute_metrics(result)
    assert m.sharpe == 0.0


def test_sharpe_positive_for_uptrending_equity():
    cfg = BacktestConfig(initial_capital=10_000.0)
    eq = _equity([10_000.0 * (1.001) ** i for i in range(100)])
    # Add small Gaussian noise so std > 0; otherwise the constant-growth
    # equity curve has zero variance and Sharpe collapses to 0 by the
    # zero-volatility guard.
    rng = np.random.default_rng(1)
    eq_noisy = eq * (1 + rng.normal(0, 0.0005, size=len(eq)))
    eq_noisy = pd.Series(eq_noisy, index=eq.index)
    result = BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=eq_noisy,
        bars_processed=100,
    )
    metrics = compute_metrics(result)
    assert metrics.sharpe > 0


def test_exposure_pct():
    cfg = BacktestConfig()
    trades = [
        _trade(pnl_quote=100.0, pnl_pct=0.01, bars=10),
        _trade(pnl_quote=-50.0, pnl_pct=-0.005, bars=20),
    ]
    result = BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=_equity([10_000.0] * 100),
        bars_processed=100,
    )
    m = compute_metrics(result)
    # Total bars held = 30, total bars = 100 -> 30%
    assert m.exposure_pct == pytest.approx(30.0)
