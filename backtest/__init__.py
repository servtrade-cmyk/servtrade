"""Backtest module for botscan.

Provides historical OHLCV loading, vectorized trade simulation, performance
metrics, and reference strategies. The engine is signal-driven: it accepts
signals shaped like the dicts produced by the live scanner (entry price,
target_1, target_2, stop_loss, direction) so the same signal generator can
be evaluated live or on history.

Public API:

    from backtest import (
        Signal,
        TradeResult,
        BacktestResult,
        BacktestConfig,
        run_backtest,
        load_ohlcv,
        compute_metrics,
    )
"""

from backtest.config import BacktestConfig
from backtest.data_loader import load_ohlcv
from backtest.engine import BacktestResult, TradeResult, run_backtest
from backtest.metrics import compute_metrics
from backtest.signal import Signal

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Signal",
    "TradeResult",
    "compute_metrics",
    "load_ohlcv",
    "run_backtest",
]
