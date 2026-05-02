"""SMA crossover strategy (trend-following reference)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.signal import Signal
from backtest.strategies.base import Strategy, StrategyParams


@dataclass(frozen=True)
class SMACrossoverParams(StrategyParams):
    """Parameters for SMA crossover.

    Attributes:
        fast: Fast SMA period.
        slow: Slow SMA period (must be > ``fast``).
        atr_period: ATR period for stop / target sizing.
        atr_stop_mult: Stop = entry -/+ ``atr_stop_mult`` * ATR.
        atr_target_mult: Target_1 = entry +/- ``atr_target_mult`` * ATR.
        long_only: If True, ignore short crossovers.
    """

    fast: int = 20
    slow: int = 50
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    atr_target_mult: float = 3.0
    long_only: bool = False

    def __post_init__(self) -> None:
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("SMA periods must be positive")
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.atr_stop_mult <= 0 or self.atr_target_mult <= 0:
            raise ValueError("ATR multipliers must be positive")


def _atr(bars: pd.DataFrame, period: int) -> pd.Series:
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


class SMACrossover(Strategy):
    """Long when fast SMA crosses above slow; short on the inverse."""

    name = "sma_cross"

    def __init__(self, params: SMACrossoverParams | None = None) -> None:
        self.params = params or SMACrossoverParams()

    def generate(self, bars: pd.DataFrame, symbol: str) -> list[Signal]:
        p = self.params
        if len(bars) < p.slow + p.atr_period + 2:
            return []

        close = bars["close"]
        fast_sma = close.rolling(p.fast, min_periods=p.fast).mean()
        slow_sma = close.rolling(p.slow, min_periods=p.slow).mean()
        atr = _atr(bars, p.atr_period)

        diff = fast_sma - slow_sma
        prev_diff = diff.shift(1)
        cross_up = (prev_diff <= 0) & (diff > 0)
        cross_dn = (prev_diff >= 0) & (diff < 0)

        signals: list[Signal] = []
        for ts, is_up in cross_up.items():
            if not is_up:
                continue
            entry = float(close.loc[ts])
            atr_value = atr.loc[ts]
            if pd.isna(atr_value) or atr_value <= 0:
                continue
            stop = entry - p.atr_stop_mult * float(atr_value)
            tgt = entry + p.atr_target_mult * float(atr_value)
            if stop <= 0:
                continue
            signals.append(
                Signal(
                    timestamp=ts.to_pydatetime(),
                    symbol=symbol,
                    direction="LONG",
                    entry_price=entry,
                    stop_loss=stop,
                    target_1=tgt,
                    signal_power="reference",
                    signal_strength=60.0,
                    reasons=("SMA fast crossed above slow",),
                    meta={"strategy": self.name, "side": "long"},
                )
            )

        if not p.long_only:
            for ts, is_dn in cross_dn.items():
                if not is_dn:
                    continue
                entry = float(close.loc[ts])
                atr_value = atr.loc[ts]
                if pd.isna(atr_value) or atr_value <= 0:
                    continue
                stop = entry + p.atr_stop_mult * float(atr_value)
                tgt = entry - p.atr_target_mult * float(atr_value)
                if tgt <= 0:
                    continue
                signals.append(
                    Signal(
                        timestamp=ts.to_pydatetime(),
                        symbol=symbol,
                        direction="SHORT",
                        entry_price=entry,
                        stop_loss=stop,
                        target_1=tgt,
                        signal_power="reference",
                        signal_strength=60.0,
                        reasons=("SMA fast crossed below slow",),
                        meta={"strategy": self.name, "side": "short"},
                    )
                )

        signals.sort(key=lambda s: s.timestamp)
        return signals


__all__ = ["SMACrossover", "SMACrossoverParams"]
