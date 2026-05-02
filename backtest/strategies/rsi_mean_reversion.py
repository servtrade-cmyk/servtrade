"""RSI mean-reversion strategy (counter-trend reference)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.signal import Signal
from backtest.strategies.base import Strategy, StrategyParams


@dataclass(frozen=True)
class RSIMeanReversionParams(StrategyParams):
    """Parameters for RSI mean reversion.

    Attributes:
        rsi_period: Lookback for RSI calculation.
        oversold: Buy when RSI crosses *up* through this from below.
        overbought: Sell short when RSI crosses *down* through this from above.
        atr_period: ATR period for risk sizing.
        atr_stop_mult: Stop distance in ATR multiples.
        atr_target_mult: Target distance in ATR multiples.
        long_only: If True, suppress short signals.
    """

    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    atr_period: int = 14
    atr_stop_mult: float = 1.0
    atr_target_mult: float = 2.0
    long_only: bool = False

    def __post_init__(self) -> None:
        if self.rsi_period <= 1:
            raise ValueError("rsi_period must be > 1")
        if not 0 < self.oversold < self.overbought < 100:
            raise ValueError(
                "Need 0 < oversold < overbought < 100"
            )
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.atr_stop_mult <= 0 or self.atr_target_mult <= 0:
            raise ValueError("ATR multipliers must be positive")


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI computed via exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.fillna(50.0)
    return rsi


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


class RSIMeanReversion(Strategy):
    """Buy on RSI cross-up from oversold; short on cross-down from overbought."""

    name = "rsi_mean_reversion"

    def __init__(
        self, params: RSIMeanReversionParams | None = None
    ) -> None:
        self.params = params or RSIMeanReversionParams()

    def generate(
        self, bars: pd.DataFrame, symbol: str
    ) -> list[Signal]:
        p = self.params
        if len(bars) < max(p.rsi_period, p.atr_period) + 2:
            return []

        close = bars["close"]
        rsi = _rsi(close, p.rsi_period)
        atr = _atr(bars, p.atr_period)

        prev_rsi = rsi.shift(1)
        long_trigger = (prev_rsi <= p.oversold) & (rsi > p.oversold)
        short_trigger = (prev_rsi >= p.overbought) & (rsi < p.overbought)

        signals: list[Signal] = []
        for ts, is_long in long_trigger.items():
            if not is_long:
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
                    signal_strength=55.0,
                    reasons=(
                        f"RSI crossed up through {p.oversold:g}",
                    ),
                    meta={"strategy": self.name, "side": "long"},
                )
            )

        if not p.long_only:
            for ts, is_short in short_trigger.items():
                if not is_short:
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
                        signal_strength=55.0,
                        reasons=(
                            f"RSI crossed down through {p.overbought:g}",
                        ),
                        meta={"strategy": self.name, "side": "short"},
                    )
                )

        signals.sort(key=lambda s: s.timestamp)
        return signals


__all__ = ["RSIMeanReversion", "RSIMeanReversionParams"]
