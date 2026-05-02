"""Strategy base class.

A ``Strategy`` consumes an OHLCV DataFrame and returns a list of ``Signal``
objects. The base class is deliberately thin — strategies are free to be
stateful or stateless, vectorised or iterative, as long as ``generate`` is
deterministic for a given input frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from backtest.signal import Signal


@dataclass(frozen=True)
class StrategyParams:
    """Marker base class for strategy parameter dataclasses."""


class Strategy(ABC):
    """Abstract strategy producing signals from an OHLCV frame."""

    name: str = "strategy"

    @abstractmethod
    def generate(self, bars: pd.DataFrame, symbol: str) -> list[Signal]:
        """Generate signals from ``bars``.

        Args:
            bars: OHLCV DataFrame with a ``DatetimeIndex`` and at least the
                columns ``open``, ``high``, ``low``, ``close``.
            symbol: Market symbol (passed through to ``Signal.symbol``).

        Returns:
            List of ``Signal`` objects sorted by timestamp.
        """


__all__ = ["Strategy", "StrategyParams"]
