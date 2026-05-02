"""Configuration objects for the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    """Engine-level parameters.

    Attributes:
        initial_capital: Starting equity in quote currency (e.g. USDT).
        risk_per_trade: Fraction of *current* equity risked on each trade.
            With a stop-loss this determines position size: notional =
            risk_amount / |entry - stop| * entry_price.
        taker_fee_bps: Per-side taker fee in basis points (1 bp = 0.01%).
            Default 7.5 bps approximates BingX/Binance taker fees.
        slippage_bps: One-side slippage in basis points applied at entry and
            exit. Default 5 bps is conservative for liquid majors on 4h+ TFs.
        max_concurrent_trades: Cap on simultaneously open positions. Newer
            signals beyond the cap are dropped (counted in the result).
        bars_to_expiry: If neither target nor stop fires within this many
            bars after entry, the trade is closed at the bar's close. ``None``
            disables the timeout (trade stays open until end of data).
        allow_pyramiding: If False (default), repeat signals on the same
            symbol while a trade is open are dropped.
    """

    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    taker_fee_bps: float = 7.5
    slippage_bps: float = 5.0
    max_concurrent_trades: int = 5
    bars_to_expiry: int | None = 200
    allow_pyramiding: bool = False

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.risk_per_trade < 1:
            raise ValueError("risk_per_trade must be in (0, 1)")
        if self.taker_fee_bps < 0:
            raise ValueError("taker_fee_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.max_concurrent_trades < 1:
            raise ValueError("max_concurrent_trades must be >= 1")
        if self.bars_to_expiry is not None and self.bars_to_expiry < 1:
            raise ValueError("bars_to_expiry must be >= 1 or None")

    @property
    def fee_fraction(self) -> float:
        return self.taker_fee_bps / 10_000.0

    @property
    def slippage_fraction(self) -> float:
        return self.slippage_bps / 10_000.0
