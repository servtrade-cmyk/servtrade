"""Bar-by-bar backtest engine.

Conventions (no look-ahead):

* A ``Signal`` generated on bar ``t`` is filled at the *open* of bar ``t+1``.
* Slippage is applied on entry and exit. Long entries pay ``+slippage`` to
  the open price; short entries receive ``-slippage``. Mirrored at exit.
* Stop-loss and take-profit are checked using bar high/low. If both could
  have fired in the same bar, the **stop is assumed to fire first** —
  this is the conservative (pessimistic) convention used by
  Backtrader/vectorbt and avoids inflating win rates.
* Fees are charged in quote currency on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import pandas as pd

from backtest.config import BacktestConfig
from backtest.signal import Signal


@dataclass(frozen=True)
class TradeResult:
    """Outcome of a single simulated trade."""

    signal: Signal
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "target_1", "target_2", "stop_loss", "timeout", "eod"
    qty: float
    pnl_quote: float
    pnl_pct: float
    fees_paid: float
    bars_held: int

    @property
    def is_win(self) -> bool:
        return self.pnl_quote > 0


@dataclass
class BacktestResult:
    """Aggregate result of a backtest run."""

    config: BacktestConfig
    trades: list[TradeResult] = field(default_factory=list)
    equity_curve: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64")
    )
    skipped_signals: int = 0  # signals dropped due to caps / no future data
    bars_processed: int = 0

    def to_trades_frame(self) -> pd.DataFrame:
        """Return trades as a tidy DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "direction",
                    "entry_time",
                    "entry_price",
                    "exit_time",
                    "exit_price",
                    "exit_reason",
                    "qty",
                    "pnl_quote",
                    "pnl_pct",
                    "fees_paid",
                    "bars_held",
                ]
            )
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "symbol": t.signal.symbol,
                    "direction": t.signal.direction,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "qty": t.qty,
                    "pnl_quote": t.pnl_quote,
                    "pnl_pct": t.pnl_pct,
                    "fees_paid": t.fees_paid,
                    "bars_held": t.bars_held,
                }
            )
        return pd.DataFrame(rows)


def _validate_bars(bars: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars DataFrame missing columns: {sorted(missing)}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must have a DatetimeIndex")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars index must be monotonic increasing")
    if bars.index.has_duplicates:
        raise ValueError("bars index must not contain duplicates")


def _simulate_trade(
    signal: Signal,
    bars: pd.DataFrame,
    entry_idx: int,
    config: BacktestConfig,
    capital_at_entry: float,
) -> TradeResult | None:
    """Simulate a single trade starting at bar ``entry_idx``.

    Returns ``None`` if the trade cannot start (insufficient future data).
    """
    if entry_idx >= len(bars):
        return None

    raw_entry = float(bars["open"].iat[entry_idx])
    slip = config.slippage_fraction
    if signal.is_long:
        entry_price = raw_entry * (1.0 + slip)
    else:
        entry_price = raw_entry * (1.0 - slip)

    # Position sizing: risk a fixed fraction of capital based on stop distance.
    stop_distance = abs(entry_price - signal.stop_loss)
    if stop_distance <= 0:
        return None
    risk_amount = capital_at_entry * config.risk_per_trade
    qty = risk_amount / stop_distance
    if qty <= 0:
        return None

    fee_rate = config.fee_fraction
    entry_fee = qty * entry_price * fee_rate

    # Determine the exit by walking forward.
    exit_idx = -1
    exit_price = float("nan")
    exit_reason = "eod"

    max_bars = (
        len(bars)
        if config.bars_to_expiry is None
        else min(len(bars), entry_idx + config.bars_to_expiry + 1)
    )

    for i in range(entry_idx, max_bars):
        high = float(bars["high"].iat[i])
        low = float(bars["low"].iat[i])

        if signal.is_long:
            stop_hit = low <= signal.stop_loss
            t1_hit = high >= signal.target_1
            t2_hit = (
                signal.target_2 is not None and high >= signal.target_2
            )
        else:
            stop_hit = high >= signal.stop_loss
            t1_hit = low <= signal.target_1
            t2_hit = (
                signal.target_2 is not None and low <= signal.target_2
            )

        if stop_hit:
            exit_idx = i
            exit_price = signal.stop_loss
            exit_reason = "stop_loss"
            break
        if t2_hit:
            exit_idx = i
            exit_price = signal.target_2  # type: ignore[assignment]
            exit_reason = "target_2"
            break
        if t1_hit:
            exit_idx = i
            exit_price = signal.target_1
            exit_reason = "target_1"
            break

    if exit_idx == -1:
        # Either timed out or ran off the end of data.
        last_idx = max_bars - 1
        if last_idx < entry_idx:
            return None
        exit_idx = last_idx
        exit_price = float(bars["close"].iat[exit_idx])
        exit_reason = (
            "timeout"
            if config.bars_to_expiry is not None
            and last_idx == entry_idx + config.bars_to_expiry
            else "eod"
        )

    # Slippage on exit (price the trader actually realises).
    if signal.is_long:
        realised_exit = exit_price * (1.0 - slip)
        gross = (realised_exit - entry_price) * qty
    else:
        realised_exit = exit_price * (1.0 + slip)
        gross = (entry_price - realised_exit) * qty

    exit_fee = qty * realised_exit * fee_rate
    fees = entry_fee + exit_fee
    pnl_quote = gross - fees
    pnl_pct = pnl_quote / capital_at_entry if capital_at_entry > 0 else 0.0

    return TradeResult(
        signal=signal,
        entry_time=bars.index[entry_idx].to_pydatetime(),
        entry_price=entry_price,
        exit_time=bars.index[exit_idx].to_pydatetime(),
        exit_price=realised_exit,
        exit_reason=exit_reason,
        qty=qty,
        pnl_quote=pnl_quote,
        pnl_pct=pnl_pct,
        fees_paid=fees,
        bars_held=exit_idx - entry_idx,
    )


def run_backtest(
    bars: pd.DataFrame,
    signals: Iterable[Signal],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a vectorized-ish bar-by-bar backtest.

    Args:
        bars: OHLCV DataFrame indexed by tz-aware or tz-naive ``DatetimeIndex``
            with columns ``open``, ``high``, ``low``, ``close`` (``volume``
            optional). Must be monotonically increasing.
        signals: Iterable of ``Signal`` objects. The engine sorts them by
            timestamp and matches each to the next bar's open.
        config: Engine configuration. Defaults to ``BacktestConfig()``.

    Returns:
        ``BacktestResult`` with the trade log and an equity curve indexed by
        bar timestamp.
    """
    cfg = config or BacktestConfig()
    _validate_bars(bars)

    sigs: Sequence[Signal] = sorted(list(signals), key=lambda s: s.timestamp)
    open_count_per_symbol: dict[str, int] = {}
    open_total = 0
    skipped = 0

    # Equity curve: snapshots after each closed trade plus a baseline at
    # the first bar. We'll resample to per-bar equity at the end.
    equity_points: list[tuple[pd.Timestamp, float]] = [
        (bars.index[0], cfg.initial_capital)
    ]
    capital = cfg.initial_capital
    trades: list[TradeResult] = []

    bar_index = bars.index
    for sig in sigs:
        # Find the next bar strictly after the signal's timestamp.
        ts = pd.Timestamp(sig.timestamp)
        if bar_index.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(bar_index.tz)
        elif bar_index.tz is None and ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        next_pos = bar_index.searchsorted(ts, side="right")
        if next_pos >= len(bars):
            skipped += 1
            continue

        if (
            not cfg.allow_pyramiding
            and open_count_per_symbol.get(sig.symbol, 0) > 0
        ):
            skipped += 1
            continue
        if open_total >= cfg.max_concurrent_trades:
            skipped += 1
            continue

        result = _simulate_trade(sig, bars, int(next_pos), cfg, capital)
        if result is None:
            skipped += 1
            continue

        trades.append(result)
        capital += result.pnl_quote
        equity_points.append((pd.Timestamp(result.exit_time), capital))

        open_count_per_symbol[sig.symbol] = (
            open_count_per_symbol.get(sig.symbol, 0) + 1
        )
        open_total += 1
        # Engine simulates trades sequentially per signal so we
        # immediately decrement on close.
        open_count_per_symbol[sig.symbol] -= 1
        open_total -= 1

    # Build per-bar equity curve using last-known equity (forward fill).
    if equity_points:
        # Deduplicate by timestamp keeping the last value.
        eq_ts, eq_val = zip(*equity_points, strict=True)
        raw = pd.Series(eq_val, index=pd.DatetimeIndex(eq_ts), name="equity")
        raw = raw[~raw.index.duplicated(keep="last")]
        # Reindex to all bars and forward fill.
        equity_curve = raw.reindex(bars.index, method="ffill").ffill()
        equity_curve.name = "equity"
    else:
        equity_curve = pd.Series(
            [cfg.initial_capital] * len(bars),
            index=bars.index,
            name="equity",
        )

    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=equity_curve,
        skipped_signals=skipped,
        bars_processed=len(bars),
    )


__all__ = [
    "BacktestResult",
    "TradeResult",
    "run_backtest",
]
