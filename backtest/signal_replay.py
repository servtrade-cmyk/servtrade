"""Replay signals from ``signal_stats.SignalStatistics`` JSON files.

Lets the user evaluate the *actual* signals their live scanner produced
against fresh historical data, rather than re-running the whole monolithic
scanner on history. Pair with ``run_backtest`` to get win-rate / R:R /
Sharpe on the live-emitted signals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from backtest.signal import Signal


def load_signals_from_db(
    db_path: str | Path,
    *,
    symbol: str | None = None,
    direction: str | None = None,
    signal_type: str | None = None,
    min_strength: float | None = None,
) -> list[Signal]:
    """Load signals from a ``signals_database.json`` file.

    Args:
        db_path: Path to the JSON DB written by ``SignalStatistics``.
        symbol: Optional symbol filter (exact match, e.g. ``"BTC/USDT"``).
        direction: Optional ``"LONG"`` / ``"SHORT"`` filter (case-insensitive).
        signal_type: Optional filter on the live-bot's signal type
            (``"regular"``, ``"pump"``, ``"accumulation"``, ``"vip_pump"``,
            ``"discovery"``).
        min_strength: Optional floor on ``signal_strength``.

    Returns:
        List of ``Signal`` objects. Signals missing required fields
        (entry/target/stop) are skipped.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Signal DB not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        db = json.load(f)

    raw = db.get("signals", {}) if isinstance(db, dict) else {}
    out: list[Signal] = []
    direction_filter = direction.upper() if direction else None

    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        if symbol and entry.get("symbol") != symbol:
            continue
        if signal_type and entry.get("type") != signal_type:
            continue
        if min_strength is not None and float(
            entry.get("signal_strength", 0) or 0
        ) < min_strength:
            continue

        entry_dir = str(entry.get("direction", "")).upper()
        # Live bot stores "LONG" / "SHORT" possibly with prefixes (emoji etc).
        normalised_dir = "LONG" if "LONG" in entry_dir else (
            "SHORT" if "SHORT" in entry_dir else None
        )
        if normalised_dir is None:
            continue
        if direction_filter and normalised_dir != direction_filter:
            continue

        if not entry.get("target_1") or not entry.get("stop_loss"):
            continue

        normalised = {
            **entry,
            "direction": normalised_dir,
        }
        try:
            out.append(Signal.from_dict(normalised))
        except (ValueError, KeyError, TypeError):
            # Skip malformed records rather than failing the whole replay.
            continue

    out.sort(key=lambda s: s.timestamp)
    return out


def signals_to_dicts(signals: Iterable[Signal]) -> list[dict]:
    """Serialize signals back to the live-bot dict shape (round-trip helper)."""
    out: list[dict] = []
    for sig in signals:
        out.append(
            {
                "symbol": sig.symbol,
                "direction": sig.direction,
                "price": sig.entry_price,
                "entry_price": sig.entry_price,
                "target_1": sig.target_1,
                "target_2": sig.target_2,
                "stop_loss": sig.stop_loss,
                "signal_power": sig.signal_power,
                "signal_strength": sig.signal_strength,
                "reasons": list(sig.reasons),
                "timestamp": sig.timestamp.isoformat(),
            }
        )
    return out


__all__ = ["load_signals_from_db", "signals_to_dicts"]
