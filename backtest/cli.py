"""Command-line interface for the backtest module.

Examples::

    # Fetch + cache 4h BTC/USDT history from Binance.
    python -m backtest fetch \\
        --exchange binance --symbol BTC/USDT --timeframe 4h \\
        --since 2024-01-01

    # Backtest the SMA-cross reference strategy.
    python -m backtest run \\
        --exchange binance --symbol BTC/USDT --timeframe 4h \\
        --since 2024-01-01 --strategy sma_cross --capital 10000

    # Replay signals saved by the live bot against fresh OHLCV.
    python -m backtest replay \\
        --exchange bingx --symbol BTC/USDT --timeframe 15m \\
        --db /tmp/signals_database.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest.config import BacktestConfig
from backtest.data_loader import DEFAULT_CACHE_DIR, load_ohlcv
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.signal_replay import load_signals_from_db
from backtest.strategies import (
    RSIMeanReversion,
    RSIMeanReversionParams,
    SMACrossover,
    SMACrossoverParams,
    Strategy,
)


_TF_PERIODS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "3m": 365 * 24 * 20,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "2h": 365 * 12,
    "4h": 365 * 6,
    "6h": 365 * 4,
    "8h": 365 * 3,
    "12h": 365 * 2,
    "1d": 365,
    "3d": 365 / 3,
    "1w": 52,
}


def _periods_per_year(timeframe: str) -> float:
    return _TF_PERIODS_PER_YEAR.get(timeframe, 365.0)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _build_strategy(name: str) -> Strategy:
    if name == "sma_cross":
        return SMACrossover(SMACrossoverParams())
    if name == "rsi_mean_reversion":
        return RSIMeanReversion(RSIMeanReversionParams())
    raise SystemExit(
        f"Unknown strategy {name!r}. Choices: sma_cross, rsi_mean_reversion"
    )


def _format_metrics(metrics_dict: dict) -> str:
    rows = []
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            if v == float("inf"):
                rows.append(f"  {k:<22s} inf")
            else:
                rows.append(f"  {k:<22s} {v:>12.4f}")
        else:
            rows.append(f"  {k:<22s} {v:>12}")
    return "\n".join(rows)


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument(
        "--since", help="ISO-8601 start (UTC). Default: 1y ago."
    )
    parser.add_argument("--until", help="ISO-8601 end (UTC), exclusive.")
    parser.add_argument(
        "--market-type",
        choices=["spot", "swap", "future"],
        default=None,
        help="CCXT defaultType override.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Parquet cache directory.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk cache.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a re-fetch even if cache exists.",
    )


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--fee-bps", type=float, default=7.5)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--bars-to-expiry", type=int, default=200)
    parser.add_argument("--allow-pyramiding", action="store_true")


def _build_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        taker_fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        max_concurrent_trades=args.max_concurrent,
        bars_to_expiry=args.bars_to_expiry,
        allow_pyramiding=args.allow_pyramiding,
    )


def _cmd_fetch(args: argparse.Namespace) -> int:
    bars = load_ohlcv(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=_parse_iso(args.since),
        until=_parse_iso(args.until),
        market_type=args.market_type,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        refresh=args.refresh,
    )
    if bars.empty:
        print("No bars returned.", file=sys.stderr)
        return 1
    print(
        f"Fetched {len(bars)} bars for {args.symbol} {args.timeframe} "
        f"from {bars.index[0]} to {bars.index[-1]}"
    )
    return 0


def _save_results(
    out_dir: Path,
    bars: pd.DataFrame,
    result_dict: dict,
    trades_df: pd.DataFrame,
    equity: pd.Series,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    equity.to_csv(out_dir / "equity.csv")
    bars.head(1).to_csv(out_dir / "first_bar.csv")  # sanity sample
    print(f"\nResults written to {out_dir}/")


def _cmd_run(args: argparse.Namespace) -> int:
    bars = load_ohlcv(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=_parse_iso(args.since),
        until=_parse_iso(args.until),
        market_type=args.market_type,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        refresh=args.refresh,
    )
    if bars.empty:
        print("No bars available.", file=sys.stderr)
        return 1

    strategy = _build_strategy(args.strategy)
    signals = strategy.generate(bars, args.symbol)
    print(f"Generated {len(signals)} signals from {strategy.name}")

    cfg = _build_config(args)
    result = run_backtest(bars, signals, cfg)
    metrics = compute_metrics(
        result, periods_per_year=_periods_per_year(args.timeframe)
    )
    print(
        f"\n=== {strategy.name} on {args.symbol} {args.timeframe} ==="
    )
    print(_format_metrics(metrics.as_dict()))
    print(f"  signals_skipped       {result.skipped_signals:>12d}")

    if args.output:
        _save_results(
            Path(args.output),
            bars,
            metrics.as_dict(),
            result.to_trades_frame(),
            result.equity_curve,
        )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    bars = load_ohlcv(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=_parse_iso(args.since),
        until=_parse_iso(args.until),
        market_type=args.market_type,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        refresh=args.refresh,
    )
    if bars.empty:
        print("No bars available.", file=sys.stderr)
        return 1

    signals = load_signals_from_db(
        args.db,
        symbol=args.symbol,
        direction=args.direction,
        signal_type=args.signal_type,
        min_strength=args.min_strength,
    )
    print(
        f"Loaded {len(signals)} signals from {args.db} matching filters"
    )
    if not signals:
        return 1

    cfg = _build_config(args)
    result = run_backtest(bars, signals, cfg)
    metrics = compute_metrics(
        result, periods_per_year=_periods_per_year(args.timeframe)
    )
    print(f"\n=== Replay {args.symbol} {args.timeframe} ===")
    print(_format_metrics(metrics.as_dict()))
    print(f"  signals_skipped       {result.skipped_signals:>12d}")

    if args.output:
        _save_results(
            Path(args.output),
            bars,
            metrics.as_dict(),
            result.to_trades_frame(),
            result.equity_curve,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest",
        description="Botscan backtest CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="Fetch + cache OHLCV history.")
    _add_common_data_args(fetch)
    fetch.set_defaults(func=_cmd_fetch)

    run = sub.add_parser(
        "run", help="Run a reference strategy on historical OHLCV."
    )
    _add_common_data_args(run)
    _add_engine_args(run)
    run.add_argument(
        "--strategy",
        choices=["sma_cross", "rsi_mean_reversion"],
        default="sma_cross",
    )
    run.add_argument(
        "--output",
        help="Directory to write metrics.json/trades.csv/equity.csv.",
    )
    run.set_defaults(func=_cmd_run)

    replay = sub.add_parser(
        "replay",
        help="Replay signals from a SignalStatistics JSON DB.",
    )
    _add_common_data_args(replay)
    _add_engine_args(replay)
    replay.add_argument(
        "--db",
        required=True,
        help="Path to signals_database.json.",
    )
    replay.add_argument("--direction", choices=["LONG", "SHORT"])
    replay.add_argument(
        "--signal-type",
        help="Filter by live-bot signal type "
        "(regular/pump/accumulation/vip_pump/discovery).",
    )
    replay.add_argument(
        "--min-strength", type=float, help="Min signal_strength filter."
    )
    replay.add_argument("--output")
    replay.set_defaults(func=_cmd_replay)

    parser.add_argument(
        "-v", "--verbose", action="count", default=0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
