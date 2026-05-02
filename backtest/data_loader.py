"""Historical OHLCV loader with on-disk Parquet cache.

Uses the synchronous CCXT API so the loader works from CLI / scripts /
notebooks without an event loop. Pages over the exchange's 1500-bar limit
to fetch arbitrarily long histories. Cached data is keyed by
``(exchange, symbol, timeframe)`` and stored as Parquet under
``backtest/cache/`` by default.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Mapping from common timeframe strings to milliseconds. Mirrors CCXT's
# ``parse_timeframe`` but keeps the loader independent of CCXT being
# importable at module-import time (so unit tests can mock it).
_TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


def _timeframe_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_MS:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. "
            f"Supported: {sorted(_TIMEFRAME_MS)}"
        )
    return _TIMEFRAME_MS[timeframe]


def _safe_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _cache_path(
    cache_dir: Path, exchange: str, symbol: str, timeframe: str
) -> Path:
    return (
        cache_dir
        / exchange
        / timeframe
        / f"{_safe_filename(symbol)}.parquet"
    )


def _to_utc_ms(value: datetime | str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - cache corruption is rare
        logger.warning("Failed to read cache %s: %s", path, exc)
        return None
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _normalise_ohlcv(rows: list[list[float]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS).rename_axis("timestamp")
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], unit="ms", utc=True
    )
    frame = frame.set_index("timestamp")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[OHLCV_COLUMNS]


def _build_exchange(exchange_id: str, **options: Any) -> Any:
    """Instantiate a sync CCXT exchange.

    Imported lazily so unit tests can run without CCXT-network access.
    """
    import ccxt  # type: ignore

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange: {exchange_id}")
    klass = getattr(ccxt, exchange_id)
    options.setdefault("enableRateLimit", True)
    return klass(options)


def _fetch_via_ccxt(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None,
    limit: int,
    market_type: str | None,
) -> pd.DataFrame:
    options: dict[str, Any] = {}
    if market_type:
        options.setdefault("options", {})["defaultType"] = market_type
    ex = _build_exchange(exchange_id, **options)

    tf_ms = _timeframe_ms(timeframe)
    cursor = since_ms
    all_rows: list[list[float]] = []
    last_seen_ts: int | None = None
    now_ms = int(time.time() * 1000)
    while True:
        rows = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not rows:
            break
        # Stop if the exchange returned the same final timestamp as the
        # previous page — that means we're at the head of available data.
        last_ts = rows[-1][0]
        if last_seen_ts is not None and last_ts <= last_seen_ts:
            all_rows.extend(rows)
            break
        all_rows.extend(rows)
        last_seen_ts = last_ts
        cursor = last_ts + tf_ms
        if until_ms is not None and last_ts >= until_ms:
            break
        # Stop when we've caught up to the present (no future bars).
        if cursor >= now_ms:
            break
        # Respect rate limit.
        time.sleep(getattr(ex, "rateLimit", 200) / 1000.0)
    return _normalise_ohlcv(all_rows)


def load_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    since: datetime | str | int | None = None,
    until: datetime | str | int | None = None,
    limit: int = 1000,
    market_type: str | None = None,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load OHLCV bars from an exchange with a Parquet cache.

    Args:
        exchange: CCXT exchange id (e.g. ``"binance"``, ``"bybit"``,
            ``"bingx"``, ``"okx"``, ``"mexc"``).
        symbol: Market symbol like ``"BTC/USDT"`` or ``"BTC/USDT:USDT"``
            for swap markets.
        timeframe: One of the strings in :data:`_TIMEFRAME_MS` (e.g. ``"4h"``).
        since: Start of range. ``datetime`` (interpreted as UTC if naive),
            ISO-8601 string, or epoch ms. Required when there is no cache
            and ``use_cache=True``; defaults to 1 year ago otherwise.
        until: End of range (exclusive). ``None`` means "now".
        limit: Max rows per CCXT page (most exchanges cap at 1000-1500).
        market_type: ``"spot"`` / ``"swap"`` / ``"future"`` for exchanges
            that need it (e.g. BingX swap).
        cache_dir: Directory for Parquet cache files. Default
            ``backtest/cache/``.
        use_cache: Read/write the on-disk cache.
        refresh: Ignore the cache and re-fetch the full range.

    Returns:
        DataFrame indexed by UTC ``DatetimeIndex`` with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    since_ms = _to_utc_ms(since)
    until_ms = _to_utc_ms(until)
    if since_ms is None:
        since_ms = int(
            (
                datetime.now(timezone.utc).timestamp() - 365 * 86400
            )
            * 1000
        )

    cached: pd.DataFrame | None = None
    cache_path = _cache_path(cache_dir, exchange, symbol, timeframe)
    if use_cache and not refresh:
        cached = _read_cache(cache_path)

    fetch_from = since_ms
    if cached is not None and not cached.empty:
        last_cached_ms = int(cached.index[-1].value // 1_000_000)
        # Only top up if requested range extends past cache.
        fetch_from = max(since_ms, last_cached_ms + _timeframe_ms(timeframe))

    need_fetch = (
        cached is None
        or refresh
        or (until_ms is None and fetch_from < int(time.time() * 1000))
        or (until_ms is not None and fetch_from < until_ms)
    )

    new_df: pd.DataFrame | None = None
    if need_fetch:
        new_df = _fetch_via_ccxt(
            exchange_id=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since_ms=fetch_from,
            until_ms=until_ms,
            limit=limit,
            market_type=market_type,
        )

    if cached is not None and new_df is not None and not new_df.empty:
        combined = pd.concat([cached, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    elif new_df is not None and not new_df.empty:
        combined = new_df
    elif cached is not None:
        combined = cached
    else:
        combined = pd.DataFrame(columns=OHLCV_COLUMNS).rename_axis(
            "timestamp"
        )
        combined.index = pd.DatetimeIndex(combined.index, tz="UTC")

    if use_cache and not combined.empty:
        _write_cache(combined, cache_path)

    # Slice to requested window for the caller.
    out = combined
    if since_ms is not None:
        start = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        out = out[out.index >= start]
    if until_ms is not None:
        end = pd.Timestamp(until_ms, unit="ms", tz="UTC")
        out = out[out.index < end]
    return out


__all__ = ["load_ohlcv", "DEFAULT_CACHE_DIR", "OHLCV_COLUMNS"]
