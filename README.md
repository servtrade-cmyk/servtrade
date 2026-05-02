# botscan

Multi-exchange crypto signal scanner with Telegram alerts and a historical
backtest harness.

## Components

- `main.py` / `config.py` — live signal scanner. Connects to BingX/Bybit/MEXC
  via CCXT, runs dozens of analyzers (SMC, FVG, fractals, volume profile,
  fibonacci, pump detection, etc.), and pushes alerts to Telegram chats.
- `signal_stats.py` — `SignalStatistics` class that persists every emitted
  signal to a JSON DB and tracks pending/profit/loss outcomes against the
  bot's targets and stops.
- `websocket_manager.py` — BingX websocket data feed.
- `leverage_cache.py` — cached leverage limits per symbol.
- `backtest/` — historical OHLCV loader, signal-driven backtest engine, and
  performance metrics. Lets you grade a strategy (or replay the live bot's
  saved signals) on history.

## Live scanner setup

1. Install deps: `pip install -r requirements.txt`.
2. Copy your Telegram + exchange credentials into a `.env` file (see
   `config.py` for the variable names: `TELEGRAM_TOKEN`,
   `TELEGRAM_CHAT_ID`, `BINGX_API_KEY`, `BINGX_SECRET_KEY`, etc.).
3. Run `python main.py`.

## Backtest harness

The backtest module is signal-driven: it consumes `Signal` objects shaped
like the dicts produced by the live scanner (`entry_price`, `target_1`,
`target_2`, `stop_loss`, `direction`) and simulates trades against
historical OHLCV bars. There is no look-ahead — a signal generated on bar
`t` fills at the open of bar `t+1`, slippage and fees are applied at entry
and exit, and when a stop and a target could fire in the same bar the
engine conservatively assumes the stop hit first.

### CLI

Three subcommands. All output is plain text plus optional JSON/CSV dumps.

#### Fetch and cache history

```bash
python -m backtest fetch \
  --exchange binance --symbol BTC/USDT --timeframe 4h \
  --since 2024-01-01
```

OHLCV is cached as Parquet under `backtest/cache/<exchange>/<timeframe>/`.
Subsequent calls top up incrementally. Pass `--refresh` to force a full
re-fetch, `--no-cache` to bypass the cache entirely.

#### Run a reference strategy

```bash
python -m backtest run \
  --exchange binance --symbol BTC/USDT --timeframe 4h \
  --since 2024-01-01 --strategy sma_cross --capital 10000 \
  --output backtest/results/btc_sma
```

Available strategies (toy reference implementations, intended as a
template — port the live scanner's logic into a real strategy class for
serious backtests):

- `sma_cross` — long on fast SMA crossing above slow SMA, short on the
  inverse. ATR-based stops and targets.
- `rsi_mean_reversion` — long when RSI crosses up through the oversold
  threshold, short on the symmetric overbought crossover.

Engine knobs:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--capital` | 10000 | Starting equity in quote currency |
| `--risk-per-trade` | 0.01 | Fraction of equity risked per trade |
| `--fee-bps` | 7.5 | Per-side taker fee in basis points |
| `--slippage-bps` | 5 | Per-side slippage in basis points |
| `--max-concurrent` | 5 | Max simultaneously open trades |
| `--bars-to-expiry` | 200 | Force-close after N bars (`0` to disable) |
| `--allow-pyramiding` | false | Allow stacking trades on the same symbol |

#### Replay live-bot signals

If you have `signals_database.json` written by `SignalStatistics`,
evaluate it on fresh history without re-running the monolithic scanner:

```bash
python -m backtest replay \
  --exchange bingx --symbol BTC/USDT --timeframe 15m --market-type swap \
  --db /tmp/signals_database.json --signal-type regular --min-strength 70
```

Filters: `--symbol`, `--direction LONG|SHORT`, `--signal-type`,
`--min-strength`.

### Metrics

`compute_metrics(result)` returns total return, CAGR, Sharpe, Sortino,
max drawdown, Calmar, win rate, profit factor, average win/loss, realised
R:R, expectancy, and time-in-market. Use these to compare strategy
variants — a high win rate with a low profit factor or high drawdown is
not a good system.

### Library API

```python
from datetime import datetime, timezone
from backtest import (
    BacktestConfig, load_ohlcv, run_backtest, compute_metrics,
)
from backtest.strategies import SMACrossover, SMACrossoverParams

bars = load_ohlcv(
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="4h",
    since=datetime(2024, 1, 1, tzinfo=timezone.utc),
)

strategy = SMACrossover(SMACrossoverParams(fast=20, slow=50))
signals = strategy.generate(bars, "BTC/USDT")

result = run_backtest(bars, signals, BacktestConfig(initial_capital=10_000))
metrics = compute_metrics(result, periods_per_year=365 * 6)  # 4h
print(metrics.as_dict())
```

## Tests

```bash
pip install pytest pyarrow
pytest tests/
```

The test suite stubs CCXT and uses synthetic OHLCV, so it runs offline.

## Honest expectations

A trading system is judged by Sharpe, drawdown, and profit factor — not
by win rate alone. A strategy with a 30% win rate and 1:5 R:R can be
profitable; a strategy with an 80% win rate and 1:0.2 R:R is almost
guaranteed to lose money on a single bad bar. Use the backtest to grade
the live scanner's signals quantitatively before risking real capital,
and keep paper trading for at least a month after backtest validation.
