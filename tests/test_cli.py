"""End-to-end smoke tests for the backtest CLI.

The exchange call is patched to a stub so the tests run offline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest import cli, data_loader


class _FakeEx:
    rateLimit = 0

    def __init__(self):
        base = int(
            datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
        )
        hour = 3_600_000
        # 600 hourly bars with mild drift + noise so strategies can fire.
        rows = []
        price = 100.0
        for i in range(600):
            ts = base + i * hour
            move = 0.5 if (i // 50) % 2 == 0 else -0.5
            o = price
            c = price + move
            h = max(o, c) + 0.3
            low = min(o, c) - 0.3
            rows.append([ts, o, h, low, c, 10])
            price = c
        self.rows = rows

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        out = [r for r in self.rows if r[0] >= since]
        return out[:limit]


@pytest.fixture
def stub_ccxt(monkeypatch):
    fake = _FakeEx()
    monkeypatch.setattr(data_loader, "_build_exchange", lambda *a, **k: fake)
    return fake


def test_cli_fetch_command(stub_ccxt, tmp_path, capsys):
    rc = cli.main(
        [
            "fetch",
            "--exchange",
            "stub",
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--since",
            "2024-01-01T00:00:00",
            "--until",
            "2024-01-10T00:00:00",
            "--cache-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Fetched" in out


def test_cli_run_command(stub_ccxt, tmp_path, capsys):
    out_dir = tmp_path / "result"
    rc = cli.main(
        [
            "run",
            "--exchange",
            "stub",
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--since",
            "2024-01-01T00:00:00",
            "--until",
            "2024-01-25T00:00:00",
            "--strategy",
            "sma_cross",
            "--cache-dir",
            str(tmp_path),
            "--capital",
            "10000",
            "--output",
            str(out_dir),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "sma_cross" in out
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "equity.csv").exists()


def test_cli_run_rsi(stub_ccxt, tmp_path, capsys):
    rc = cli.main(
        [
            "run",
            "--exchange",
            "stub",
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--since",
            "2024-01-01T00:00:00",
            "--until",
            "2024-01-25T00:00:00",
            "--strategy",
            "rsi_mean_reversion",
            "--cache-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "rsi_mean_reversion" in out


def test_cli_replay_command(stub_ccxt, tmp_path, capsys):
    db = tmp_path / "signals.json"
    db.write_text(
        '{"signals": {"a": {'
        '"symbol": "BTC/USDT", "direction": "LONG", "price": 100.0, '
        '"target_1": 110.0, "stop_loss": 95.0, '
        '"signal_strength": 80, "type": "regular", '
        '"created_at": "2024-01-02T00:00:00"}}}',
        encoding="utf-8",
    )
    rc = cli.main(
        [
            "replay",
            "--exchange",
            "stub",
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--since",
            "2024-01-01T00:00:00",
            "--until",
            "2024-01-15T00:00:00",
            "--db",
            str(db),
            "--cache-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Replay" in out
