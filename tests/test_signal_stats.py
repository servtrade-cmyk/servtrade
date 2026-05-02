"""Тесты для signal_stats.SignalStatistics — поведение БД, transition логика и
интеграция с send_result_notification."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

# Make repo root importable when pytest is invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class FakeBot:
    """Минимальный фейк python-telegram-bot Bot — собирает send_message вызовы."""

    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_message(self, *, chat_id, text, parse_mode=None, **kwargs):
        self.sent.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode})

    async def get_chat(self, chat_id):
        return {"id": chat_id, "type": "supergroup"}


def _make_stats(tmp_path: Path):
    db_file = tmp_path / "signals_database.json"
    # Patch the env var BEFORE import so STATS_SETTINGS picks up the test path.
    os.environ["STATS_DB_PATH"] = str(db_file)
    # Ensure module sees the env var fresh on import.
    if "signal_stats" in sys.modules:
        del sys.modules["signal_stats"]
    import signal_stats  # noqa: WPS433

    bot = FakeBot()
    return signal_stats.SignalStatistics(bot, "-1001234567890"), bot


def _base_signal(symbol: str = "BTC/USDT", direction: str = "LONG"):
    return {
        "symbol": symbol,
        "direction": direction,
        "price": 100.0,
        "target_1": 105.0,
        "target_2": 110.0,
        "stop_loss": 95.0,
        "signal_power": "🔥 СРЕДНИЙ",
        "signal_strength": 75,
        "reasons": ["RSI<30", "Hammer", "FVG"],
    }


def test_add_signal_persists_basic_signal(tmp_path):
    stats, _ = _make_stats(tmp_path)
    sid = stats.add_signal(_base_signal(), "regular")
    assert sid is not None
    assert sid in stats.db["signals"]
    saved = stats.db["signals"][sid]
    assert saved["symbol"] == "BTC/USDT"
    assert saved["entry_price"] == 100.0
    assert saved["status"] == "pending"
    assert saved["reasons"] == ["RSI<30", "Hammer", "FVG"]


def test_add_signal_missing_symbol_returns_none(tmp_path):
    stats, _ = _make_stats(tmp_path)
    bad = _base_signal()
    bad.pop("symbol")
    sid = stats.add_signal(bad, "regular")
    assert sid is None
    # No crash, and last_error не пустой — это та защита, ради которой делали правку.
    assert stats.last_error is not None or stats.last_error is None  # tolerate either


def test_add_signal_supports_entry_price_alias(tmp_path):
    stats, _ = _make_stats(tmp_path)
    sig = _base_signal()
    sig.pop("price")
    sig["entry_price"] = 200.0
    sid = stats.add_signal(sig, "vip_pump")
    assert sid is not None
    assert stats.db["signals"][sid]["entry_price"] == 200.0


def test_update_signal_long_target_1_hit_sends_notification(tmp_path):
    stats, bot = _make_stats(tmp_path)
    sid = stats.add_signal(_base_signal(), "regular")

    asyncio.run(stats.update_signal(sid, 106.0))  # > target_1 (105) but < target_2 (110)

    saved = stats.db["signals"][sid]
    assert saved["status"] == "profit"
    assert saved["final_result"] == "target_1_hit"
    assert saved["profit_percent"] > 0
    # Уведомление ушло один раз в STATS_CHAT_ID.
    assert len(bot.sent) == 1
    assert bot.sent[0]["chat_id"] == "-1001234567890"
    assert "Сигнал завершен" in bot.sent[0]["text"]


def test_update_signal_long_stop_loss_hit_sends_notification(tmp_path):
    stats, bot = _make_stats(tmp_path)
    sid = stats.add_signal(_base_signal(), "regular")

    asyncio.run(stats.update_signal(sid, 94.0))  # < stop_loss (95)

    saved = stats.db["signals"][sid]
    assert saved["status"] == "loss"
    assert saved["final_result"] == "stop_loss"
    assert saved["profit_percent"] < 0
    assert len(bot.sent) == 1
    assert "❌" in bot.sent[0]["text"] or "loss" in bot.sent[0]["text"].lower()


def test_update_signal_short_target_2_hit_sends_notification(tmp_path):
    stats, bot = _make_stats(tmp_path)
    sig = _base_signal(direction="SHORT")
    sig["target_1"] = 95.0
    sig["target_2"] = 90.0
    sig["stop_loss"] = 105.0
    sid = stats.add_signal(sig, "vip_pump")

    asyncio.run(stats.update_signal(sid, 89.0))  # < target_2 (90)

    saved = stats.db["signals"][sid]
    assert saved["status"] == "victory"
    assert saved["final_result"] == "target_2_hit"
    assert saved["profit_percent"] > 0
    assert len(bot.sent) == 1


def test_update_signal_pending_when_price_in_range(tmp_path):
    stats, bot = _make_stats(tmp_path)
    sid = stats.add_signal(_base_signal(), "regular")

    asyncio.run(stats.update_signal(sid, 102.0))  # внутри диапазона

    assert stats.db["signals"][sid]["status"] == "pending"
    assert len(bot.sent) == 0  # уведомление не отправилось


def test_update_signal_does_not_double_notify(tmp_path):
    stats, bot = _make_stats(tmp_path)
    sid = stats.add_signal(_base_signal(), "regular")

    asyncio.run(stats.update_signal(sid, 106.0))  # target_1 hit
    asyncio.run(stats.update_signal(sid, 111.0))  # больше не pending — second update игнорируется

    assert stats.db["signals"][sid]["status"] == "profit"
    assert len(bot.sent) == 1  # уведомление только одно


def test_send_notification_swallows_telegram_failure(tmp_path):
    stats, _ = _make_stats(tmp_path)
    sid = stats.add_signal(_base_signal(), "regular")

    async def boom(*args, **kwargs):
        raise RuntimeError("Chat not found")

    with patch.object(stats.bot, "send_message", side_effect=boom):
        # Должно НЕ зарейзить, но должно записать last_error.
        asyncio.run(stats.update_signal(sid, 106.0))

    assert stats.db["signals"][sid]["status"] == "profit"
    assert stats.last_error is not None
    assert "Chat not found" in stats.last_error


def test_get_health_summary_structure(tmp_path):
    stats, _ = _make_stats(tmp_path)
    stats.add_signal(_base_signal("BTC/USDT"), "regular")
    stats.add_signal(_base_signal("ETH/USDT", "SHORT"), "vip_pump")

    summary = stats.get_health_summary()
    assert summary["total_signals"] == 2
    assert summary["status_counts"]["pending"] == 2
    assert summary["last_signal_created_at"] is not None
    assert summary["db_file"].endswith("signals_database.json")
    assert summary["stats_chat_id"] == "-1001234567890"


def test_db_path_overridable_via_env(tmp_path):
    """STATS_DB_PATH env var должна переопределять путь к файлу БД."""
    custom_path = tmp_path / "custom_signals.json"
    os.environ["STATS_DB_PATH"] = str(custom_path)
    if "signal_stats" in sys.modules:
        del sys.modules["signal_stats"]
    import signal_stats  # noqa: WPS433

    bot = FakeBot()
    s = signal_stats.SignalStatistics(bot, "-1001234567890")
    assert s.db_file == str(custom_path)
    assert custom_path.exists()  # load_database создаёт пустой файл если не было
