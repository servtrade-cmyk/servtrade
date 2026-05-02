"""Tests for BacktestConfig."""

import pytest

from backtest.config import BacktestConfig


def test_defaults_are_reasonable():
    cfg = BacktestConfig()
    assert cfg.initial_capital == 10_000.0
    assert cfg.fee_fraction == pytest.approx(0.00075)
    assert cfg.slippage_fraction == pytest.approx(0.0005)


@pytest.mark.parametrize(
    "field,value,err",
    [
        ("initial_capital", 0, "initial_capital"),
        ("initial_capital", -1, "initial_capital"),
        ("risk_per_trade", 0, "risk_per_trade"),
        ("risk_per_trade", 1.0, "risk_per_trade"),
        ("taker_fee_bps", -1, "taker_fee_bps"),
        ("slippage_bps", -1, "slippage_bps"),
        ("max_concurrent_trades", 0, "max_concurrent"),
        ("bars_to_expiry", 0, "bars_to_expiry"),
    ],
)
def test_invalid_values(field, value, err):
    with pytest.raises(ValueError, match=err):
        BacktestConfig(**{field: value})  # type: ignore[arg-type]


def test_bars_to_expiry_none_allowed():
    cfg = BacktestConfig(bars_to_expiry=None)
    assert cfg.bars_to_expiry is None
