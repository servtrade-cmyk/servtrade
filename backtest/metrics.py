"""Performance metrics for backtest results.

All metrics are computed from the equity curve (per-bar) and the trade log.
Risk-adjusted ratios use simple-return statistics (no log returns) which is
fine for the daily-or-coarser horizons this engine targets. Annualisation is
done via the ``periods_per_year`` argument so callers can reuse the function
for 1d, 4h, 1h etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


@dataclass(frozen=True)
class Metrics:
    """Headline metrics for a backtest run."""

    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    calmar: float
    n_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_rr_realised: float
    expectancy_pct: float
    exposure_pct: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown_pct": self.max_drawdown_pct,
            "calmar": self.calmar,
            "n_trades": self.n_trades,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "avg_rr_realised": self.avg_rr_realised,
            "expectancy_pct": self.expectancy_pct,
            "exposure_pct": self.exposure_pct,
        }


def _max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown as a positive fraction (e.g. 0.23 == 23%)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    return float(-drawdowns.min())


def _annualised_return(
    equity: pd.Series, periods_per_year: float
) -> float:
    """Compound annual growth rate (CAGR) as a fraction."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_periods = len(equity) - 1
    if total_periods <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0
    years = total_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def _sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    mean = returns.mean()
    return float(mean / std * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty:
        return float("inf") if returns.mean() > 0 else 0.0
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    mean = returns.mean()
    return float(mean / dd * np.sqrt(periods_per_year))


def compute_metrics(
    result: BacktestResult,
    periods_per_year: float = 365.0,
) -> Metrics:
    """Compute headline metrics from a ``BacktestResult``.

    Args:
        result: Output of :func:`backtest.engine.run_backtest`.
        periods_per_year: Bars per year. Use ``365`` for daily, ``365*6`` for
            4h, ``365*24`` for 1h, etc. Used for CAGR and risk-adjusted
            ratio annualisation.

    Returns:
        ``Metrics`` snapshot.
    """
    equity = result.equity_curve
    cfg = result.config
    initial = cfg.initial_capital

    if equity.empty:
        final = initial
    else:
        final = float(equity.iloc[-1])

    total_return = (final / initial) - 1.0 if initial > 0 else 0.0
    cagr = _annualised_return(equity, periods_per_year)

    if len(equity) > 1:
        returns = equity.pct_change().dropna()
    else:
        returns = pd.Series(dtype="float64")
    sharpe = _sharpe(returns, periods_per_year)
    sortino = _sortino(returns, periods_per_year)
    max_dd = _max_drawdown(equity)
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    trades = result.trades
    n_trades = len(trades)
    if n_trades == 0:
        return Metrics(
            initial_capital=initial,
            final_equity=final,
            total_return_pct=total_return * 100,
            cagr_pct=cagr * 100,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown_pct=max_dd * 100,
            calmar=calmar,
            n_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            avg_rr_realised=0.0,
            expectancy_pct=0.0,
            exposure_pct=0.0,
        )

    pnls = np.array([t.pnl_quote for t in trades])
    pnls_pct = np.array([t.pnl_pct for t in trades])
    wins = pnls_pct[pnls > 0]
    losses = pnls_pct[pnls < 0]
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(-pnls[pnls < 0].sum())

    win_rate = float(len(wins)) / n_trades
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )

    avg_win_pct = float(wins.mean()) if len(wins) else 0.0
    avg_loss_pct = float(losses.mean()) if len(losses) else 0.0
    avg_rr_realised = (
        avg_win_pct / abs(avg_loss_pct) if avg_loss_pct < 0 else 0.0
    )
    expectancy_pct = float(pnls_pct.mean())

    total_bars = result.bars_processed
    exposure_bars = sum(t.bars_held for t in trades)
    exposure = exposure_bars / total_bars if total_bars else 0.0

    return Metrics(
        initial_capital=initial,
        final_equity=final,
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd * 100,
        calmar=calmar,
        n_trades=n_trades,
        win_rate_pct=win_rate * 100,
        profit_factor=profit_factor,
        avg_win_pct=avg_win_pct * 100,
        avg_loss_pct=avg_loss_pct * 100,
        avg_rr_realised=avg_rr_realised,
        expectancy_pct=expectancy_pct * 100,
        exposure_pct=exposure * 100,
    )


__all__ = ["Metrics", "compute_metrics"]
