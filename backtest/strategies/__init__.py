"""Reference strategies that emit signals consumable by the backtest engine.

Strategies are intentionally simple — they exist to prove the engine works
end-to-end, not to be production trading systems. Use them as a template
for porting the live scanner's signal logic into something testable.
"""

from backtest.strategies.base import Strategy, StrategyParams
from backtest.strategies.rsi_mean_reversion import (
    RSIMeanReversion,
    RSIMeanReversionParams,
)
from backtest.strategies.sma_cross import SMACrossover, SMACrossoverParams

__all__ = [
    "RSIMeanReversion",
    "RSIMeanReversionParams",
    "SMACrossover",
    "SMACrossoverParams",
    "Strategy",
    "StrategyParams",
]
