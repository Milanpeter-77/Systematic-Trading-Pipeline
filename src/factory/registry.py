from __future__ import annotations

from typing import Type

from src.strategies.base import BaseStrategy
from src.strategies.carry.fx_carry import FXCarryStrategy
from src.strategies.mean_reversion.zscore import (
    MeanReversionStrategy,
)
from src.strategies.momentum.time_series import MomentumStrategy
from src.strategies.stat_arb.pairs_zscore import PairsZScoreStrategy
from src.strategies.trend_following.ema_crossover import TrendStrategy
from src.strategies.volatility.breakout import VolatilityBreakoutStrategy


STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    TrendStrategy.family_name: TrendStrategy,
    MeanReversionStrategy.family_name: MeanReversionStrategy,
    MomentumStrategy.family_name: MomentumStrategy,
    VolatilityBreakoutStrategy.family_name: VolatilityBreakoutStrategy,
    PairsZScoreStrategy.family_name: PairsZScoreStrategy,
    FXCarryStrategy.family_name: FXCarryStrategy,
}


def get_strategy_class(
    family: str,
) -> Type[BaseStrategy]:
    """
    Retrieve a registered strategy family by name.
    """
    try:
        return STRATEGY_REGISTRY[family]
    except KeyError as error:
        available = sorted(STRATEGY_REGISTRY)

        raise KeyError(
            f"Unknown strategy family '{family}'. "
            f"Available families: {available}"
        ) from error


def create_strategy(
    family: str,
    parameters: dict[str, int | float],
) -> BaseStrategy:
    """
    Instantiate one registered strategy.
    """
    strategy_class = get_strategy_class(family)

    return strategy_class(**parameters)