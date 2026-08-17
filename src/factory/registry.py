from __future__ import annotations

from typing import Type

from src.strategies.base import BaseStrategy
from src.strategies.mean_reversion.zscore import (
    MeanReversionStrategy,
)
from src.strategies.trend_following.ema_crossover import TrendStrategy


STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    TrendStrategy.family_name: TrendStrategy,
    MeanReversionStrategy.family_name: MeanReversionStrategy,
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