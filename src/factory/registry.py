from __future__ import annotations

from typing import Type

from src.strategies.base import BaseStrategy
from src.strategies.carry.fx_carry import FXCarryStrategy
from src.strategies.carry.rate_momentum import CarryRateMomentumStrategy
from src.strategies.carry.zscore import CarryZScoreStrategy
from src.strategies.mean_reversion.return_zscore import (
    MeanReversionReturnZScoreStrategy,
)
from src.strategies.mean_reversion.rsi import MeanReversionRsiStrategy
from src.strategies.mean_reversion.zscore import (
    MeanReversionStrategy,
)
from src.strategies.momentum.ewma import MomentumEwmaStrategy
from src.strategies.momentum.risk_adjusted import MomentumRiskAdjustedStrategy
from src.strategies.momentum.time_series import MomentumStrategy
from src.strategies.stat_arb.pairs_zscore import PairsZScoreStrategy
from src.strategies.stat_arb.return_zscore import StatArbReturnZScoreStrategy
from src.strategies.stat_arb.rsi import StatArbRsiStrategy
from src.strategies.trend_following.donchian import TrendDonchianStrategy
from src.strategies.trend_following.ema_crossover import TrendStrategy
from src.strategies.trend_following.price_sma import TrendSmaFilterStrategy
from src.strategies.volatility.atr_breakout import VolatilityAtrBreakoutStrategy
from src.strategies.volatility.breakout import VolatilityBreakoutStrategy
from src.strategies.volatility.range_expansion import (
    VolatilityRangeExpansionStrategy,
)


STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    TrendStrategy.family_name: TrendStrategy,
    TrendSmaFilterStrategy.family_name: TrendSmaFilterStrategy,
    TrendDonchianStrategy.family_name: TrendDonchianStrategy,
    MeanReversionStrategy.family_name: MeanReversionStrategy,
    MeanReversionRsiStrategy.family_name: MeanReversionRsiStrategy,
    MeanReversionReturnZScoreStrategy.family_name: (
        MeanReversionReturnZScoreStrategy
    ),
    MomentumStrategy.family_name: MomentumStrategy,
    MomentumEwmaStrategy.family_name: MomentumEwmaStrategy,
    MomentumRiskAdjustedStrategy.family_name: MomentumRiskAdjustedStrategy,
    VolatilityBreakoutStrategy.family_name: VolatilityBreakoutStrategy,
    VolatilityAtrBreakoutStrategy.family_name: VolatilityAtrBreakoutStrategy,
    VolatilityRangeExpansionStrategy.family_name: (
        VolatilityRangeExpansionStrategy
    ),
    PairsZScoreStrategy.family_name: PairsZScoreStrategy,
    StatArbRsiStrategy.family_name: StatArbRsiStrategy,
    StatArbReturnZScoreStrategy.family_name: StatArbReturnZScoreStrategy,
    FXCarryStrategy.family_name: FXCarryStrategy,
    CarryRateMomentumStrategy.family_name: CarryRateMomentumStrategy,
    CarryZScoreStrategy.family_name: CarryZScoreStrategy,
}


def get_strategy_class(family: str) -> Type[BaseStrategy]:
    """Retrieve a registered strategy family by name."""
    try:
        return STRATEGY_REGISTRY[family]
    except KeyError as error:
        available = sorted(STRATEGY_REGISTRY)
        raise KeyError(
            f"Unknown strategy family '{family}'. "
            f"Available families: {available}"
        ) from error


def create_strategy(family: str, parameters: dict[str, int | float]) -> BaseStrategy:
    """Instantiate one registered strategy."""
    strategy_class = get_strategy_class(family)
    return strategy_class(**parameters)
