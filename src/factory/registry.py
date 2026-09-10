from __future__ import annotations

from typing import Type

from src.strategies.base import BaseStrategy
from src.strategies.carry.fx_carry import FXCarryStrategy
from src.strategies.carry.rate_momentum import CarryRateMomentumStrategy
from src.strategies.carry.real_rate import CarryRealRateStrategy
from src.strategies.carry.risk_adjusted import CarryRiskAdjustedStrategy
from src.strategies.carry.term_slope import CarryTermSlopeStrategy
from src.strategies.carry.zscore import CarryZScoreStrategy
from src.strategies.mean_reversion.bollinger_pct_b import (
    MeanReversionBollingerPctBStrategy,
)
from src.strategies.mean_reversion.keltner import MeanReversionKeltnerStrategy
from src.strategies.mean_reversion.return_zscore import (
    MeanReversionReturnZScoreStrategy,
)
from src.strategies.mean_reversion.rsi import MeanReversionRsiStrategy
from src.strategies.mean_reversion.vwap_deviation import (
    MeanReversionVwapDeviationStrategy,
)
from src.strategies.mean_reversion.zscore import (
    MeanReversionStrategy,
)
from src.strategies.momentum.acceleration import MomentumAccelerationStrategy
from src.strategies.momentum.ewma import MomentumEwmaStrategy
from src.strategies.momentum.macd_histogram import (
    MomentumMacdHistogramStrategy,
)
from src.strategies.momentum.risk_adjusted import MomentumRiskAdjustedStrategy
from src.strategies.momentum.time_series import MomentumStrategy
from src.strategies.momentum.volume_confirmed import (
    MomentumVolumeConfirmedStrategy,
)
from src.strategies.stat_arb.correlation_breakdown import (
    StatArbCorrelationBreakdownStrategy,
)
from src.strategies.stat_arb.macd_spread import StatArbMacdStrategy
from src.strategies.stat_arb.pairs_zscore import PairsZScoreStrategy
from src.strategies.stat_arb.return_zscore import StatArbReturnZScoreStrategy
from src.strategies.stat_arb.rolling_hedge_zscore import (
    StatArbRollingHedgeZScoreStrategy,
)
from src.strategies.stat_arb.rsi import StatArbRsiStrategy
from src.strategies.trend_following.donchian import TrendDonchianStrategy
from src.strategies.trend_following.ema_crossover import TrendStrategy
from src.strategies.trend_following.price_sma import TrendSmaFilterStrategy
from src.strategies.trend_following.trend_adx import TrendAdxStrategy
from src.strategies.trend_following.trend_macd import TrendMacdStrategy
from src.strategies.trend_following.trend_psar import TrendPsarStrategy
from src.strategies.volatility.atr_breakout import VolatilityAtrBreakoutStrategy
from src.strategies.volatility.breakout import VolatilityBreakoutStrategy
from src.strategies.volatility.range_expansion import (
    VolatilityRangeExpansionStrategy,
)
from src.strategies.volatility.squeeze import VolatilitySqueezeStrategy
from src.strategies.volatility.vix_regime import VolatilityVixRegimeStrategy
from src.strategies.volatility.vol_mean_reversion import (
    VolatilityMeanReversionStrategy,
)


STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    TrendStrategy.family_name: TrendStrategy,
    TrendSmaFilterStrategy.family_name: TrendSmaFilterStrategy,
    TrendDonchianStrategy.family_name: TrendDonchianStrategy,
    TrendMacdStrategy.family_name: TrendMacdStrategy,
    TrendAdxStrategy.family_name: TrendAdxStrategy,
    TrendPsarStrategy.family_name: TrendPsarStrategy,
    MeanReversionStrategy.family_name: MeanReversionStrategy,
    MeanReversionRsiStrategy.family_name: MeanReversionRsiStrategy,
    MeanReversionReturnZScoreStrategy.family_name: (
        MeanReversionReturnZScoreStrategy
    ),
    MeanReversionBollingerPctBStrategy.family_name: (
        MeanReversionBollingerPctBStrategy
    ),
    MeanReversionKeltnerStrategy.family_name: MeanReversionKeltnerStrategy,
    MeanReversionVwapDeviationStrategy.family_name: (
        MeanReversionVwapDeviationStrategy
    ),
    MomentumStrategy.family_name: MomentumStrategy,
    MomentumEwmaStrategy.family_name: MomentumEwmaStrategy,
    MomentumRiskAdjustedStrategy.family_name: MomentumRiskAdjustedStrategy,
    MomentumMacdHistogramStrategy.family_name: MomentumMacdHistogramStrategy,
    MomentumVolumeConfirmedStrategy.family_name: (
        MomentumVolumeConfirmedStrategy
    ),
    MomentumAccelerationStrategy.family_name: MomentumAccelerationStrategy,
    VolatilityBreakoutStrategy.family_name: VolatilityBreakoutStrategy,
    VolatilityAtrBreakoutStrategy.family_name: VolatilityAtrBreakoutStrategy,
    VolatilityRangeExpansionStrategy.family_name: (
        VolatilityRangeExpansionStrategy
    ),
    VolatilityVixRegimeStrategy.family_name: VolatilityVixRegimeStrategy,
    VolatilityMeanReversionStrategy.family_name: (
        VolatilityMeanReversionStrategy
    ),
    VolatilitySqueezeStrategy.family_name: VolatilitySqueezeStrategy,
    PairsZScoreStrategy.family_name: PairsZScoreStrategy,
    StatArbRsiStrategy.family_name: StatArbRsiStrategy,
    StatArbReturnZScoreStrategy.family_name: StatArbReturnZScoreStrategy,
    StatArbMacdStrategy.family_name: StatArbMacdStrategy,
    StatArbRollingHedgeZScoreStrategy.family_name: (
        StatArbRollingHedgeZScoreStrategy
    ),
    StatArbCorrelationBreakdownStrategy.family_name: (
        StatArbCorrelationBreakdownStrategy
    ),
    FXCarryStrategy.family_name: FXCarryStrategy,
    CarryRateMomentumStrategy.family_name: CarryRateMomentumStrategy,
    CarryZScoreStrategy.family_name: CarryZScoreStrategy,
    CarryRealRateStrategy.family_name: CarryRealRateStrategy,
    CarryTermSlopeStrategy.family_name: CarryTermSlopeStrategy,
    CarryRiskAdjustedStrategy.family_name: CarryRiskAdjustedStrategy,
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
