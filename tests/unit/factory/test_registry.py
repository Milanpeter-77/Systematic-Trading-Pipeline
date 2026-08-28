from __future__ import annotations

import pytest

from src.factory.generator import expand_parameter_grid
from src.factory.registry import STRATEGY_REGISTRY, create_strategy, get_strategy_class


EXPECTED_FAMILIES = {
    "trend",
    "trend_sma",
    "trend_donchian",
    "mean_reversion",
    "mean_reversion_rsi",
    "mean_reversion_return_zscore",
    "momentum",
    "momentum_ewma",
    "momentum_risk_adjusted",
    "volatility",
    "volatility_atr",
    "volatility_range_expansion",
    "stat_arb",
    "stat_arb_rsi",
    "stat_arb_return_zscore",
    "carry",
    "carry_rate_momentum",
    "carry_zscore",
}


def test_registry_has_three_strategies_per_family():
    assert set(STRATEGY_REGISTRY) == EXPECTED_FAMILIES
    assert len(STRATEGY_REGISTRY) == 18


def test_every_registered_class_is_keyed_by_its_own_family_name():
    for family, strategy_class in STRATEGY_REGISTRY.items():
        assert strategy_class.family_name == family


def test_every_registered_class_instantiates_from_its_own_grid():
    for strategy_class in STRATEGY_REGISTRY.values():
        combinations = expand_parameter_grid(strategy_class.parameter_grid)
        strategy_class(**combinations[0])


def test_get_strategy_class_returns_registered_class():
    assert get_strategy_class("momentum_ewma") is STRATEGY_REGISTRY["momentum_ewma"]


def test_get_strategy_class_raises_for_unknown_family():
    with pytest.raises(KeyError):
        get_strategy_class("not_a_real_family")


def test_create_strategy_instantiates_registered_class():
    strategy = create_strategy("trend_sma", {"window": 120})
    assert strategy.family_name == "trend_sma"
