from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

from src.strategies.volatility.atr_breakout import VolatilityAtrBreakoutStrategy
from src.strategies.volatility.breakout import VolatilityBreakoutStrategy
from src.strategies.volatility.range_expansion import (
    VolatilityRangeExpansionStrategy,
)


class TestVolatilityAtrBreakoutStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            VolatilityAtrBreakoutStrategy(window=20.0, num_atr=2.0)

    def test_rejects_non_positive_num_atr(self):
        with pytest.raises(ValueError):
            VolatilityAtrBreakoutStrategy(window=20, num_atr=0)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = VolatilityAtrBreakoutStrategy(window=20, num_atr=2.0)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestVolatilityRangeExpansionStrategy:
    def test_rejects_expansion_multiple_below_one(self):
        with pytest.raises(ValueError):
            VolatilityRangeExpansionStrategy(window=20, expansion_multiple=0.5)

    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            VolatilityRangeExpansionStrategy(window=20.0, expansion_multiple=1.5)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = VolatilityRangeExpansionStrategy(
            window=20, expansion_multiple=1.5
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestVolatilityExitNumStdRegression:
    """
    Regression coverage for the exit_num_std parameter added to the
    existing close-to-close Bollinger-band breakout strategy.
    """

    def test_exit_num_std_must_be_below_num_std(self):
        with pytest.raises(ValueError):
            VolatilityBreakoutStrategy(window=20, num_std=2.0, exit_num_std=2.0)

        with pytest.raises(ValueError):
            VolatilityBreakoutStrategy(window=20, num_std=2.0, exit_num_std=-0.1)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = VolatilityBreakoutStrategy(
            window=20, num_std=2.0, exit_num_std=0.5
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)
