from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

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


class TestVolatilityVixRegimeStrategy:
    def test_rejects_non_integer_realized_vol_window(self):
        with pytest.raises(TypeError):
            VolatilityVixRegimeStrategy(
                realized_vol_window=24.0, threshold=5.0
            )

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            VolatilityVixRegimeStrategy(realized_vol_window=24, threshold=-1.0)

    def test_requires_vix_level_column(self, synthetic_ohlc_data):
        strategy = VolatilityVixRegimeStrategy(
            realized_vol_window=24, threshold=5.0
        )
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_ohlc_data)

    def test_generate_positions_shape(self, synthetic_ohlc_with_vix):
        strategy = VolatilityVixRegimeStrategy(
            realized_vol_window=24, threshold=5.0
        )
        result = strategy.generate_positions(synthetic_ohlc_with_vix)
        assert_valid_positions(result, synthetic_ohlc_with_vix)


class TestVolatilityMeanReversionStrategy:
    def test_rejects_non_integer_short_window(self):
        with pytest.raises(TypeError):
            VolatilityMeanReversionStrategy(
                short_window=24.0, entry_z=1.5, exit_z=0.5
            )

    def test_rejects_exit_z_not_below_entry_z(self):
        with pytest.raises(ValueError):
            VolatilityMeanReversionStrategy(
                short_window=24, entry_z=1.5, exit_z=1.5
            )

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = VolatilityMeanReversionStrategy(
            short_window=24, entry_z=1.5, exit_z=0.5
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestVolatilitySqueezeStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            VolatilitySqueezeStrategy(
                window=20.0, num_std=2.0, atr_multiple=2.0
            )

    def test_rejects_non_positive_atr_multiple(self):
        with pytest.raises(ValueError):
            VolatilitySqueezeStrategy(window=20, num_std=2.0, atr_multiple=0)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = VolatilitySqueezeStrategy(
            window=20, num_std=2.0, atr_multiple=2.0
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)
