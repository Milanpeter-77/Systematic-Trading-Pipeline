from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

from src.strategies.carry.rate_momentum import CarryRateMomentumStrategy
from src.strategies.carry.real_rate import CarryRealRateStrategy
from src.strategies.carry.risk_adjusted import CarryRiskAdjustedStrategy
from src.strategies.carry.term_slope import CarryTermSlopeStrategy
from src.strategies.carry.zscore import CarryZScoreStrategy


class TestCarryRateMomentumStrategy:
    def test_rejects_non_integer_lookback(self):
        with pytest.raises(TypeError):
            CarryRateMomentumStrategy(lookback=720.0, threshold=0.25)

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            CarryRateMomentumStrategy(lookback=720, threshold=-0.1)

    def test_requires_interest_rate_differential_column(
        self, synthetic_ohlc_data
    ):
        strategy = CarryRateMomentumStrategy(lookback=720, threshold=0.25)
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_ohlc_data)

    def test_generate_positions_shape(
        self, synthetic_ohlc_with_rate_differential
    ):
        strategy = CarryRateMomentumStrategy(lookback=720, threshold=0.25)
        result = strategy.generate_positions(synthetic_ohlc_with_rate_differential)
        assert_valid_positions(result, synthetic_ohlc_with_rate_differential)


class TestCarryZScoreStrategy:
    def test_rejects_non_positive_entry_z(self):
        with pytest.raises(ValueError):
            CarryZScoreStrategy(lookback=720, entry_z=0)

    def test_requires_interest_rate_differential_column(
        self, synthetic_ohlc_data
    ):
        strategy = CarryZScoreStrategy(lookback=720, entry_z=1.0)
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_ohlc_data)

    def test_generate_positions_shape(
        self, synthetic_ohlc_with_rate_differential
    ):
        strategy = CarryZScoreStrategy(lookback=720, entry_z=1.0)
        result = strategy.generate_positions(synthetic_ohlc_with_rate_differential)
        assert_valid_positions(result, synthetic_ohlc_with_rate_differential)


class TestCarryRealRateStrategy:
    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            CarryRealRateStrategy(threshold=-0.1)

    def test_requires_real_rate_differential_column(
        self, synthetic_ohlc_data
    ):
        strategy = CarryRealRateStrategy(threshold=0.25)
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_ohlc_data)

    def test_generate_positions_shape(
        self, synthetic_ohlc_with_real_rate_differential
    ):
        strategy = CarryRealRateStrategy(threshold=0.25)
        result = strategy.generate_positions(
            synthetic_ohlc_with_real_rate_differential
        )
        assert_valid_positions(
            result, synthetic_ohlc_with_real_rate_differential
        )


class TestCarryTermSlopeStrategy:
    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            CarryTermSlopeStrategy(threshold=-0.1)

    def test_requires_term_slope_differential_column(
        self, synthetic_ohlc_data
    ):
        strategy = CarryTermSlopeStrategy(threshold=0.25)
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_ohlc_data)

    def test_generate_positions_shape(
        self, synthetic_ohlc_with_term_slope_differential
    ):
        strategy = CarryTermSlopeStrategy(threshold=0.25)
        result = strategy.generate_positions(
            synthetic_ohlc_with_term_slope_differential
        )
        assert_valid_positions(
            result, synthetic_ohlc_with_term_slope_differential
        )


class TestCarryRiskAdjustedStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            CarryRiskAdjustedStrategy(window=24.0, threshold=0.25)

    def test_requires_interest_rate_differential_column(
        self, synthetic_ohlc_data
    ):
        strategy = CarryRiskAdjustedStrategy(window=24, threshold=0.25)
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_ohlc_data)

    def test_generate_positions_shape(
        self, synthetic_ohlc_with_rate_differential
    ):
        strategy = CarryRiskAdjustedStrategy(window=24, threshold=0.25)
        result = strategy.generate_positions(synthetic_ohlc_with_rate_differential)
        assert_valid_positions(result, synthetic_ohlc_with_rate_differential)
