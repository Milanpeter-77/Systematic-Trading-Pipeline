from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

from src.strategies.momentum.acceleration import MomentumAccelerationStrategy
from src.strategies.momentum.ewma import MomentumEwmaStrategy
from src.strategies.momentum.macd_histogram import (
    MomentumMacdHistogramStrategy,
)
from src.strategies.momentum.risk_adjusted import MomentumRiskAdjustedStrategy
from src.strategies.momentum.volume_confirmed import (
    MomentumVolumeConfirmedStrategy,
)


class TestMomentumEwmaStrategy:
    def test_rejects_non_integer_span(self):
        with pytest.raises(TypeError):
            MomentumEwmaStrategy(span=24.0, threshold=0.0)

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            MomentumEwmaStrategy(span=24, threshold=-0.01)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MomentumEwmaStrategy(span=24, threshold=0.0)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMomentumRiskAdjustedStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            MomentumRiskAdjustedStrategy(window=24.0, threshold=0.0)

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            MomentumRiskAdjustedStrategy(window=24, threshold=-0.05)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MomentumRiskAdjustedStrategy(window=24, threshold=0.0)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMomentumMacdHistogramStrategy:
    def test_rejects_non_integer_fast_span(self):
        with pytest.raises(TypeError):
            MomentumMacdHistogramStrategy(
                fast_span=12.0, slow_span=26, signal_span=9, threshold=0.0
            )

    def test_rejects_fast_span_not_below_slow_span(self):
        with pytest.raises(ValueError):
            MomentumMacdHistogramStrategy(
                fast_span=26, slow_span=26, signal_span=9, threshold=0.0
            )

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            MomentumMacdHistogramStrategy(
                fast_span=12, slow_span=26, signal_span=9, threshold=-0.001
            )

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MomentumMacdHistogramStrategy(
            fast_span=12, slow_span=26, signal_span=9, threshold=0.0
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMomentumVolumeConfirmedStrategy:
    def test_rejects_non_integer_lookback(self):
        with pytest.raises(TypeError):
            MomentumVolumeConfirmedStrategy(
                lookback=24.0, return_threshold=0.0, volume_ratio=1.5
            )

    def test_rejects_non_positive_volume_ratio(self):
        with pytest.raises(ValueError):
            MomentumVolumeConfirmedStrategy(
                lookback=24, return_threshold=0.0, volume_ratio=0.0
            )

    def test_requires_volume_column(self, synthetic_ohlc_data):
        strategy = MomentumVolumeConfirmedStrategy(
            lookback=24, return_threshold=0.0, volume_ratio=1.5
        )
        data_without_volume = synthetic_ohlc_data.drop(columns=["volume"])
        with pytest.raises(ValueError):
            strategy.generate_positions(data_without_volume)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MomentumVolumeConfirmedStrategy(
            lookback=24, return_threshold=0.0, volume_ratio=1.5
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMomentumAccelerationStrategy:
    def test_rejects_non_integer_lookback(self):
        with pytest.raises(TypeError):
            MomentumAccelerationStrategy(lookback=24.0, threshold=0.0)

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            MomentumAccelerationStrategy(lookback=24, threshold=-0.005)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MomentumAccelerationStrategy(lookback=24, threshold=0.0)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)
