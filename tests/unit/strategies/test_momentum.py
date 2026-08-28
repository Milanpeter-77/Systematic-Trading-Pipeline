from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

from src.strategies.momentum.ewma import MomentumEwmaStrategy
from src.strategies.momentum.risk_adjusted import MomentumRiskAdjustedStrategy


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
