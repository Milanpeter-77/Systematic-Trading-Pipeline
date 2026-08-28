from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

from src.strategies.trend_following.donchian import TrendDonchianStrategy
from src.strategies.trend_following.price_sma import TrendSmaFilterStrategy


class TestTrendSmaFilterStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            TrendSmaFilterStrategy(window=120.0)

    def test_rejects_window_below_minimum(self):
        with pytest.raises(ValueError):
            TrendSmaFilterStrategy(window=2)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = TrendSmaFilterStrategy(window=120)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestTrendDonchianStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            TrendDonchianStrategy(window=120.0)

    def test_rejects_window_below_minimum(self):
        with pytest.raises(ValueError):
            TrendDonchianStrategy(window=2)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = TrendDonchianStrategy(window=120)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)

    def test_channel_mid_is_not_self_defeating(self, synthetic_ohlc_data):
        """
        Regression guard for the self-referential-breakout pitfall this
        strategy deliberately avoids: with a current-bar-inclusive
        channel midpoint, close should cross above/below it regularly,
        not almost never (which a literal "breaks its own rolling max"
        test would produce).
        """
        strategy = TrendDonchianStrategy(window=120)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert (result["target_position"] != 0).sum() > 0
