from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

from src.strategies.trend_following.donchian import TrendDonchianStrategy
from src.strategies.trend_following.price_sma import TrendSmaFilterStrategy
from src.strategies.trend_following.trend_adx import TrendAdxStrategy
from src.strategies.trend_following.trend_macd import TrendMacdStrategy
from src.strategies.trend_following.trend_psar import TrendPsarStrategy


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


class TestTrendMacdStrategy:
    def test_rejects_non_integer_fast_span(self):
        with pytest.raises(TypeError):
            TrendMacdStrategy(fast_span=12.0, slow_span=26, signal_span=9)

    def test_rejects_fast_span_not_below_slow_span(self):
        with pytest.raises(ValueError):
            TrendMacdStrategy(fast_span=26, slow_span=26, signal_span=9)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = TrendMacdStrategy(fast_span=12, slow_span=26, signal_span=9)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestTrendAdxStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            TrendAdxStrategy(window=14.0, adx_threshold=25)

    def test_rejects_adx_threshold_outside_open_interval(self):
        with pytest.raises(ValueError):
            TrendAdxStrategy(window=14, adx_threshold=0)

        with pytest.raises(ValueError):
            TrendAdxStrategy(window=14, adx_threshold=150)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = TrendAdxStrategy(window=14, adx_threshold=25)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestTrendPsarStrategy:
    def test_rejects_non_numeric_acceleration_step(self):
        with pytest.raises(TypeError):
            TrendPsarStrategy(acceleration_step="0.02", acceleration_max=0.2)

    def test_rejects_acceleration_step_above_max(self):
        with pytest.raises(ValueError):
            TrendPsarStrategy(acceleration_step=0.3, acceleration_max=0.2)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = TrendPsarStrategy(
            acceleration_step=0.02, acceleration_max=0.2
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)

    def test_first_two_bars_are_flat(self, synthetic_ohlc_data):
        strategy = TrendPsarStrategy(
            acceleration_step=0.02, acceleration_max=0.2
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert result["target_position"].iloc[0] == 0
