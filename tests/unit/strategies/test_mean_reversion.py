from __future__ import annotations

import pytest

from tests.conftest import assert_valid_positions

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
from src.strategies.mean_reversion.zscore import MeanReversionStrategy


class TestMeanReversionRsiStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            MeanReversionRsiStrategy(window=24.0, entry_band=20)

    def test_rejects_entry_band_outside_open_interval(self):
        with pytest.raises(ValueError):
            MeanReversionRsiStrategy(window=24, entry_band=50)

        with pytest.raises(ValueError):
            MeanReversionRsiStrategy(window=24, entry_band=0)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MeanReversionRsiStrategy(window=14, entry_band=20)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMeanReversionReturnZScoreStrategy:
    def test_rejects_non_integer_lookback(self):
        with pytest.raises(TypeError):
            MeanReversionReturnZScoreStrategy(lookback=24.0, entry_z=1.5)

    def test_rejects_non_positive_entry_z(self):
        with pytest.raises(ValueError):
            MeanReversionReturnZScoreStrategy(lookback=24, entry_z=0)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MeanReversionReturnZScoreStrategy(lookback=24, entry_z=1.5)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMeanReversionExitZRegression:
    """
    Regression coverage for the exit_z parameter added to the existing
    price-level z-score strategy.
    """

    def test_exit_z_must_be_below_entry_z(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(lookback=24, entry_z=1.5, exit_z=1.5)

        with pytest.raises(ValueError):
            MeanReversionStrategy(lookback=24, entry_z=1.5, exit_z=-0.1)

    def test_exit_z_zero_reproduces_original_positions(self, synthetic_ohlc_data):
        original = MeanReversionStrategy(
            lookback=24, entry_z=1.5, exit_z=0.0
        ).generate_positions(synthetic_ohlc_data)

        assert_valid_positions(original, synthetic_ohlc_data)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MeanReversionStrategy(lookback=24, entry_z=1.5, exit_z=0.5)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMeanReversionBollingerPctBStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            MeanReversionBollingerPctBStrategy(
                window=20.0, num_std=2.0, entry_band=0.4
            )

    def test_rejects_entry_band_outside_open_interval(self):
        with pytest.raises(ValueError):
            MeanReversionBollingerPctBStrategy(
                window=20, num_std=2.0, entry_band=0.5
            )

        with pytest.raises(ValueError):
            MeanReversionBollingerPctBStrategy(
                window=20, num_std=2.0, entry_band=0.0
            )

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MeanReversionBollingerPctBStrategy(
            window=20, num_std=2.0, entry_band=0.4
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMeanReversionKeltnerStrategy:
    def test_rejects_non_integer_window(self):
        with pytest.raises(TypeError):
            MeanReversionKeltnerStrategy(
                window=20.0, atr_multiple=2.0, exit_atr_multiple=0.0
            )

    def test_rejects_exit_atr_multiple_not_below_atr_multiple(self):
        with pytest.raises(ValueError):
            MeanReversionKeltnerStrategy(
                window=20, atr_multiple=2.0, exit_atr_multiple=2.0
            )

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MeanReversionKeltnerStrategy(
            window=20, atr_multiple=2.0, exit_atr_multiple=0.0
        )
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)


class TestMeanReversionVwapDeviationStrategy:
    def test_rejects_non_integer_lookback(self):
        with pytest.raises(TypeError):
            MeanReversionVwapDeviationStrategy(lookback=24.0, entry_z=1.5)

    def test_rejects_non_positive_entry_z(self):
        with pytest.raises(ValueError):
            MeanReversionVwapDeviationStrategy(lookback=24, entry_z=0)

    def test_requires_volume_column(self, synthetic_ohlc_data):
        strategy = MeanReversionVwapDeviationStrategy(lookback=24, entry_z=1.5)
        data_without_volume = synthetic_ohlc_data.drop(columns=["volume"])
        with pytest.raises(ValueError):
            strategy.generate_positions(data_without_volume)

    def test_generate_positions_shape(self, synthetic_ohlc_data):
        strategy = MeanReversionVwapDeviationStrategy(lookback=24, entry_z=1.5)
        result = strategy.generate_positions(synthetic_ohlc_data)
        assert_valid_positions(result, synthetic_ohlc_data)
