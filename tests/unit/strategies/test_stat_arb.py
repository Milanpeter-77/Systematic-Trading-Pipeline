from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import assert_valid_positions

from src.strategies.stat_arb.pairs_zscore import PairsZScoreStrategy
from src.strategies.stat_arb.return_zscore import StatArbReturnZScoreStrategy
from src.strategies.stat_arb.rsi import StatArbRsiStrategy


@pytest.fixture
def synthetic_pair_spread_data(synthetic_ohlc_data: pd.DataFrame) -> pd.DataFrame:
    """
    A pair pseudo-instrument shaped like src.data.features.statistical.
    compute_pair_spread's output: open=high=low=close=spread.
    """
    spread = synthetic_ohlc_data["close"]
    return pd.DataFrame(
        {
            "open": spread,
            "high": spread,
            "low": spread,
            "close": spread,
        },
        index=synthetic_ohlc_data.index,
    )


class TestStatArbRsiStrategy:
    def test_rejects_entry_band_outside_open_interval(self):
        with pytest.raises(ValueError):
            StatArbRsiStrategy(window=24, entry_band=50)

    def test_generate_positions_shape(self, synthetic_pair_spread_data):
        strategy = StatArbRsiStrategy(window=14, entry_band=20)
        result = strategy.generate_positions(synthetic_pair_spread_data)
        assert_valid_positions(result, synthetic_pair_spread_data)


class TestStatArbReturnZScoreStrategy:
    def test_rejects_non_positive_entry_z(self):
        with pytest.raises(ValueError):
            StatArbReturnZScoreStrategy(lookback=24, entry_z=0)

    def test_uses_diff_not_pct_change(self, synthetic_pair_spread_data):
        """
        Regression guard: this strategy must use close.diff(), not
        close.pct_change(), because compute_pair_spread shifts the raw
        spread by an arbitrary positive constant, making a percentage
        return an artifact of that shift rather than a real signal.
        Shifting the input spread up by a large constant must not change
        the resulting positions, since diff() is shift-invariant while
        pct_change() would not be.
        """
        strategy = StatArbReturnZScoreStrategy(lookback=24, entry_z=1.5)

        base_result = strategy.generate_positions(synthetic_pair_spread_data)

        shifted_data = synthetic_pair_spread_data + 1_000_000.0
        shifted_result = strategy.generate_positions(shifted_data)

        pd.testing.assert_series_equal(
            base_result["target_position"],
            shifted_result["target_position"],
        )

    def test_generate_positions_shape(self, synthetic_pair_spread_data):
        strategy = StatArbReturnZScoreStrategy(lookback=24, entry_z=1.5)
        result = strategy.generate_positions(synthetic_pair_spread_data)
        assert_valid_positions(result, synthetic_pair_spread_data)


class TestStatArbExitZRegression:
    """
    Regression coverage for the exit_z parameter added to the existing
    pair-spread z-score strategy.
    """

    def test_exit_z_must_be_below_entry_z(self):
        with pytest.raises(ValueError):
            PairsZScoreStrategy(lookback=24, entry_z=1.5, exit_z=1.5)

    def test_generate_positions_shape(self, synthetic_pair_spread_data):
        strategy = PairsZScoreStrategy(lookback=24, entry_z=1.5, exit_z=0.5)
        result = strategy.generate_positions(synthetic_pair_spread_data)
        assert_valid_positions(result, synthetic_pair_spread_data)
