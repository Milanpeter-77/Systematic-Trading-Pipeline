from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import assert_valid_positions

from src.data.features.statistical import compute_pair_spread
from src.strategies.stat_arb.correlation_breakdown import (
    StatArbCorrelationBreakdownStrategy,
)
from src.strategies.stat_arb.macd_spread import StatArbMacdStrategy
from src.strategies.stat_arb.pairs_zscore import PairsZScoreStrategy
from src.strategies.stat_arb.return_zscore import StatArbReturnZScoreStrategy
from src.strategies.stat_arb.rolling_hedge_zscore import (
    StatArbRollingHedgeZScoreStrategy,
)
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


@pytest.fixture
def synthetic_pair_spread_data_with_legs(
    synthetic_ohlc_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extends synthetic_pair_spread_data with leg_a_close/leg_b_close, as
    added by src.pipelines.alpha_factory.pipeline.prepare_pair_features
    for stat_arb_rolling_hedge and stat_arb_correlation_breakdown. Leg B
    is a linear function of leg A plus independent noise, so the two are
    correlated but not identical -- close enough to a real cointegrated
    pair to exercise both strategies' hedge-ratio/correlation math
    without needing a real cointegration screen.
    """
    rng = np.random.default_rng(11)

    leg_a_close = synthetic_ohlc_data["close"]
    noise = rng.normal(0, leg_a_close.std() * 0.05, len(leg_a_close))
    leg_b_close = 0.5 * leg_a_close + 50 + noise

    spread, _ = compute_pair_spread(leg_a_close, leg_b_close)
    price_like_spread = spread - spread.min() + spread.std()

    return pd.DataFrame(
        {
            "open": price_like_spread,
            "high": price_like_spread,
            "low": price_like_spread,
            "close": price_like_spread,
            "leg_a_close": leg_a_close,
            "leg_b_close": leg_b_close,
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


class TestStatArbMacdStrategy:
    def test_rejects_non_integer_fast_span(self):
        with pytest.raises(TypeError):
            StatArbMacdStrategy(fast_span=12.0, slow_span=26, signal_span=9)

    def test_rejects_fast_span_not_below_slow_span(self):
        with pytest.raises(ValueError):
            StatArbMacdStrategy(fast_span=26, slow_span=26, signal_span=9)

    def test_shift_invariant_to_arbitrary_spread_offset(
        self, synthetic_pair_spread_data
    ):
        """
        Regression guard: this strategy must use the raw (EMA-difference)
        MACD, not a percentage/PPO-normalized version, because
        compute_pair_spread shifts the raw spread by an arbitrary positive
        constant. Shifting the input spread up by a large constant must
        not change the resulting positions, the same shift-invariance
        property already guarded for stat_arb_return_zscore.
        """
        strategy = StatArbMacdStrategy(fast_span=12, slow_span=26, signal_span=9)

        base_result = strategy.generate_positions(synthetic_pair_spread_data)

        shifted_data = synthetic_pair_spread_data + 1_000_000.0
        shifted_result = strategy.generate_positions(shifted_data)

        pd.testing.assert_series_equal(
            base_result["target_position"],
            shifted_result["target_position"],
        )

    def test_generate_positions_shape(self, synthetic_pair_spread_data):
        strategy = StatArbMacdStrategy(fast_span=12, slow_span=26, signal_span=9)
        result = strategy.generate_positions(synthetic_pair_spread_data)
        assert_valid_positions(result, synthetic_pair_spread_data)


class TestStatArbRollingHedgeZScoreStrategy:
    def test_rejects_non_positive_entry_z(self):
        with pytest.raises(ValueError):
            StatArbRollingHedgeZScoreStrategy(lookback=24, entry_z=0, exit_z=0.0)

    def test_requires_leg_columns(self, synthetic_pair_spread_data):
        strategy = StatArbRollingHedgeZScoreStrategy(
            lookback=48, entry_z=1.5, exit_z=0.5
        )
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_pair_spread_data)

    def test_generate_positions_shape(
        self, synthetic_pair_spread_data_with_legs
    ):
        strategy = StatArbRollingHedgeZScoreStrategy(
            lookback=48, entry_z=1.5, exit_z=0.5
        )
        result = strategy.generate_positions(
            synthetic_pair_spread_data_with_legs
        )
        assert_valid_positions(result, synthetic_pair_spread_data_with_legs)


class TestStatArbCorrelationBreakdownStrategy:
    def test_rejects_non_positive_entry_z(self):
        with pytest.raises(ValueError):
            StatArbCorrelationBreakdownStrategy(
                corr_window=48, entry_z=0, exit_z=0.0
            )

    def test_requires_leg_columns(self, synthetic_pair_spread_data):
        strategy = StatArbCorrelationBreakdownStrategy(
            corr_window=48, entry_z=1.5, exit_z=0.5
        )
        with pytest.raises(ValueError):
            strategy.generate_positions(synthetic_pair_spread_data)

    def test_generate_positions_shape(
        self, synthetic_pair_spread_data_with_legs
    ):
        strategy = StatArbCorrelationBreakdownStrategy(
            corr_window=48, entry_z=1.5, exit_z=0.5
        )
        result = strategy.generate_positions(
            synthetic_pair_spread_data_with_legs
        )
        assert_valid_positions(result, synthetic_pair_spread_data_with_legs)
