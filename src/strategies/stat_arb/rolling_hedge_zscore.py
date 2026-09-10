from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.statistical import compute_rolling_pair_spread
from src.strategies.base import BaseStrategy


class StatArbRollingHedgeZScoreStrategy(BaseStrategy):
    """
    Stateful pairs-trading strategy based on a rolling z-score of a
    cointegrated pair's spread, where the hedge ratio itself is
    re-estimated in a trailing window rather than fixed at screening time.

    Behavioral hypothesis
    ---------------------
    Two instruments whose prices are cointegrated (src.data.features.
    statistical.screen_cointegrated_pairs) share a long-run equilibrium
    relationship; short-term deviations from it tend to revert -- the
    same hypothesis as stat_arb's other members. This variant additionally
    allows for the relationship itself (the hedge ratio) to drift over
    time, which pairs_zscore.py's single full-sample hedge ratio cannot
    capture.

    Free parameters
    ---------------
    lookback
        Rolling window used both to re-estimate the hedge ratio (see
        src.data.features.statistical.compute_rolling_pair_spread) and to
        estimate the local spread mean and volatility for the z-score --
        the same dual-purpose reuse of one window RSI already uses for
        its own window.
    entry_z
        Absolute z-score required to enter a contrarian position.
    exit_z
        Absolute z-score at which an open position is closed. Set below
        entry_z to exit before the spread fully reverts to its mean (a
        partial-reversion exit); exit_z=0 reproduces exiting exactly at
        the mean.

    Fixed design choices
    --------------------
    - Unlike every other stat_arb member, this strategy does not read the
      precomputed static spread's close column. It reads leg_a_close/
      leg_b_close directly (added to the pair pseudo-instrument by
      src.pipelines.alpha_factory.pipeline.prepare_pair_features
      specifically for this and stat_arb_correlation_breakdown) and
      re-derives its own adaptive spread via compute_rolling_pair_spread.
    - The hedge ratio needs lookback bars of warmup before it's available,
      and the z-score needs a further lookback bars of valid spread
      before it's available -- roughly 2*lookback bars of total warmup,
      the same "second differencing needs more warmup" trade-off
      momentum_acceleration already documents for its own construction.
    - Enter long the spread when z <= -entry_z.
    - Enter short the spread when z >= entry_z.
    - Exit a long position when z >= -exit_z.
    - Exit a short position when z <= exit_z.
    - Hold the current position between entry and exit.
    - Otherwise identical state-machine mechanics to pairs_zscore.py.
    - No stop-loss or holding-period parameter.
    """

    family_name = "stat_arb_rolling_hedge"
    parameter_names = ("lookback", "entry_z", "exit_z")
    parameter_grid = {
        "lookback": [24, 48, 96, 168],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.0, 0.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        lookback = self.parameters["lookback"]
        entry_z = self.parameters["entry_z"]
        exit_z = self.parameters["exit_z"]

        if not isinstance(lookback, int):
            raise TypeError("lookback must be an integer.")

        if lookback < 3:
            raise ValueError("lookback must be at least 3.")

        if not isinstance(entry_z, (int, float)):
            raise TypeError("entry_z must be numeric.")

        if entry_z <= 0:
            raise ValueError("entry_z must be positive.")

        if not isinstance(exit_z, (int, float)):
            raise TypeError("exit_z must be numeric.")

        if not 0 <= exit_z < entry_z:
            raise ValueError("exit_z must satisfy 0 <= exit_z < entry_z.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        missing = {"leg_a_close", "leg_b_close"} - set(data.columns)

        if missing:
            raise ValueError(
                "Rolling-hedge stat_arb strategy input is missing "
                f"{sorted(missing)}. These columns are added to the pair "
                "pseudo-instrument during alpha-factory feature prep -- "
                "see src.pipelines.alpha_factory.pipeline."
                "prepare_pair_features."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        lookback = self.parameters["lookback"]
        entry_z = float(self.parameters["entry_z"])
        exit_z = float(self.parameters["exit_z"])

        result = data.copy()

        spread, hedge_ratio = compute_rolling_pair_spread(
            result["leg_a_close"],
            result["leg_b_close"],
            window=lookback,
        )

        spread = spread.reindex(result.index)
        hedge_ratio = hedge_ratio.reindex(result.index)

        result["rolling_spread"] = spread
        result["hedge_ratio"] = hedge_ratio

        rolling_mean = spread.rolling(
            window=lookback,
            min_periods=lookback,
        ).mean()

        rolling_std = spread.rolling(
            window=lookback,
            min_periods=lookback,
        ).std(ddof=0)

        valid_std = rolling_std.where(rolling_std > 0)

        result["z_score"] = (spread - rolling_mean) / valid_std

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        z_values = result["z_score"].to_numpy()

        for index, z_score in enumerate(z_values):
            if not np.isfinite(z_score):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if z_score <= -entry_z:
                    current_position = 1
                elif z_score >= entry_z:
                    current_position = -1

            elif current_position == 1:
                if z_score >= -exit_z:
                    current_position = 0

            elif current_position == -1:
                if z_score <= exit_z:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                result["z_score"] <= -entry_z,
                result["z_score"] >= entry_z,
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
