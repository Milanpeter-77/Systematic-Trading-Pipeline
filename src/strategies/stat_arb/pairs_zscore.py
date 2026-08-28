from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class PairsZScoreStrategy(BaseStrategy):
    """
    Stateful pairs-trading strategy based on a rolling z-score of a
    cointegrated pair's spread.

    Behavioral hypothesis
    ---------------------
    Two instruments whose prices are cointegrated (src.data.features.
    statistical.screen_cointegrated_pairs) share a long-run equilibrium
    relationship; short-term deviations from it tend to revert, driven by
    the same liquidity/overreaction effects that motivate single-instrument
    mean reversion, but expressed as a relative (spread) trade instead of a
    directional bet on either leg.

    Free parameters
    ---------------
    lookback
        Rolling window used to estimate the local spread mean and volatility.
    entry_z
        Absolute z-score required to enter a contrarian position.
    exit_z
        Absolute z-score at which an open position is closed. Set below
        entry_z to exit before the spread fully reverts to its mean (a
        partial-reversion exit); exit_z=0 reproduces exiting exactly at
        the mean.

    Fixed design choices
    --------------------
    - Operates on a precomputed spread series (src.data.features.
      statistical.compute_pair_spread), packaged as a single-column
      pseudo-instrument with open=high=low=close=spread -- the hedge ratio
      is fixed at screening time, not re-estimated per bar.
    - Enter long the spread when z <= -entry_z.
    - Enter short the spread when z >= entry_z.
    - Exit a long position when z >= -exit_z.
    - Exit a short position when z <= exit_z.
    - Hold the current position between entry and exit.
    - No stop-loss or holding-period parameter -- otherwise identical
      mechanics to mean_reversion's zscore.py, applied to a spread instead
      of a raw price.
    """

    family_name = "stat_arb"
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

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        lookback = self.parameters["lookback"]
        entry_z = float(self.parameters["entry_z"])
        exit_z = float(self.parameters["exit_z"])

        result = data.copy()

        result["rolling_mean"] = result["close"].rolling(
            window=lookback,
            min_periods=lookback,
        ).mean()

        result["rolling_std"] = result["close"].rolling(
            window=lookback,
            min_periods=lookback,
        ).std(ddof=0)

        valid_std = result["rolling_std"].where(
            result["rolling_std"] > 0
        )

        result["z_score"] = (
            result["close"] - result["rolling_mean"]
        ) / valid_std

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
