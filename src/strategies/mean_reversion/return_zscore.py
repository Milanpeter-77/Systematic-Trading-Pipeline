from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MeanReversionReturnZScoreStrategy(BaseStrategy):
    """
    Stateful short-horizon mean-reversion strategy based on a rolling
    z-score of period-over-period returns, rather than the price level.

    Behavioral hypothesis
    ---------------------
    A single-period return that is unusually large relative to its own
    recent distribution reflects a short-term overreaction -- forced
    trading, order-flow imbalance, or a liquidity shock -- that tends to
    partially reverse over the next few bars, in the spirit of the
    short-term-reversal literature. This targets a shorter, sharper
    horizon than the price-level z-score strategy, which reacts to a
    sustained deviation building up over the whole lookback window.

    Free parameters
    ---------------
    lookback
        Rolling window used to estimate the local mean and volatility of
        period returns.
    entry_z
        Absolute z-score required to enter a contrarian position.

    Fixed design choices
    --------------------
    - Enter long when the return z-score z <= -entry_z.
    - Enter short when z >= entry_z.
    - Exit a long position when z >= 0.
    - Exit a short position when z <= 0.
    - Hold the current position between entry and exit.
    - Identical state-machine mechanics to mean_reversion's zscore.py,
      applied to close.pct_change() instead of the price level itself.
    - No stop-loss, holding-period, or separate exit parameter.
    """

    family_name = "mean_reversion_return_zscore"
    parameter_names = ("lookback", "entry_z")
    parameter_grid = {
        "lookback": [12, 24, 48],
        "entry_z": [1.5, 2.0],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        lookback = self.parameters["lookback"]
        entry_z = self.parameters["entry_z"]

        if not isinstance(lookback, int):
            raise TypeError("lookback must be an integer.")

        if lookback < 3:
            raise ValueError("lookback must be at least 3.")

        if not isinstance(entry_z, (int, float)):
            raise TypeError("entry_z must be numeric.")

        if entry_z <= 0:
            raise ValueError("entry_z must be positive.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        lookback = self.parameters["lookback"]
        entry_z = float(self.parameters["entry_z"])

        result = data.copy()

        result["period_return"] = result["close"].pct_change()

        result["rolling_mean"] = result["period_return"].rolling(
            window=lookback,
            min_periods=lookback,
        ).mean()

        result["rolling_std"] = result["period_return"].rolling(
            window=lookback,
            min_periods=lookback,
        ).std(ddof=0)

        valid_std = result["rolling_std"].where(
            result["rolling_std"] > 0
        )

        result["z_score"] = (
            result["period_return"] - result["rolling_mean"]
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
                if z_score >= 0:
                    current_position = 0

            elif current_position == -1:
                if z_score <= 0:
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
