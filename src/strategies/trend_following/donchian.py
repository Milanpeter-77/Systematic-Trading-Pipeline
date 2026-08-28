from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class TrendDonchianStrategy(BaseStrategy):
    """
    Long-short trend-following strategy based on price versus the midpoint
    of its rolling high-low trading range (Donchian channel).

    Behavioral hypothesis
    ---------------------
    A price sustained on one side of its own recent trading-range center
    reflects a persistent shift in the balance of buying and selling
    pressure, in the same spirit as the dual-EMA and SMA-filter trend
    strategies, but built from range extremes instead of any moving
    average of the close.

    Free parameters
    ---------------
    window
        Rolling window used to compute the high-low channel.

    Fixed design choices
    --------------------
    - channel_mid = (rolling max(high) + rolling min(low)) / 2, both
      current-bar-inclusive.
    - Long when close is above channel_mid.
    - Short when close is below channel_mid.
    - Flat until the channel is available.
    - Deliberately uses the channel midpoint rather than testing whether
      close breaks its own rolling max/min: since the current bar's own
      high/low are included in that rolling max/min, a literal breakout
      test against a current-bar-inclusive extreme is self-referential
      and would almost never trigger.
    - Stateless: recomputed fresh every bar, like the other trend
      strategies.
    - No additional volatility, stop-loss, or confirmation parameters.
    """

    family_name = "trend_donchian"
    parameter_names = ("window",)
    parameter_grid = {
        "window": [120, 240, 480],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]

        result = data.copy()

        rolling_high = result["high"].rolling(
            window=window,
            min_periods=window,
        ).max()

        rolling_low = result["low"].rolling(
            window=window,
            min_periods=window,
        ).min()

        result["channel_mid"] = (rolling_high + rolling_low) / 2

        indicator_available = result["channel_mid"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["close"] > result["channel_mid"]),
                indicator_available
                & (result["close"] < result["channel_mid"]),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
