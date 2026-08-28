from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class TrendSmaFilterStrategy(BaseStrategy):
    """
    Long-short trend-following strategy based on price versus a single
    rolling simple moving average.

    Behavioral hypothesis
    ---------------------
    Price trends may persist because information is incorporated gradually,
    investors adjust positions slowly, and market participants exhibit
    underreaction and herding -- the same hypothesis as the dual-EMA trend
    strategy, expressed here as price relative to one trailing benchmark
    instead of the difference between two moving averages.

    Free parameters
    ---------------
    window
        Rolling window used to compute the simple moving average benchmark.

    Fixed design choices
    --------------------
    - Long when close is above the rolling SMA.
    - Short when close is below the rolling SMA.
    - Flat until the SMA is available.
    - Stateless: recomputed fresh every bar, like the dual-EMA trend
      strategy, unlike the hold-until-exit state machines used by mean
      reversion and volatility breakout.
    - No additional volatility, stop-loss, or confirmation parameters.
    """

    family_name = "trend_sma"
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

        result["rolling_sma"] = result["close"].rolling(
            window=window,
            min_periods=window,
        ).mean()

        indicator_available = result["rolling_sma"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["close"] > result["rolling_sma"]),
                indicator_available
                & (result["close"] < result["rolling_sma"]),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
