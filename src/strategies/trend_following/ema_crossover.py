from __future__ import annotations
from unittest import result

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class TrendStrategy(BaseStrategy):
    """
    Long-short time-series momentum strategy based on two exponential
    moving averages.

    Behavioral hypothesis
    ---------------------
    Price trends may persist because information is incorporated gradually,
    investors adjust positions slowly, and market participants exhibit
    underreaction and herding.

    Free parameters
    ---------------
    fast_window
        Span of the faster exponential moving average.
    slow_window
        Span of the slower exponential moving average.

    Fixed design choices
    --------------------
    - Long when the fast EMA is above the slow EMA.
    - Short when the fast EMA is below the slow EMA.
    - Flat until both indicators are available.
    - No additional volatility, stop-loss, or confirmation parameters.
    """

    family_name = "trend"
    parameter_names = ("fast_window", "slow_window")
    parameter_grid = {
        "fast_window": [12, 24],
        "slow_window": [72, 120, 240],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        fast_window = self.parameters["fast_window"]
        slow_window = self.parameters["slow_window"]

        if not isinstance(fast_window, int):
            raise TypeError("fast_window must be an integer.")

        if not isinstance(slow_window, int):
            raise TypeError("slow_window must be an integer.")

        if fast_window < 2:
            raise ValueError("fast_window must be at least 2.")

        if slow_window < 3:
            raise ValueError("slow_window must be at least 3.")

        if fast_window >= slow_window:
            raise ValueError(
                "fast_window must be strictly smaller than slow_window."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        fast_window = self.parameters["fast_window"]
        slow_window = self.parameters["slow_window"]

        result = data.copy()

        result["fast_ema"] = result["close"].ewm(
            span=fast_window,
            adjust=False,
            min_periods=fast_window,
        ).mean()

        result["slow_ema"] = result["close"].ewm(
            span=slow_window,
            adjust=False,
            min_periods=slow_window,
        ).mean()

        indicators_available = (
            result["fast_ema"].notna()
            & result["slow_ema"].notna()
        )

        result["raw_signal"] = np.select(
            [
                indicators_available
                & (result["fast_ema"] > result["slow_ema"]),
                indicators_available
                & (result["fast_ema"] < result["slow_ema"]),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result