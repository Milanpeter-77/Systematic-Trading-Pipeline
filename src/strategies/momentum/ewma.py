from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MomentumEwmaStrategy(BaseStrategy):
    """
    Time-series momentum strategy based on the sign of an exponentially
    weighted average of period returns.

    Behavioral hypothesis
    ---------------------
    Assets that have risen (fallen) over the recent past tend to keep
    rising (falling), because information diffuses gradually and market
    participants underreact to it -- the same hypothesis as the
    fixed-lookback total-return momentum strategy, but recency-weighted
    so a single endpoint bar can't dominate the signal the way it can in
    a two-point total-return measurement.

    Free parameters
    ---------------
    span
        Span of the exponentially weighted average of period returns.
    threshold
        Minimum absolute smoothed average return required to take a
        position; smaller drifts are treated as noise and left flat.

    Fixed design choices
    --------------------
    - Long when the EWMA-smoothed period return exceeds threshold.
    - Short when it is below -threshold.
    - Flat when it is within [-threshold, threshold] or unavailable.
    - Stateless: the position is recomputed fresh every bar, like the
      fixed-lookback momentum strategy.
    - No additional volatility, stop-loss, or confirmation parameters.
    """

    family_name = "momentum_ewma"
    parameter_names = ("span", "threshold")
    parameter_grid = {
        "span": [24, 96, 240],
        "threshold": [0.0, 0.0002],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        span = self.parameters["span"]
        threshold = self.parameters["threshold"]

        if not isinstance(span, int):
            raise TypeError("span must be an integer.")

        if span < 2:
            raise ValueError("span must be at least 2.")

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        span = self.parameters["span"]
        threshold = float(self.parameters["threshold"])

        result = data.copy()

        period_return = result["close"].pct_change()

        result["smoothed_return"] = period_return.ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()

        indicator_available = result["smoothed_return"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["smoothed_return"] > threshold),
                indicator_available
                & (result["smoothed_return"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
