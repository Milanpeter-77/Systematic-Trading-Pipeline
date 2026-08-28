from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Time-series momentum strategy based on the sign of a fixed-lookback
    total return.

    Behavioral hypothesis
    ---------------------
    Assets that have risen (fallen) over the recent past tend to keep
    rising (falling) over the near-term horizon, because information
    diffuses gradually and market participants underreact to it.

    Free parameters
    ---------------
    lookback
        Number of bars over which the total return is measured.
    threshold
        Minimum absolute total return required to take a position; smaller
        moves are treated as noise and left flat.

    Fixed design choices
    --------------------
    - Long when the lookback return exceeds threshold.
    - Short when the lookback return is below -threshold.
    - Flat when the return is within [-threshold, threshold] or unavailable.
    - Stateless: the position is recomputed fresh every bar, unlike the
      hold-until-exit state machine used by mean reversion.
    - No additional volatility, stop-loss, or confirmation parameters.
    """

    family_name = "momentum"
    parameter_names = ("lookback", "threshold")
    parameter_grid = {
        "lookback": [24, 36, 48, 96, 168],
        "threshold": [0.0, 0.01, 0.02],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        lookback = self.parameters["lookback"]
        threshold = self.parameters["threshold"]

        if not isinstance(lookback, int):
            raise TypeError("lookback must be an integer.")

        if lookback < 2:
            raise ValueError("lookback must be at least 2.")

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        lookback = self.parameters["lookback"]
        threshold = float(self.parameters["threshold"])

        result = data.copy()

        result["momentum_return"] = result["close"].pct_change(
            periods=lookback,
        )

        indicator_available = result["momentum_return"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["momentum_return"] > threshold),
                indicator_available
                & (result["momentum_return"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
