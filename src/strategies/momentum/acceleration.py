from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MomentumAccelerationStrategy(BaseStrategy):
    """
    Time-series momentum strategy based on the sign of the change in a
    fixed-lookback total return over another lookback window -- momentum
    of momentum.

    Behavioral hypothesis
    ---------------------
    Assets that have risen (fallen) over the recent past tend to keep
    rising (falling), because information diffuses gradually and market
    participants underreact to it -- the same hypothesis as the plain
    fixed-lookback momentum strategy. This variant targets the second
    derivative: not whether the recent return is large, but whether it is
    growing relative to the prior period's return of the same length,
    which may better capture a trend that is still building rather than
    one that has already run its course.

    Free parameters
    ---------------
    lookback
        Number of bars over which the total return is measured, and the
        number of bars back over which that return is differenced.
    threshold
        Minimum absolute acceleration required to take a position;
        smaller changes are treated as noise and left flat.

    Fixed design choices
    --------------------
    - momentum_return = close.pct_change(lookback).
    - acceleration = momentum_return.diff(lookback) (the change in the
      lookback-bar return compared to lookback bars earlier) -- needs
      2 * lookback bars of warmup before it is available.
    - Long when acceleration exceeds threshold.
    - Short when acceleration is below -threshold.
    - Flat when within [-threshold, threshold] or unavailable.
    - Stateless: the position is recomputed fresh every bar, like the
      other momentum strategies.
    - A second difference is noisier and smaller in magnitude than a
      first difference, so this strategy's threshold grid is deliberately
      smaller than the plain momentum strategy's.
    """

    family_name = "momentum_acceleration"
    parameter_names = ("lookback", "threshold")
    parameter_grid = {
        "lookback": [24, 48, 96],
        "threshold": [0.0, 0.005, 0.01],
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

        result["acceleration"] = result["momentum_return"].diff(
            periods=lookback,
        )

        indicator_available = result["acceleration"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available & (result["acceleration"] > threshold),
                indicator_available & (result["acceleration"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
