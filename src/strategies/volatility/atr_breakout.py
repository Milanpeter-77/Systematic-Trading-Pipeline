from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class VolatilityAtrBreakoutStrategy(BaseStrategy):
    """
    Stateful volatility-breakout strategy based on rolling bands built
    from Average True Range (ATR) instead of close-to-close standard
    deviation.

    Behavioral hypothesis
    ---------------------
    A close that breaks outside a band sized by the market's own recent
    true range signals a shift in the supply/demand balance strong enough
    to overcome typical short-term noise, and tends to continue in the
    breakout direction rather than immediately revert -- the same
    hypothesis as the close-to-close breakout strategy, but the band
    width reacts to gaps and intrabar range (true range), not just
    bar-to-bar closing-price dispersion.

    Free parameters
    ---------------
    window
        Rolling window used to estimate the local price mean and ATR.
    num_atr
        Number of ATRs from the rolling mean that defines the breakout
        band.

    Fixed design choices
    --------------------
    - true_range = max(high - low, |high - previous_close|,
      |low - previous_close|), Wilder's standard definition.
    - atr = rolling mean of true_range.
    - Enter long when close breaks above rolling_mean + num_atr * atr.
    - Enter short when close breaks below rolling_mean - num_atr * atr.
    - Exit a long position when close falls back to or below rolling_mean.
    - Exit a short position when close rises back to or above rolling_mean.
    - Hold the current position between entry and exit.
    - Trades with the breakout, like the close-to-close breakout strategy.
    - No stop-loss, holding-period, or separate exit parameter.
    """

    family_name = "volatility_atr"
    parameter_names = ("window", "num_atr")
    parameter_grid = {
        "window": [20, 40],
        "num_atr": [1.5, 2.0, 2.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        num_atr = self.parameters["num_atr"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(num_atr, (int, float)):
            raise TypeError("num_atr must be numeric.")

        if num_atr <= 0:
            raise ValueError("num_atr must be positive.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        num_atr = float(self.parameters["num_atr"])

        result = data.copy()

        previous_close = result["close"].shift(1)

        true_range = pd.concat(
            [
                result["high"] - result["low"],
                (result["high"] - previous_close).abs(),
                (result["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        result["atr"] = true_range.rolling(
            window=window,
            min_periods=window,
        ).mean()

        result["rolling_mean"] = result["close"].rolling(
            window=window,
            min_periods=window,
        ).mean()

        upper_band = result["rolling_mean"] + num_atr * result["atr"]
        lower_band = result["rolling_mean"] - num_atr * result["atr"]

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        close_values = result["close"].to_numpy()
        mean_values = result["rolling_mean"].to_numpy()
        upper_values = upper_band.to_numpy()
        lower_values = lower_band.to_numpy()

        for index in range(len(result)):
            close = close_values[index]
            mean = mean_values[index]
            upper = upper_values[index]
            lower = lower_values[index]

            if not np.isfinite(mean) or not np.isfinite(upper):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if close > upper:
                    current_position = 1
                elif close < lower:
                    current_position = -1

            elif current_position == 1:
                if close <= mean:
                    current_position = 0

            elif current_position == -1:
                if close >= mean:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                result["close"] > upper_band,
                result["close"] < lower_band,
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
