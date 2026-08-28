from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Stateful volatility-breakout strategy based on rolling Bollinger-style
    bands.

    Behavioral hypothesis
    ---------------------
    A close that breaks outside its recent trading range signals a shift
    in the supply/demand balance strong enough to overcome typical
    short-term noise, and tends to continue in the breakout direction
    rather than immediately revert.

    Free parameters
    ---------------
    window
        Rolling window used to estimate the local price mean and volatility.
    num_std
        Number of standard deviations from the rolling mean that defines
        the breakout band.
    exit_num_std
        Number of standard deviations from the rolling mean at which an
        open position is closed. Set below num_std to exit before price
        fully reverts to the mean (a partial-reversion exit);
        exit_num_std=0 reproduces exiting exactly at the mean.

    Fixed design choices
    --------------------
    - Enter long when close breaks above rolling_mean + num_std * rolling_std.
    - Enter short when close breaks below rolling_mean - num_std * rolling_std.
    - Exit a long position when close falls back to or below
      rolling_mean + exit_num_std * rolling_std.
    - Exit a short position when close rises back to or above
      rolling_mean - exit_num_std * rolling_std.
    - Hold the current position between entry and exit.
    - Trades with the breakout, unlike the existing z-score mean-reversion
      strategy, which fades it.
    - No stop-loss or holding-period parameter.
    """

    family_name = "volatility"
    parameter_names = ("window", "num_std", "exit_num_std")
    parameter_grid = {
        "window": [20, 40, 80],
        "num_std": [1.5, 2.0, 2.5, 3.0],
        "exit_num_std": [0.0, 0.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        num_std = self.parameters["num_std"]
        exit_num_std = self.parameters["exit_num_std"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(num_std, (int, float)):
            raise TypeError("num_std must be numeric.")

        if num_std <= 0:
            raise ValueError("num_std must be positive.")

        if not isinstance(exit_num_std, (int, float)):
            raise TypeError("exit_num_std must be numeric.")

        if not 0 <= exit_num_std < num_std:
            raise ValueError(
                "exit_num_std must satisfy 0 <= exit_num_std < num_std."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        num_std = float(self.parameters["num_std"])
        exit_num_std = float(self.parameters["exit_num_std"])

        result = data.copy()

        result["rolling_mean"] = result["close"].rolling(
            window=window,
            min_periods=window,
        ).mean()

        result["rolling_std"] = result["close"].rolling(
            window=window,
            min_periods=window,
        ).std(ddof=0)

        upper_band = result["rolling_mean"] + num_std * result["rolling_std"]
        lower_band = result["rolling_mean"] - num_std * result["rolling_std"]
        exit_upper_band = (
            result["rolling_mean"] + exit_num_std * result["rolling_std"]
        )
        exit_lower_band = (
            result["rolling_mean"] - exit_num_std * result["rolling_std"]
        )

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        close_values = result["close"].to_numpy()
        mean_values = result["rolling_mean"].to_numpy()
        upper_values = upper_band.to_numpy()
        lower_values = lower_band.to_numpy()
        exit_upper_values = exit_upper_band.to_numpy()
        exit_lower_values = exit_lower_band.to_numpy()

        for index in range(len(result)):
            close = close_values[index]
            mean = mean_values[index]
            upper = upper_values[index]
            lower = lower_values[index]
            exit_upper = exit_upper_values[index]
            exit_lower = exit_lower_values[index]

            if not np.isfinite(mean):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if close > upper:
                    current_position = 1
                elif close < lower:
                    current_position = -1

            elif current_position == 1:
                if close <= exit_upper:
                    current_position = 0

            elif current_position == -1:
                if close >= exit_lower:
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
