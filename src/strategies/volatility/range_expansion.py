from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class VolatilityRangeExpansionStrategy(BaseStrategy):
    """
    Stateless single-bar volatility-expansion strategy: trades in the
    direction of any bar whose own high-low range is an unusual multiple
    of its recent average range.

    Behavioral hypothesis
    ---------------------
    A bar whose trading range suddenly expands well beyond its recent
    norm reflects a genuine, information-driven repricing rather than
    routine noise, and the direction that bar closed in captures the
    initial thrust of that repricing -- a day-trading-style range-
    expansion signal, distinct from the multi-bar close-price bands used
    by the other two volatility strategies.

    Free parameters
    ---------------
    window
        Rolling window used to estimate the recent average bar range.
    expansion_multiple
        Multiple of the average range that the current bar's own range
        must exceed to count as an expansion.

    Fixed design choices
    --------------------
    - current_range = high - low (this bar only).
    - average_range = rolling mean of current_range over the preceding
      window bars, excluding this bar, so a bar can't inflate its own
      baseline.
    - Long when current_range > expansion_multiple * average_range and
      this bar closed above its open.
    - Short when current_range > expansion_multiple * average_range and
      this bar closed below its open.
    - Flat otherwise, including when the average range is unavailable or
      zero.
    - expansion_multiple must be at least 1.0: it is only meaningful as a
      multiple of the baseline range, not a fraction of it.
    - Stateless: the position is recomputed fresh every bar, unlike the
      hold-until-exit ATR/close-band breakout strategies in this family.
    - No stop-loss or holding-period parameter.
    """

    family_name = "volatility_range_expansion"
    parameter_names = ("window", "expansion_multiple")
    parameter_grid = {
        "window": [20, 40],
        "expansion_multiple": [1.5, 2.0],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        expansion_multiple = self.parameters["expansion_multiple"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(expansion_multiple, (int, float)):
            raise TypeError("expansion_multiple must be numeric.")

        if expansion_multiple < 1.0:
            raise ValueError("expansion_multiple must be at least 1.0.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        expansion_multiple = float(self.parameters["expansion_multiple"])

        result = data.copy()

        current_range = result["high"] - result["low"]

        result["average_range"] = current_range.shift(1).rolling(
            window=window,
            min_periods=window,
        ).mean()

        valid_average_range = result["average_range"].where(
            result["average_range"] > 0
        )

        expanded = current_range > (
            expansion_multiple * valid_average_range
        )

        result["raw_signal"] = np.select(
            [
                expanded & (result["close"] > result["open"]),
                expanded & (result["close"] < result["open"]),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
