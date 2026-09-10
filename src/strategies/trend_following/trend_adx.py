from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import adx
from src.strategies.base import BaseStrategy


class TrendAdxStrategy(BaseStrategy):
    """
    Long-short trend-following strategy based on a +DI/-DI directional
    crossover, gated by the Average Directional Index (ADX) so a position
    is only taken once trend strength is confirmed.

    Behavioral hypothesis
    ---------------------
    Price trends may persist because information is incorporated gradually,
    investors adjust positions slowly, and market participants exhibit
    underreaction and herding -- the same hypothesis as the other trend
    strategies, but this one adds an explicit strength filter: a
    directional crossover during a genuinely weak/choppy regime (low ADX)
    is more likely to be noise than the start of a persistent trend.

    Free parameters
    ---------------
    window
        Rolling window used to compute +DI, -DI, and ADX (see
        src.data.features.technical.adx).
    adx_threshold
        Minimum ADX level required to trade a directional crossover;
        below this, the regime is treated as non-trending and the
        strategy stays flat regardless of which of +DI/-DI is larger.

    Fixed design choices
    --------------------
    - Long when +DI is above -DI and ADX exceeds adx_threshold.
    - Short when -DI is above +DI and ADX exceeds adx_threshold.
    - Flat whenever ADX is at or below adx_threshold, or the indicators
      are unavailable -- unlike the other trend strategies, direction
      alone is not sufficient to take a position here.
    - Stateless: recomputed fresh every bar, like every other trend_
      following sibling -- there is no separate hold-until-exit state,
      the position simply tracks whatever the gated directional signal
      says on each bar.
    - No additional volatility, stop-loss, or confirmation parameters
      beyond the ADX gate itself.
    """

    family_name = "trend_adx"
    parameter_names = ("window", "adx_threshold")
    parameter_grid = {
        "window": [14, 24, 48],
        "adx_threshold": [20, 25, 30],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        adx_threshold = self.parameters["adx_threshold"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(adx_threshold, (int, float)):
            raise TypeError("adx_threshold must be numeric.")

        if not 0 < adx_threshold <= 100:
            raise ValueError(
                "adx_threshold must satisfy 0 < adx_threshold <= 100."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        adx_threshold = float(self.parameters["adx_threshold"])

        result = data.copy()

        plus_di, minus_di, adx_value = adx(
            result["high"],
            result["low"],
            result["close"],
            window,
        )

        result["plus_di"] = plus_di
        result["minus_di"] = minus_di
        result["adx"] = adx_value

        indicators_available = (
            plus_di.notna() & minus_di.notna() & adx_value.notna()
        )
        trend_confirmed = indicators_available & (adx_value > adx_threshold)

        result["raw_signal"] = np.select(
            [
                trend_confirmed & (plus_di > minus_di),
                trend_confirmed & (minus_di > plus_di),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
