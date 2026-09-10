from __future__ import annotations

import pandas as pd

from src.data.features.technical import parabolic_sar
from src.strategies.base import BaseStrategy


class TrendPsarStrategy(BaseStrategy):
    """
    Long-short trend-following strategy based on Wilder's Parabolic SAR
    (stop-and-reverse) indicator.

    Behavioral hypothesis
    ---------------------
    Price trends may persist because information is incorporated gradually,
    investors adjust positions slowly, and market participants exhibit
    underreaction and herding -- the same hypothesis as the other trend
    strategies. SAR expresses this as a trailing, accelerating stop level
    that flips direction the moment price closes through it, rather than
    a comparison between two smoothed price benchmarks.

    Free parameters
    ---------------
    acceleration_step
        Both the initial acceleration factor and the amount it increases
        by each time the trend makes a new extreme.
    acceleration_max
        Cap on the acceleration factor.

    Fixed design choices
    --------------------
    - This is a third kind of construction distinct from every other
      strategy in this codebase: not stateless (unlike the rest of the
      trend_following family) and not the enter/hold/exit position state
      machine used by mean_reversion/volatility/stat_arb. SAR is a
      *recursive indicator* -- sar[t] depends on sar[t-1], the running
      trend direction, the trend's running extreme point, and an
      accelerating factor -- so it cannot be vectorized with .rolling()/
      .ewm() and is computed with one explicit bar-by-bar loop (see
      src.data.features.technical.parabolic_sar) that produces the
      direction, and therefore target_position, directly. There is no
      separate entry/exit threshold on top of it: direction itself is the
      position, the instant price crosses the SAR level.
    - Flat on the first two bars (index 0 and 1), which are needed to
      establish the initial direction and extreme point.
    - No additional volatility, stop-loss, or confirmation parameters --
      the trailing stop already built into SAR is the only risk control.
    """

    family_name = "trend_psar"
    parameter_names = ("acceleration_step", "acceleration_max")
    parameter_grid = {
        "acceleration_step": [0.01, 0.02, 0.03],
        "acceleration_max": [0.1, 0.2, 0.3],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        acceleration_step = self.parameters["acceleration_step"]
        acceleration_max = self.parameters["acceleration_max"]

        if not isinstance(acceleration_step, (int, float)):
            raise TypeError("acceleration_step must be numeric.")

        if not isinstance(acceleration_max, (int, float)):
            raise TypeError("acceleration_max must be numeric.")

        if acceleration_step <= 0:
            raise ValueError("acceleration_step must be positive.")

        if acceleration_step > acceleration_max:
            raise ValueError(
                "acceleration_step must satisfy "
                "acceleration_step <= acceleration_max."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        acceleration_step = float(self.parameters["acceleration_step"])
        acceleration_max = float(self.parameters["acceleration_max"])

        result = data.copy()

        direction = parabolic_sar(
            result["high"],
            result["low"],
            result["close"],
            acceleration_step,
            acceleration_max,
        )

        result["sar_direction"] = direction
        result["raw_signal"] = direction.astype("int8")
        result["target_position"] = result["raw_signal"].astype("int8")

        return result
