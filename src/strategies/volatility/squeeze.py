from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import bollinger_bands, keltner_channels
from src.strategies.base import BaseStrategy


class VolatilitySqueezeStrategy(BaseStrategy):
    """
    Stateful strategy that trades the release of a volatility "squeeze":
    a Bollinger Band width contracting inside its own Keltner Channel
    width, followed by a directionally-confirmed breakout.

    Behavioral hypothesis
    ---------------------
    When close-to-close price dispersion (Bollinger Band width) contracts
    inside true-range-based dispersion (Keltner Channel width), the market
    is in an unusually quiet, range-bound state that tends to precede a
    volatility expansion -- a shift in the supply/demand balance strong
    enough to overcome the recent quiet, similar in spirit to the other
    volatility-breakout strategies, but this one explicitly identifies the
    quiet "coiling" regime first rather than reacting only once a band is
    already broken.

    Free parameters
    ---------------
    window
        Rolling window shared by both the Bollinger Band statistics and
        the Keltner Channel's EMA/ATR calculation.
    num_std
        Number of standard deviations from the rolling mean that defines
        the Bollinger Band width.
    atr_multiple
        Number of ATRs from the EMA midline that defines the Keltner
        Channel width.

    Fixed design choices
    --------------------
    - squeeze_on = Bollinger Band width < Keltner Channel width.
    - squeeze_release = squeeze_on was True on the previous bar and is
      False on the current bar (the squeeze has just ended).
    - Enter long on squeeze_release when close > open (directional
      confirmation of an upward break).
    - Enter short on squeeze_release when close < open.
    - No entry when close == open -- no directional confirmation.
    - Exit (either direction) once squeeze_on becomes True again (the
      squeeze has re-engaged), regardless of position direction.
    - Hold the current position between entry and exit.
    - Shares its Bollinger/Keltner band math with mean_reversion/
      bollinger_pct_b.py and mean_reversion/keltner.py (via
      src.data.features.technical.bollinger_bands/keltner_channels), but
      trades the interaction between the two bands rather than either
      band alone.
    - No stop-loss or holding-period parameter.
    """

    family_name = "volatility_squeeze"
    parameter_names = ("window", "num_std", "atr_multiple")
    parameter_grid = {
        "window": [20, 40],
        "num_std": [1.5, 2.0],
        "atr_multiple": [1.5, 2.0],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        num_std = self.parameters["num_std"]
        atr_multiple = self.parameters["atr_multiple"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(num_std, (int, float)):
            raise TypeError("num_std must be numeric.")

        if num_std <= 0:
            raise ValueError("num_std must be positive.")

        if not isinstance(atr_multiple, (int, float)):
            raise TypeError("atr_multiple must be numeric.")

        if atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        num_std = float(self.parameters["num_std"])
        atr_multiple = float(self.parameters["atr_multiple"])

        result = data.copy()

        _, bb_upper, bb_lower = bollinger_bands(result["close"], window, num_std)
        _, kc_upper, kc_lower = keltner_channels(
            result["high"], result["low"], result["close"], window, atr_multiple
        )

        bb_width = bb_upper - bb_lower
        kc_width = kc_upper - kc_lower

        squeeze_on = bb_width < kc_width
        squeeze_release = squeeze_on.shift(1).fillna(False) & (~squeeze_on)

        result["bb_width"] = bb_width
        result["kc_width"] = kc_width
        result["squeeze_on"] = squeeze_on

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        bb_width_values = bb_width.to_numpy()
        squeeze_on_values = squeeze_on.to_numpy()
        squeeze_release_values = squeeze_release.to_numpy()
        close_values = result["close"].to_numpy()
        open_values = result["open"].to_numpy()

        for index in range(len(result)):
            if not np.isfinite(bb_width_values[index]):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if squeeze_release_values[index]:
                    if close_values[index] > open_values[index]:
                        current_position = 1
                    elif close_values[index] < open_values[index]:
                        current_position = -1

            elif current_position == 1:
                if squeeze_on_values[index]:
                    current_position = 0

            elif current_position == -1:
                if squeeze_on_values[index]:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                squeeze_release & (result["close"] > result["open"]),
                squeeze_release & (result["close"] < result["open"]),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
