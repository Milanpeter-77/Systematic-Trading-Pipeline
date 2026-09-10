from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import atr, keltner_channels
from src.strategies.base import BaseStrategy


class MeanReversionKeltnerStrategy(BaseStrategy):
    """
    Stateful short-horizon mean-reversion strategy based on price versus a
    Keltner Channel (an EMA midline banded by a multiple of ATR).

    Behavioral hypothesis
    ---------------------
    Large short-term price deviations may partially reverse because of
    temporary liquidity imbalances, forced trading, market overreaction,
    and dealer inventory effects -- the same hypothesis as the price
    z-score mean-reversion strategy, expressed here through a band sized
    by the market's own recent true range (ATR) rather than close-to-close
    standard deviation.

    Free parameters
    ---------------
    window
        Rolling window used to compute the EMA midline and ATR.
    atr_multiple
        Number of ATRs from the midline that defines the entry band.
    exit_atr_multiple
        Number of ATRs from the midline at which an open position is
        closed. Set below atr_multiple to exit before price fully reverts
        to the midline (a partial-reversion exit); exit_atr_multiple=0
        reproduces exiting exactly at the midline.

    Fixed design choices
    --------------------
    - Enter long when close breaks below midline - atr_multiple * ATR
      (price unusually low, bet on reversion up).
    - Enter short when close breaks above midline + atr_multiple * ATR.
    - Exit a long position when close rises back to or above
      midline - exit_atr_multiple * ATR.
    - Exit a short position when close falls back to or below
      midline + exit_atr_multiple * ATR.
    - Hold the current position between entry and exit.
    - Fades the band, unlike volatility/atr_breakout.py, which trades the
      same ATR-band construction *with* the break.
    - Uses ATR-sized bands rather than mean_reversion/bollinger_pct_b.py's
      close-to-close standard-deviation bands.
    - No stop-loss or holding-period parameter.
    """

    family_name = "mean_reversion_keltner"
    parameter_names = ("window", "atr_multiple", "exit_atr_multiple")
    parameter_grid = {
        "window": [20, 40, 80],
        "atr_multiple": [1.5, 2.0, 2.5],
        "exit_atr_multiple": [0.0, 0.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        atr_multiple = self.parameters["atr_multiple"]
        exit_atr_multiple = self.parameters["exit_atr_multiple"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(atr_multiple, (int, float)):
            raise TypeError("atr_multiple must be numeric.")

        if atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive.")

        if not isinstance(exit_atr_multiple, (int, float)):
            raise TypeError("exit_atr_multiple must be numeric.")

        if not 0 <= exit_atr_multiple < atr_multiple:
            raise ValueError(
                "exit_atr_multiple must satisfy "
                "0 <= exit_atr_multiple < atr_multiple."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        atr_multiple = float(self.parameters["atr_multiple"])
        exit_atr_multiple = float(self.parameters["exit_atr_multiple"])

        result = data.copy()

        mid, upper, lower = keltner_channels(
            result["high"],
            result["low"],
            result["close"],
            window,
            atr_multiple,
        )
        band_atr = atr(result["high"], result["low"], result["close"], window)

        exit_long_level = mid - exit_atr_multiple * band_atr
        exit_short_level = mid + exit_atr_multiple * band_atr

        result["keltner_mid"] = mid

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        close_values = result["close"].to_numpy()
        mid_values = mid.to_numpy()
        upper_values = upper.to_numpy()
        lower_values = lower.to_numpy()
        exit_long_values = exit_long_level.to_numpy()
        exit_short_values = exit_short_level.to_numpy()

        for index in range(len(result)):
            close = close_values[index]
            mid_value = mid_values[index]

            if not np.isfinite(mid_value):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if close < lower_values[index]:
                    current_position = 1
                elif close > upper_values[index]:
                    current_position = -1

            elif current_position == 1:
                if close >= exit_long_values[index]:
                    current_position = 0

            elif current_position == -1:
                if close <= exit_short_values[index]:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                result["close"] < lower,
                result["close"] > upper,
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
