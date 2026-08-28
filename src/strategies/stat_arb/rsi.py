from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class StatArbRsiStrategy(BaseStrategy):
    """
    Stateful pairs-trading strategy based on the Relative Strength Index
    (RSI) of a cointegrated pair's spread, a bounded overbought/oversold
    oscillator.

    Behavioral hypothesis
    ---------------------
    Two cointegrated instruments share a long-run equilibrium relationship;
    short-term deviations from it tend to revert, driven by the same
    liquidity/overreaction effects that motivate single-instrument mean
    reversion -- the same hypothesis as the pair-spread z-score strategy,
    expressed here through a bounded ratio of average gains to average
    losses in the spread instead of an unbounded spread z-score.

    Free parameters
    ---------------
    window
        Rolling window (in hourly bars) used to compute average gains and
        losses for RSI.
    entry_band
        Distance from the RSI midline (50) required to enter a contrarian
        position.

    Fixed design choices
    --------------------
    - Operates on a precomputed spread series (src.data.features.
      statistical.compute_pair_spread), packaged as a single-column
      pseudo-instrument with open=high=low=close=spread.
    - window is applied directly to hourly bars, matching
      mean_reversion_rsi's own calibration: a multi-day/weekly window
      damps RSI's average-gain/average-loss ratio toward the midline on
      hourly data (law of large numbers), producing far too few trades to
      be a viable test.
    - rsi = 100 - 100 / (1 + average_gain / average_loss); a window with
      zero average loss is treated as an unavailable indicator (flat),
      matching mean_reversion_rsi's own guard.
    - Enter long the spread when rsi <= 50 - entry_band.
    - Enter short the spread when rsi >= 50 + entry_band.
    - Exit a long position when rsi >= 50.
    - Exit a short position when rsi <= 50.
    - Hold the current position between entry and exit.
    - Identical mechanics to mean_reversion_rsi.py, applied to a spread
      instead of a raw price.
    - No stop-loss, holding-period, or separate exit parameter.
    """

    family_name = "stat_arb_rsi"
    parameter_names = ("window", "entry_band")
    parameter_grid = {
        "window": [14, 24, 48],
        "entry_band": [15, 20, 30],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        entry_band = self.parameters["entry_band"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(entry_band, (int, float)):
            raise TypeError("entry_band must be numeric.")

        if not 0 < entry_band < 50:
            raise ValueError(
                "entry_band must be strictly between 0 and 50."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        entry_band = float(self.parameters["entry_band"])

        result = data.copy()

        price_change = result["close"].diff()

        result["average_gain"] = price_change.clip(lower=0).rolling(
            window=window,
            min_periods=window,
        ).mean()

        result["average_loss"] = (-price_change).clip(lower=0).rolling(
            window=window,
            min_periods=window,
        ).mean()

        valid_average_loss = result["average_loss"].where(
            result["average_loss"] > 0
        )

        relative_strength = result["average_gain"] / valid_average_loss

        result["rsi"] = 100 - 100 / (1 + relative_strength)

        rsi_centered = result["rsi"] - 50

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        rsi_centered_values = rsi_centered.to_numpy()

        for index, centered in enumerate(rsi_centered_values):
            if not np.isfinite(centered):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if centered <= -entry_band:
                    current_position = 1
                elif centered >= entry_band:
                    current_position = -1

            elif current_position == 1:
                if centered >= 0:
                    current_position = 0

            elif current_position == -1:
                if centered <= 0:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                rsi_centered <= -entry_band,
                rsi_centered >= entry_band,
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
