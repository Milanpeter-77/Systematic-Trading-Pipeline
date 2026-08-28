from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MeanReversionRsiStrategy(BaseStrategy):
    """
    Stateful short-horizon mean-reversion strategy based on the Relative
    Strength Index (RSI), a bounded overbought/oversold oscillator.

    Behavioral hypothesis
    ---------------------
    Large short-term price deviations may partially reverse because of
    temporary liquidity imbalances, forced trading, market overreaction,
    and dealer inventory effects -- the same hypothesis as the price
    z-score mean-reversion strategy, expressed here through a bounded
    ratio of average gains to average losses instead of an unbounded
    price z-score.

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
    - window is applied directly to hourly bars (the classic RSI-14
      convention used at whatever bar interval is being traded, not
      scaled to a daily-equivalent horizon): RSI's average-gain/average-
      loss ratio is a law-of-large-numbers estimate that damps toward the
      midline as the window grows, so windows scaled the way this
      repo's other rolling lookbacks are (multi-day/weekly) would make
      RSI too smooth to ever reach the entry bands -- confirmed against
      real EURUSD data, where a 168-336 bar window with these entry
      bands produced zero trades across the whole 5-year history.
    - rsi = 100 - 100 / (1 + average_gain / average_loss). A window with
      zero average loss is treated as an unavailable indicator, i.e.
      flat, rather than forced to an extreme rsi=100 reading -- the same
      zero-denominator guard used by the price z-score strategy's
      rolling_std.
    - Enter long when rsi <= 50 - entry_band.
    - Enter short when rsi >= 50 + entry_band.
    - Exit a long position when rsi >= 50.
    - Exit a short position when rsi <= 50.
    - Hold the current position between entry and exit.
    - Identical state-machine mechanics to mean_reversion's zscore.py,
      applied to a centered RSI instead of a price z-score.
    - No stop-loss, holding-period, or separate exit parameter.
    """

    family_name = "mean_reversion_rsi"
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
