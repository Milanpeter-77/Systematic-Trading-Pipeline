from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import bollinger_bands
from src.strategies.base import BaseStrategy


class MeanReversionBollingerPctBStrategy(BaseStrategy):
    """
    Stateful short-horizon mean-reversion strategy based on %B, a price's
    position within its own Bollinger Bands.

    Behavioral hypothesis
    ---------------------
    Large short-term price deviations may partially reverse because of
    temporary liquidity imbalances, forced trading, market overreaction,
    and dealer inventory effects -- the same hypothesis as the price
    z-score mean-reversion strategy, expressed here as the price's
    position within its own band rather than a raw z-score.

    Free parameters
    ---------------
    window
        Rolling window used to compute the Bollinger Band mean and
        standard deviation.
    num_std
        Number of standard deviations from the rolling mean that defines
        the band.
    entry_band
        Distance from the %B midline (0.5) required to enter a contrarian
        position.

    Fixed design choices
    --------------------
    - %B = (close - lower_band) / (upper_band - lower_band), bounded in
      [0, 1] when close is within the band; a zero band width (rolling_std
      == 0) is treated as an unavailable indicator, i.e. flat, the same
      zero-denominator guard used elsewhere in this codebase.
    - Enter long when centered %B (%B - 0.5) <= -entry_band.
    - Enter short when centered %B >= entry_band.
    - Exit a long position when centered %B >= 0.
    - Exit a short position when centered %B <= 0.
    - Hold the current position between entry and exit.
    - Identical state-machine mechanics to mean_reversion's rsi.py, applied
      to a centered %B instead of a centered RSI.
    - Fades the band, unlike volatility/breakout.py, which trades the same
      Bollinger Band construction (src.data.features.technical.
      bollinger_bands) *with* the break.
    - No stop-loss or holding-period parameter.
    """

    family_name = "mean_reversion_bollinger"
    parameter_names = ("window", "num_std", "entry_band")
    parameter_grid = {
        "window": [20, 40, 80],
        "num_std": [1.5, 2.0, 2.5],
        "entry_band": [0.35, 0.4, 0.45],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        num_std = self.parameters["num_std"]
        entry_band = self.parameters["entry_band"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(num_std, (int, float)):
            raise TypeError("num_std must be numeric.")

        if num_std <= 0:
            raise ValueError("num_std must be positive.")

        if not isinstance(entry_band, (int, float)):
            raise TypeError("entry_band must be numeric.")

        if not 0 < entry_band < 0.5:
            raise ValueError(
                "entry_band must be strictly between 0 and 0.5."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        num_std = float(self.parameters["num_std"])
        entry_band = float(self.parameters["entry_band"])

        result = data.copy()

        _, upper, lower = bollinger_bands(result["close"], window, num_std)

        band_width = upper - lower
        valid_band_width = band_width.where(band_width > 0)

        pct_b = (result["close"] - lower) / valid_band_width
        pct_b_centered = pct_b - 0.5

        result["pct_b"] = pct_b

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        centered_values = pct_b_centered.to_numpy()

        for index, centered in enumerate(centered_values):
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
                pct_b_centered <= -entry_band,
                pct_b_centered >= entry_band,
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
