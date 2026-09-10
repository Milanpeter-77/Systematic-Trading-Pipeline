from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import macd
from src.strategies.base import BaseStrategy


class MomentumMacdHistogramStrategy(BaseStrategy):
    """
    Time-series momentum strategy based on the sign and slope of the MACD
    histogram (macd_line - signal_line), normalized by price.

    Behavioral hypothesis
    ---------------------
    Assets that have risen (fallen) over the recent past tend to keep
    rising (falling), because information diffuses gradually and market
    participants underreact to it -- the same hypothesis as the other
    momentum strategies. This one specifically looks for *accelerating*
    momentum: a histogram that is not just positive/negative but growing
    in that direction, distinct from trend_macd's line/signal crossover
    (which trades the crossover itself, with no separate magnitude or
    acceleration condition).

    Free parameters
    ---------------
    fast_span
        Span of the faster EMA used to build the MACD line.
    slow_span
        Span of the slower EMA used to build the MACD line.
    signal_span
        Span of the EMA applied to the MACD line to produce the signal
        line.
    threshold
        Minimum absolute bar-over-bar change in the price-normalized
        histogram required to take a position; smaller changes are
        treated as noise and left flat.

    Fixed design choices
    --------------------
    - Uses the raw (EMA-difference) MACD from src.data.features.technical.
      macd, then normalizes the histogram by dividing by close (a
      zero-close bar is treated as unavailable) -- unlike trend_macd and
      stat_arb_macd, this strategy thresholds an absolute magnitude, so it
      needs a value that's comparable across instruments trading at very
      different price levels (e.g. USDJPY vs EURUSD).
    - Long when the normalized histogram is positive and its bar-over-bar
      change exceeds threshold (positive momentum accelerating).
    - Short when the normalized histogram is negative and its bar-over-bar
      change is below -threshold (negative momentum accelerating).
    - Flat otherwise, including when unavailable.
    - Stateless: the position is recomputed fresh every bar, like the
      other momentum strategies.
    """

    family_name = "momentum_macd_histogram"
    parameter_names = ("fast_span", "slow_span", "signal_span", "threshold")
    parameter_grid = {
        "fast_span": [12, 24],
        "slow_span": [26, 52, 104],
        "signal_span": [9, 18],
        "threshold": [0.0, 0.0005],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        fast_span = self.parameters["fast_span"]
        slow_span = self.parameters["slow_span"]
        signal_span = self.parameters["signal_span"]
        threshold = self.parameters["threshold"]

        if not isinstance(fast_span, int):
            raise TypeError("fast_span must be an integer.")

        if not isinstance(slow_span, int):
            raise TypeError("slow_span must be an integer.")

        if not isinstance(signal_span, int):
            raise TypeError("signal_span must be an integer.")

        if fast_span < 2:
            raise ValueError("fast_span must be at least 2.")

        if slow_span < 3:
            raise ValueError("slow_span must be at least 3.")

        if signal_span < 2:
            raise ValueError("signal_span must be at least 2.")

        if fast_span >= slow_span:
            raise ValueError(
                "fast_span must be strictly smaller than slow_span."
            )

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        fast_span = self.parameters["fast_span"]
        slow_span = self.parameters["slow_span"]
        signal_span = self.parameters["signal_span"]
        threshold = float(self.parameters["threshold"])

        result = data.copy()

        _, _, histogram = macd(
            result["close"],
            fast_span,
            slow_span,
            signal_span,
        )

        valid_close = result["close"].where(result["close"] != 0)
        normalized_histogram = histogram / valid_close
        histogram_slope = normalized_histogram.diff()

        result["macd_histogram"] = normalized_histogram

        indicators_available = (
            normalized_histogram.notna() & histogram_slope.notna()
        )

        result["raw_signal"] = np.select(
            [
                indicators_available
                & (normalized_histogram > 0)
                & (histogram_slope > threshold),
                indicators_available
                & (normalized_histogram < 0)
                & (histogram_slope < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
