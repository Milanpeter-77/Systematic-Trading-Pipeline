from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import macd
from src.strategies.base import BaseStrategy


class TrendMacdStrategy(BaseStrategy):
    """
    Long-short trend-following strategy based on a MACD line/signal-line
    crossover.

    Behavioral hypothesis
    ---------------------
    Price trends may persist because information is incorporated gradually,
    investors adjust positions slowly, and market participants exhibit
    underreaction and herding -- the same hypothesis as the dual-EMA trend
    strategy, expressed here through the difference of two EMAs (and its
    own smoothed signal line) rather than a direct comparison of the two
    EMAs themselves.

    Free parameters
    ---------------
    fast_span
        Span of the faster EMA used to build the MACD line.
    slow_span
        Span of the slower EMA used to build the MACD line.
    signal_span
        Span of the EMA applied to the MACD line to produce the signal
        line.

    Fixed design choices
    --------------------
    - macd_line = fast EMA(close) - slow EMA(close).
    - signal_line = EMA(macd_line, signal_span).
    - Long when macd_line is above signal_line.
    - Short when macd_line is below signal_line.
    - Flat until both are available.
    - Stateless: recomputed fresh every bar, like every other trend_following
      sibling.
    - Uses the raw (EMA-difference) MACD from src.data.features.technical.
      macd, not a percentage/PPO-normalized version: like the dual-EMA,
      SMA-filter, and Donchian trend strategies, this only ever compares
      the indicator's sign relative to another quantity (here, the signal
      line) -- it has no absolute-magnitude threshold parameter -- so no
      cross-instrument scale normalization is needed. Contrast with
      momentum_macd_histogram (same underlying MACD construction, but
      normalized because it thresholds an absolute magnitude) and
      stat_arb_macd (this same construction applied to a pair spread
      instead of a real instrument's price).
    """

    family_name = "trend_macd"
    parameter_names = ("fast_span", "slow_span", "signal_span")
    parameter_grid = {
        "fast_span": [12, 24],
        "slow_span": [26, 52, 104],
        "signal_span": [9, 18],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        fast_span = self.parameters["fast_span"]
        slow_span = self.parameters["slow_span"]
        signal_span = self.parameters["signal_span"]

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

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        fast_span = self.parameters["fast_span"]
        slow_span = self.parameters["slow_span"]
        signal_span = self.parameters["signal_span"]

        result = data.copy()

        macd_line, signal_line, _ = macd(
            result["close"],
            fast_span,
            slow_span,
            signal_span,
        )

        result["macd_line"] = macd_line
        result["signal_line"] = signal_line

        indicators_available = macd_line.notna() & signal_line.notna()

        result["raw_signal"] = np.select(
            [
                indicators_available & (macd_line > signal_line),
                indicators_available & (macd_line < signal_line),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
