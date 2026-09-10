from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import macd
from src.strategies.base import BaseStrategy


class StatArbMacdStrategy(BaseStrategy):
    """
    Stateless pairs-trading strategy based on a MACD line/signal-line
    crossover applied to a cointegrated pair's spread.

    Behavioral hypothesis
    ---------------------
    Two instruments whose prices are cointegrated (src.data.features.
    statistical.screen_cointegrated_pairs) share a long-run equilibrium
    relationship; short-term deviations from it tend to revert -- the
    same hypothesis as the other stat_arb strategies, but this one trades
    the spread's own trend/momentum dynamics via MACD rather than fading
    a z-score or RSI extreme.

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
    - Operates on a precomputed spread series (src.data.features.
      statistical.compute_pair_spread), packaged as a single-column
      pseudo-instrument with open=high=low=close=spread.
    - Uses the raw (EMA-difference) MACD from src.data.features.technical.
      macd, not a percentage/PPO-normalized version -- this matters here
      specifically: prepare_pair_features() shifts the raw spread by an
      arbitrary positive constant to keep it safely positive for the
      backtest engine's price-ratio return convention, and a raw EMA
      difference is shift-invariant (unaffected by that constant) the
      same way stat_arb/return_zscore.py's docstring already documents
      for its own use of diff() instead of pct_change() -- a
      percentage-normalized MACD would not be.
    - Long the spread when macd_line is above signal_line.
    - Short the spread when macd_line is below signal_line.
    - Flat until both are available.
    - Stateless: recomputed fresh every bar, mirroring trend_macd.py's
      construction applied to the spread instead of a real instrument's
      price -- unlike stat_arb's other members, there is no separate
      hold-until-exit state.
    """

    family_name = "stat_arb_macd"
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
