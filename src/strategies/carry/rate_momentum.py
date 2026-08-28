from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class CarryRateMomentumStrategy(BaseStrategy):
    """
    Stateless FX carry strategy based on the sign of the recent change in
    the base-vs-quote short-term policy-rate differential, rather than
    its absolute level.

    Behavioral hypothesis
    ---------------------
    A widening rate differential signals building carry-trade
    attractiveness (and typically follows a diverging monetary-policy
    cycle that tends to persist for several meetings), while a narrowing
    differential signals fading attractiveness -- capturing the trend in
    carry conditions rather than the static carry strategy's level test,
    which only says whether the differential currently favors one side,
    not whether that favor is building or fading.

    Free parameters
    ---------------
    lookback
        Number of bars over which the change in the rate differential is
        measured.
    threshold
        Minimum absolute change in the differential (in percentage
        points) required to take a position; smaller shifts are treated
        as noise and left flat.

    Fixed design choices
    --------------------
    - lookback is expressed in calendar-day units (24 bars/day), not the
      shorter trading-day-scale lookbacks used by the price-based
      families: FOMC/ECB/BOE/RBA meet roughly every 6-8 weeks, so the
      differential (which is forward-filled monthly from FRED policy
      rates -- see src.pipelines.data_ingestion.pipeline.
      add_interest_rate_differential) only moves on a policy-cycle
      cadence, and a short lookback would show near-zero change most of
      the time between meetings.
    - Long when the differential's change over lookback bars exceeds
      threshold.
    - Short when it is below -threshold.
    - Flat when it is within [-threshold, threshold] or unavailable.
    - Stateless: recomputed fresh every bar, matching the static carry
      strategy.
    - Because policy-rate regimes shift infrequently, some lookback/
      threshold combinations may realistically produce very few, or over
      a given historical window even zero, trades -- an honest
      consequence of how rarely central banks move, not a bug (the
      pre-existing static carry strategy already exhibits the same
      thinness across its own threshold grid).
    - Depends on an interest_rate_differential column already being
      present on the input data (CASH/FX instruments only).
    """

    family_name = "carry_rate_momentum"
    parameter_names = ("lookback", "threshold")
    parameter_grid = {
        "lookback": [720, 1440, 2160],
        "threshold": [0.0, 0.25, 0.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        lookback = self.parameters["lookback"]
        threshold = self.parameters["threshold"]

        if not isinstance(lookback, int):
            raise TypeError("lookback must be an integer.")

        if lookback < 2:
            raise ValueError("lookback must be at least 2.")

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        if "interest_rate_differential" not in data.columns:
            raise ValueError(
                "Carry strategy input is missing "
                "'interest_rate_differential'. This is added during data "
                "ingestion for CASH/FX instruments only -- see "
                "src.pipelines.data_ingestion.pipeline."
                "add_interest_rate_differential."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        lookback = self.parameters["lookback"]
        threshold = float(self.parameters["threshold"])

        result = data.copy()

        result["differential_change"] = result[
            "interest_rate_differential"
        ].diff(periods=lookback)

        indicator_available = result["differential_change"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["differential_change"] > threshold),
                indicator_available
                & (result["differential_change"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
