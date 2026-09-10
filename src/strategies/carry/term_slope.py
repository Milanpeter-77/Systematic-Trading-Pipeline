from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class CarryTermSlopeStrategy(BaseStrategy):
    """
    Stateless FX carry strategy based on the sign of the base-vs-quote
    term-structure slope differential (10-year yield minus policy rate).

    Behavioral hypothesis
    ---------------------
    A currency whose government yield curve is steeper (10-year yield well
    above the policy rate) signals the market expects further tightening
    or compensates investors for duration risk more generously than a
    currency with a flatter or inverted curve -- a "roll-down" proxy
    distinct from the plain carry strategy's short-term policy-rate
    differential, capturing the shape of the curve rather than just its
    short end.

    Free parameters
    ---------------
    threshold
        Minimum absolute term-slope differential (in percentage points)
        required to take a position; smaller differentials are treated as
        noise and left flat.

    Fixed design choices
    --------------------
    - Long the pair (long base currency) when the term-slope differential
      exceeds threshold.
    - Short the pair when the differential is below -threshold.
    - Flat when the differential is within [-threshold, threshold] or
      unavailable.
    - Stateless: recomputed fresh every bar, mirroring the plain carry
      strategy -- the differential itself only changes slowly.
    - Depends on a term_slope_differential column already being present
      on the input data (added upstream during data ingestion, for
      CASH/FX instruments only) -- this is not something the strategy
      computes itself.
    """

    family_name = "carry_term_slope"
    parameter_names = ("threshold",)
    parameter_grid = {
        "threshold": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        threshold = self.parameters["threshold"]

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        if "term_slope_differential" not in data.columns:
            raise ValueError(
                "Term-slope carry strategy input is missing "
                "'term_slope_differential'. This is added during data "
                "ingestion for CASH/FX instruments only -- see "
                "src.pipelines.data_ingestion.pipeline."
                "add_term_slope_differential."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        threshold = float(self.parameters["threshold"])

        result = data.copy()

        indicator_available = result["term_slope_differential"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["term_slope_differential"] > threshold),
                indicator_available
                & (result["term_slope_differential"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
