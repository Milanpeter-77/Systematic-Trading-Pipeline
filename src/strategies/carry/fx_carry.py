from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class FXCarryStrategy(BaseStrategy):
    """
    Stateless FX carry strategy based on the sign of the base-vs-quote
    short-term policy-rate differential.

    Behavioral hypothesis
    ---------------------
    Holding the higher-yielding currency of a pair and funding it by
    borrowing the lower-yielding one earns a persistent return (the
    classic carry trade), because interest-rate differentials are far
    slower-moving and more persistent than the spot exchange rate itself,
    and uncovered interest parity holds only weakly and inconsistently in
    practice.

    Free parameters
    ---------------
    threshold
        Minimum absolute rate differential (in percentage points) required
        to take a position; smaller differentials are treated as noise and
        left flat.

    Fixed design choices
    --------------------
    - Long the pair (long base currency) when the differential exceeds
      threshold.
    - Short the pair when the differential is below -threshold.
    - Flat when the differential is within [-threshold, threshold] or
      unavailable.
    - Stateless: recomputed fresh every bar from the current differential,
      not a running position held until some separate exit signal --
      matches how the differential itself only changes slowly (monthly,
      per src.pipelines.data_ingestion.pipeline.add_interest_rate_
      differential), so there's no separate "entry" event to react to.
    - Depends on an interest_rate_differential column already being
      present on the input data (added upstream during data ingestion,
      for CASH/FX instruments only) -- this is not something the strategy
      computes itself, matching every other strategy's pattern of taking
      already-prepared market data and computing only its own indicators
      from it.
    """

    family_name = "carry"
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

        threshold = float(self.parameters["threshold"])

        result = data.copy()

        indicator_available = result["interest_rate_differential"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["interest_rate_differential"] > threshold),
                indicator_available
                & (result["interest_rate_differential"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
