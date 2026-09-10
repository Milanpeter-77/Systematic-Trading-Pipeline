from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class CarryRealRateStrategy(BaseStrategy):
    """
    Stateless FX carry strategy based on the sign of the base-vs-quote
    real (inflation-adjusted) policy-rate differential.

    Behavioral hypothesis
    ---------------------
    Holding the higher-yielding currency of a pair and funding it by
    borrowing the lower-yielding one earns a persistent return -- the
    same carry hypothesis as the plain policy-rate carry strategy, but
    adjusted for inflation: a currency's nominal yield advantage is a
    less meaningful carry signal when it is largely offset by higher
    domestic inflation eroding that currency's purchasing power.

    Free parameters
    ---------------
    threshold
        Minimum absolute real-rate differential (in percentage points)
        required to take a position; smaller differentials are treated as
        noise and left flat.

    Fixed design choices
    --------------------
    - Long the pair (long base currency) when the real-rate differential
      exceeds threshold.
    - Short the pair when the differential is below -threshold.
    - Flat when the differential is within [-threshold, threshold] or
      unavailable.
    - Stateless: recomputed fresh every bar, mirroring the plain carry
      strategy -- the differential itself only changes slowly.
    - Depends on a real_rate_differential column already being present on
      the input data (added upstream during data ingestion, for CASH/FX
      instruments only) -- this is not something the strategy computes
      itself, matching every other strategy's pattern of taking already-
      prepared market data and computing only its own indicators from it.
    """

    family_name = "carry_real_rate"
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

        if "real_rate_differential" not in data.columns:
            raise ValueError(
                "Real-rate carry strategy input is missing "
                "'real_rate_differential'. This is added during data "
                "ingestion for CASH/FX instruments only -- see "
                "src.pipelines.data_ingestion.pipeline."
                "add_real_rate_differential."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        threshold = float(self.parameters["threshold"])

        result = data.copy()

        indicator_available = result["real_rate_differential"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["real_rate_differential"] > threshold),
                indicator_available
                & (result["real_rate_differential"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
