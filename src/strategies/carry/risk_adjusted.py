from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class CarryRiskAdjustedStrategy(BaseStrategy):
    """
    Stateless FX carry strategy based on the sign of the policy-rate
    differential scaled by the pair's own realized volatility.

    Behavioral hypothesis
    ---------------------
    Holding the higher-yielding currency of a pair and funding it by
    borrowing the lower-yielding one earns a persistent return -- the
    same carry hypothesis as the plain carry strategy. This variant scales
    the raw differential by the pair's own recent realized volatility, the
    same risk-adjustment idea as momentum_risk_adjusted: a given
    differential is a more attractive carry trade when the pair itself is
    calm than when it is volatile enough to regularly erode that
    differential through spot moves alone.

    Free parameters
    ---------------
    window
        Rolling window used to estimate the pair's own realized volatility
        of returns.
    threshold
        Minimum absolute risk-adjusted carry required to take a position.

    Fixed design choices
    --------------------
    - risk_adjusted_carry = interest_rate_differential /
      rolling_std(close.pct_change(), window), with the same zero-
      denominator guard used elsewhere in this codebase. Unlike
      momentum_risk_adjusted, no annualization is applied -- matching
      that strategy's own unannualized convention exactly.
    - Long the pair when risk_adjusted_carry exceeds threshold.
    - Short the pair when it is below -threshold.
    - Flat when within [-threshold, threshold] or unavailable (including
      when rolling volatility is zero).
    - Stateless: recomputed fresh every bar, mirroring the plain carry
      strategy.
    - Depends on an interest_rate_differential column already being
      present on the input data (added upstream during data ingestion,
      for CASH/FX instruments only) -- unlike carry_real_rate and
      carry_term_slope, this needs no new data pull: it's computed purely
      from the differential already used by the plain carry strategy plus
      this instrument's own OHLC.
    """

    family_name = "carry_risk_adjusted"
    parameter_names = ("window", "threshold")
    parameter_grid = {
        "window": [24, 48, 96],
        "threshold": [0.0, 0.25, 0.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        window = self.parameters["window"]
        threshold = self.parameters["threshold"]

        if not isinstance(window, int):
            raise TypeError("window must be an integer.")

        if window < 3:
            raise ValueError("window must be at least 3.")

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        if "interest_rate_differential" not in data.columns:
            raise ValueError(
                "Risk-adjusted carry strategy input is missing "
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

        window = self.parameters["window"]
        threshold = float(self.parameters["threshold"])

        result = data.copy()

        realized_vol = result["close"].pct_change().rolling(
            window=window,
            min_periods=window,
        ).std(ddof=0)

        valid_vol = realized_vol.where(realized_vol > 0)

        risk_adjusted_carry = result["interest_rate_differential"] / valid_vol
        result["risk_adjusted_carry"] = risk_adjusted_carry

        indicator_available = risk_adjusted_carry.notna()

        result["raw_signal"] = np.select(
            [
                indicator_available & (risk_adjusted_carry > threshold),
                indicator_available & (risk_adjusted_carry < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
