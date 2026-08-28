from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MomentumRiskAdjustedStrategy(BaseStrategy):
    """
    Time-series momentum strategy based on the sign of a rolling
    risk-adjusted return (mean period return divided by its own rolling
    standard deviation).

    Behavioral hypothesis
    ---------------------
    Drift that is consistent relative to its own noise is more likely to
    reflect genuine gradual information diffusion than drift of the same
    raw magnitude achieved through a few large, noisy moves -- the same
    underreaction hypothesis as the other two momentum strategies, scaled
    by the recent volatility of returns instead of measured as a raw
    magnitude.

    Free parameters
    ---------------
    window
        Rolling window used to estimate the local mean and volatility of
        period returns.
    threshold
        Minimum absolute risk-adjusted return required to take a position.

    Fixed design choices
    --------------------
    - risk_adjusted_return = rolling_mean(period_return) /
      rolling_std(period_return).
    - Long when risk_adjusted_return exceeds threshold.
    - Short when it is below -threshold.
    - Flat when it is within [-threshold, threshold] or unavailable
      (including when rolling volatility is zero).
    - Stateless: the position is recomputed fresh every bar, like the
      other two momentum strategies.
    - No additional volatility, stop-loss, or confirmation parameters
      beyond the risk-adjustment already built into the signal itself.
    """

    family_name = "momentum_risk_adjusted"
    parameter_names = ("window", "threshold")
    parameter_grid = {
        "window": [24, 96, 240],
        "threshold": [0.0, 0.05],
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

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        window = self.parameters["window"]
        threshold = float(self.parameters["threshold"])

        result = data.copy()

        period_return = result["close"].pct_change()

        result["rolling_mean_return"] = period_return.rolling(
            window=window,
            min_periods=window,
        ).mean()

        result["rolling_std_return"] = period_return.rolling(
            window=window,
            min_periods=window,
        ).std(ddof=0)

        valid_std = result["rolling_std_return"].where(
            result["rolling_std_return"] > 0
        )

        result["risk_adjusted_return"] = (
            result["rolling_mean_return"] / valid_std
        )

        indicator_available = result["risk_adjusted_return"].notna()

        result["raw_signal"] = np.select(
            [
                indicator_available
                & (result["risk_adjusted_return"] > threshold),
                indicator_available
                & (result["risk_adjusted_return"] < -threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
