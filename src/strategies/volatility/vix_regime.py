from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class VolatilityVixRegimeStrategy(BaseStrategy):
    """
    Stateless strategy based on the spread between the CBOE VIX and this
    instrument's own annualized realized volatility.

    Behavioral hypothesis
    ---------------------
    The VIX reflects the market's aggregate, forward-looking price of
    volatility risk. When it sits well above what an individual instrument
    has actually been realizing, systemic/risk-off pressure is elevated
    relative to that instrument's own recent calm -- treated here as a
    signal to reduce risk-taking exposure. When an instrument is realizing
    materially more volatility than the broad market is currently pricing,
    that is treated as idiosyncratic, local stress rather than a systemic
    event, and faded.

    Free parameters
    ---------------
    realized_vol_window
        Rolling window used to estimate this instrument's own annualized
        realized volatility.
    threshold
        Minimum absolute vol_premium (in the same percentage-point units
        as VIX) required to take a position.

    Fixed design choices
    --------------------
    - realized_vol_pct = rolling_std(close.pct_change(), realized_vol_window)
      * sqrt(24 * 252) * 100 -- annualized to VIX's own quoted percentage-
      point scale (24 bars/day for these hourly bars, 252 trading days/year).
    - vol_premium = vix_level - realized_vol_pct.
    - Short when vol_premium exceeds threshold (implied fear elevated
      relative to this instrument's own calm -- a risk-off gate).
    - Long when vol_premium is below -threshold (this instrument realizing
      more than the broad market currently prices -- fade the local
      stress).
    - Flat when within [-threshold, threshold] or unavailable.
    - Stateless: recomputed fresh every bar.
    - Honest limitation: the same short/long convention is applied across
      every asset class this pipeline trades (FX, commodities, equities)
      by necessity -- one shared formula, one shared parameter grid -- but
      the economically "correct" direction is genuinely asset-class
      dependent (e.g. gold has historically behaved as a risk-off hedge,
      plausibly the opposite of a risk-sensitive FX pair or equity). This
      is a simplifying convention, not a claim of universal correctness.
    - Depends on a vix_level column already being present on the input
      data (added upstream during data ingestion, for every instrument)
      -- this is not something the strategy computes itself.
    """

    family_name = "volatility_vix_regime"
    parameter_names = ("realized_vol_window", "threshold")
    parameter_grid = {
        "realized_vol_window": [24, 48, 96],
        "threshold": [0.0, 5.0, 10.0],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        realized_vol_window = self.parameters["realized_vol_window"]
        threshold = self.parameters["threshold"]

        if not isinstance(realized_vol_window, int):
            raise TypeError("realized_vol_window must be an integer.")

        if realized_vol_window < 3:
            raise ValueError("realized_vol_window must be at least 3.")

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        if "vix_level" not in data.columns:
            raise ValueError(
                "VIX-regime strategy input is missing 'vix_level'. This "
                "is added during data ingestion for every instrument -- "
                "see src.pipelines.data_ingestion.pipeline.add_vix_regime."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        realized_vol_window = self.parameters["realized_vol_window"]
        threshold = float(self.parameters["threshold"])

        result = data.copy()

        period_return = result["close"].pct_change()

        realized_vol_pct = period_return.rolling(
            window=realized_vol_window,
            min_periods=realized_vol_window,
        ).std(ddof=0) * np.sqrt(24 * 252) * 100

        vol_premium = result["vix_level"] - realized_vol_pct
        result["vol_premium"] = vol_premium

        indicator_available = vol_premium.notna()

        result["raw_signal"] = np.select(
            [
                indicator_available & (vol_premium > threshold),
                indicator_available & (vol_premium < -threshold),
            ],
            [-1, 1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
