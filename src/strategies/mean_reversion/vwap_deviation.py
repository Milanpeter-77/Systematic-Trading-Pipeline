from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.technical import vwap
from src.strategies.base import BaseStrategy


class MeanReversionVwapDeviationStrategy(BaseStrategy):
    """
    Stateful short-horizon mean-reversion strategy based on a rolling
    z-score of price's deviation from its own rolling volume-weighted
    average price (VWAP).

    Behavioral hypothesis
    ---------------------
    Large short-term price deviations may partially reverse because of
    temporary liquidity imbalances, forced trading, market overreaction,
    and dealer inventory effects -- the same hypothesis as the price
    z-score mean-reversion strategy, expressed here relative to a
    volume-weighted benchmark instead of a simple rolling mean, so bars
    with unusually heavy volume pull the benchmark toward themselves.

    Free parameters
    ---------------
    lookback
        Rolling window used to compute VWAP and the local mean/volatility
        of price's deviation from it.
    entry_z
        Absolute z-score required to enter a contrarian position.

    Fixed design choices
    --------------------
    - deviation = close - rolling_vwap.
    - z = deviation / rolling_std(deviation, lookback), with the same
      zero-denominator guard used elsewhere in this codebase.
    - Enter long when z <= -entry_z.
    - Enter short when z >= entry_z.
    - Exit a long position when z >= 0.
    - Exit a short position when z <= 0.
    - Hold the current position between entry and exit.
    - Identical state-machine mechanics to mean_reversion's
      return_zscore.py, applied to a VWAP deviation instead of a period
      return.
    - Depends on a volume column already being present on the input data
      (present on every instrument's OHLCV bars today) -- this is not
      something the strategy computes itself.
    - No stop-loss, holding-period, or separate exit parameter.
    """

    family_name = "mean_reversion_vwap"
    parameter_names = ("lookback", "entry_z")
    parameter_grid = {
        "lookback": [24, 48, 96],
        "entry_z": [1.5, 2.0],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        lookback = self.parameters["lookback"]
        entry_z = self.parameters["entry_z"]

        if not isinstance(lookback, int):
            raise TypeError("lookback must be an integer.")

        if lookback < 3:
            raise ValueError("lookback must be at least 3.")

        if not isinstance(entry_z, (int, float)):
            raise TypeError("entry_z must be numeric.")

        if entry_z <= 0:
            raise ValueError("entry_z must be positive.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        if "volume" not in data.columns:
            raise ValueError(
                "VWAP deviation strategy input is missing 'volume'. This "
                "column is present on every instrument's OHLCV bars "
                "fetched via IBKR -- see "
                "src.data.retrieval.fetch_instrument_history."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        lookback = self.parameters["lookback"]
        entry_z = float(self.parameters["entry_z"])

        result = data.copy()

        rolling_vwap = vwap(
            result["high"],
            result["low"],
            result["close"],
            result["volume"],
            lookback,
        )
        result["rolling_vwap"] = rolling_vwap

        deviation = result["close"] - rolling_vwap

        rolling_std = deviation.rolling(
            window=lookback,
            min_periods=lookback,
        ).std(ddof=0)

        valid_std = rolling_std.where(rolling_std > 0)

        result["z_score"] = deviation / valid_std

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        z_values = result["z_score"].to_numpy()

        for index, z_score in enumerate(z_values):
            if not np.isfinite(z_score):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if z_score <= -entry_z:
                    current_position = 1
                elif z_score >= entry_z:
                    current_position = -1

            elif current_position == 1:
                if z_score >= 0:
                    current_position = 0

            elif current_position == -1:
                if z_score <= 0:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                result["z_score"] <= -entry_z,
                result["z_score"] >= entry_z,
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
