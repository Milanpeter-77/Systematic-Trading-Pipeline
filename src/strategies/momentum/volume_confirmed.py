from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MomentumVolumeConfirmedStrategy(BaseStrategy):
    """
    Time-series momentum strategy based on a fixed-lookback total return,
    only signaling when volume confirms the move.

    Behavioral hypothesis
    ---------------------
    Assets that have risen (fallen) over the recent past tend to keep
    rising (falling), because information diffuses gradually and market
    participants underreact to it -- the same hypothesis as the plain
    fixed-lookback momentum strategy. This variant adds the idea that a
    price move accompanied by unusually heavy volume better reflects
    genuine information-driven participation than the same move on thin
    volume, which is more likely to be noise or a temporary imbalance.

    Free parameters
    ---------------
    lookback
        Number of bars over which the total return and average volume are
        measured.
    return_threshold
        Minimum absolute total return required to take a position.
    volume_ratio
        Minimum ratio of current volume to its own rolling average
        required to confirm the move.

    Fixed design choices
    --------------------
    - momentum_return = close.pct_change(lookback).
    - volume_confirmed = volume > volume_ratio * rolling_mean(volume, lookback).
    - Long when momentum_return exceeds return_threshold AND volume is
      confirmed.
    - Short when momentum_return is below -return_threshold AND volume is
      confirmed.
    - Flat otherwise -- in particular, momentum without volume
      confirmation goes flat rather than being traded as a weaker signal:
      unconfirmed momentum is treated as noise, not a valid but smaller
      edge.
    - Stateless: the position is recomputed fresh every bar, like the
      other momentum strategies.
    - Depends on a volume column already being present on the input data
      (present on every instrument's OHLCV bars today) -- this is not
      something the strategy computes itself.
    """

    family_name = "momentum_volume_confirmed"
    parameter_names = ("lookback", "return_threshold", "volume_ratio")
    parameter_grid = {
        "lookback": [24, 48, 96],
        "return_threshold": [0.0, 0.01, 0.02],
        "volume_ratio": [1.0, 1.5, 2.0],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        lookback = self.parameters["lookback"]
        return_threshold = self.parameters["return_threshold"]
        volume_ratio = self.parameters["volume_ratio"]

        if not isinstance(lookback, int):
            raise TypeError("lookback must be an integer.")

        if lookback < 2:
            raise ValueError("lookback must be at least 2.")

        if not isinstance(return_threshold, (int, float)):
            raise TypeError("return_threshold must be numeric.")

        if return_threshold < 0:
            raise ValueError("return_threshold must be non-negative.")

        if not isinstance(volume_ratio, (int, float)):
            raise TypeError("volume_ratio must be numeric.")

        if volume_ratio <= 0:
            raise ValueError("volume_ratio must be positive.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        if "volume" not in data.columns:
            raise ValueError(
                "Volume-confirmed momentum strategy input is missing "
                "'volume'. This column is present on every instrument's "
                "OHLCV bars fetched via IBKR -- see "
                "src.data.retrieval.fetch_instrument_history."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        lookback = self.parameters["lookback"]
        return_threshold = float(self.parameters["return_threshold"])
        volume_ratio = float(self.parameters["volume_ratio"])

        result = data.copy()

        result["momentum_return"] = result["close"].pct_change(
            periods=lookback,
        )

        result["avg_volume"] = result["volume"].rolling(
            window=lookback,
            min_periods=lookback,
        ).mean()

        volume_confirmed = result["volume"] > (
            volume_ratio * result["avg_volume"]
        )

        indicator_available = (
            result["momentum_return"].notna() & result["avg_volume"].notna()
        )

        result["raw_signal"] = np.select(
            [
                indicator_available
                & volume_confirmed
                & (result["momentum_return"] > return_threshold),
                indicator_available
                & volume_confirmed
                & (result["momentum_return"] < -return_threshold),
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = result["raw_signal"].astype("int8")

        return result
