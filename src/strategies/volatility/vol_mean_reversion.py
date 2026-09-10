from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class VolatilityMeanReversionStrategy(BaseStrategy):
    """
    Stateful strategy that fades the price move responsible for an
    unusual spike in this instrument's own realized volatility.

    Behavioral hypothesis
    ---------------------
    Volatility itself tends to mean-revert: a sharp spike in realized
    volatility relative to its own recent history is more often a
    temporary overreaction (forced trading, a liquidity shock, a
    short-lived news event) than the start of a sustained new regime, and
    the price move that produced the spike tends to partially retrace
    once volatility normalizes -- distinct from mean_reversion's members,
    which time entry/exit off price's own z-score rather than off
    volatility's.

    Free parameters
    ---------------
    short_window
        Rolling window used to estimate this instrument's own realized
        volatility.
    entry_z
        Absolute z-score of realized volatility (against its own longer-
        run mean/std) required to treat a vol spike as unusual enough to
        trade.
    exit_z
        Volatility z-score at which an open position is closed. Set below
        entry_z to exit before volatility fully normalizes (a partial-
        normalization exit); exit_z=0 reproduces exiting once volatility
        is back at or below its own recent mean.

    Fixed design choices
    --------------------
    - long_window (used to characterize volatility's own "normal" range)
      is fixed internally at 4 * short_window, not a free parameter --
      the same fixed-multiple pattern stat_arb_correlation_breakdown uses
      for its own z-window.
    - realized_vol = rolling_std(close.pct_change(), short_window).
    - vol_z = (realized_vol - realized_vol's own rolling mean) /
      realized_vol's own rolling std, the same z-score construction used
      throughout this codebase, with the same zero-denominator guard.
    - price_direction = sign(close - close.shift(short_window)) -- the
      direction of the move that produced the vol spike.
    - Enter short when vol_z >= entry_z and price_direction is up (fade
      the up-move that spiked vol).
    - Enter long when vol_z >= entry_z and price_direction is down (fade
      the down-move).
    - No entry when vol_z >= entry_z but price_direction is exactly flat
      -- there is no move to fade.
    - Exit (either direction) when vol_z <= exit_z (volatility has
      normalized), regardless of position direction.
    - Hold the current position between entry and exit.
    - No stop-loss or holding-period parameter.
    """

    family_name = "volatility_mean_reversion"
    parameter_names = ("short_window", "entry_z", "exit_z")
    parameter_grid = {
        "short_window": [12, 24, 48],
        "entry_z": [1.5, 2.0],
        "exit_z": [0.0, 0.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        short_window = self.parameters["short_window"]
        entry_z = self.parameters["entry_z"]
        exit_z = self.parameters["exit_z"]

        if not isinstance(short_window, int):
            raise TypeError("short_window must be an integer.")

        if short_window < 3:
            raise ValueError("short_window must be at least 3.")

        if not isinstance(entry_z, (int, float)):
            raise TypeError("entry_z must be numeric.")

        if entry_z <= 0:
            raise ValueError("entry_z must be positive.")

        if not isinstance(exit_z, (int, float)):
            raise TypeError("exit_z must be numeric.")

        if not 0 <= exit_z < entry_z:
            raise ValueError("exit_z must satisfy 0 <= exit_z < entry_z.")

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        short_window = self.parameters["short_window"]
        entry_z = float(self.parameters["entry_z"])
        exit_z = float(self.parameters["exit_z"])
        long_window = short_window * 4

        result = data.copy()

        period_return = result["close"].pct_change()

        realized_vol = period_return.rolling(
            window=short_window,
            min_periods=short_window,
        ).std(ddof=0)

        vol_mean = realized_vol.rolling(
            window=long_window,
            min_periods=long_window,
        ).mean()

        vol_std = realized_vol.rolling(
            window=long_window,
            min_periods=long_window,
        ).std(ddof=0)

        valid_vol_std = vol_std.where(vol_std > 0)

        vol_z = (realized_vol - vol_mean) / valid_vol_std
        price_direction = np.sign(
            result["close"] - result["close"].shift(short_window)
        )

        result["realized_vol"] = realized_vol
        result["vol_z"] = vol_z

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        vol_z_values = vol_z.to_numpy()
        direction_values = price_direction.to_numpy()

        for index in range(len(result)):
            z_score = vol_z_values[index]
            direction = direction_values[index]

            if not np.isfinite(z_score) or not np.isfinite(direction):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if z_score >= entry_z:
                    if direction > 0:
                        current_position = -1
                    elif direction < 0:
                        current_position = 1

            elif current_position == 1:
                if z_score <= exit_z:
                    current_position = 0

            elif current_position == -1:
                if z_score <= exit_z:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                (vol_z >= entry_z) & (price_direction > 0),
                (vol_z >= entry_z) & (price_direction < 0),
            ],
            [-1, 1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
