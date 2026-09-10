from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class StatArbCorrelationBreakdownStrategy(BaseStrategy):
    """
    Stateful pairs-trading strategy that enters when a cointegrated pair's
    rolling return correlation breaks down unusually far below its own
    recent norm, betting on the relationship normalizing.

    Behavioral hypothesis
    ---------------------
    Two instruments whose prices are cointegrated (src.data.features.
    statistical.screen_cointegrated_pairs) share a long-run equilibrium
    relationship, driven in large part by co-movement in their returns.
    A sharp, unusual drop in that co-movement signals the relationship is
    under temporary stress (e.g. an idiosyncratic shock to one leg) rather
    than a permanent regime change, and tends to normalize -- distinct
    from stat_arb's other members, which time entry off the spread's own
    level or oscillator reading; this one times entry off the legs'
    relationship itself breaking down.

    Free parameters
    ---------------
    corr_window
        Rolling window used to compute the two legs' return correlation.
    entry_z
        Absolute z-score (of the correlation, against its own longer-run
        mean/std) required to treat a correlation drop as a genuine
        breakdown.
    exit_z
        Correlation z-score at which an open position is closed. Set
        below entry_z to exit before correlation fully normalizes (a
        partial-normalization exit); exit_z=0 reproduces exiting once
        correlation is back at or above its own recent mean.

    Fixed design choices
    --------------------
    - z_window (used to characterize correlation's own "normal" range) is
      fixed internally at 4 * corr_window, not a free parameter -- long
      enough to establish a stable baseline for a quantity that is itself
      already a rolling statistic.
    - corr_z = (rolling_corr - rolling_corr's own rolling mean) /
      rolling_corr's own rolling std, the same z-score construction used
      throughout this codebase, with the same zero-denominator guard.
    - Correlation breakdown alone has no natural long/short interpretation
      -- direction instead comes from the precomputed static spread
      (this pseudo-instrument's own close column, from
      src.data.features.statistical.compute_pair_spread): enter long when
      a breakdown coincides with the spread trading below its own rolling
      mean, enter short when it coincides with the spread trading above
      its rolling mean. A breakdown with the spread exactly at its mean
      produces no entry (no directional edge to trade).
    - Exit when corr_z rises back to or above -exit_z (correlation has
      normalized), regardless of position direction, since normalization
      -- not spread level -- is what closes this trade.
    - Hold the current position between entry and exit.
    - Depends on leg_a_close/leg_b_close being present on the input data
      (added to the pair pseudo-instrument by src.pipelines.alpha_factory.
      pipeline.prepare_pair_features specifically for this and
      stat_arb_rolling_hedge_zscore) in addition to the static spread's
      close column that every other stat_arb strategy already reads.
    - No stop-loss or holding-period parameter.
    """

    family_name = "stat_arb_correlation_breakdown"
    parameter_names = ("corr_window", "entry_z", "exit_z")
    parameter_grid = {
        "corr_window": [48, 96, 168],
        "entry_z": [1.5, 2.0],
        "exit_z": [0.0, 0.5],
    }
    enabled = True

    def validate_parameters(self) -> None:
        super().validate_parameters()

        corr_window = self.parameters["corr_window"]
        entry_z = self.parameters["entry_z"]
        exit_z = self.parameters["exit_z"]

        if not isinstance(corr_window, int):
            raise TypeError("corr_window must be an integer.")

        if corr_window < 3:
            raise ValueError("corr_window must be at least 3.")

        if not isinstance(entry_z, (int, float)):
            raise TypeError("entry_z must be numeric.")

        if entry_z <= 0:
            raise ValueError("entry_z must be positive.")

        if not isinstance(exit_z, (int, float)):
            raise TypeError("exit_z must be numeric.")

        if not 0 <= exit_z < entry_z:
            raise ValueError("exit_z must satisfy 0 <= exit_z < entry_z.")

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        BaseStrategy.validate_input_data(data)

        missing = {"leg_a_close", "leg_b_close"} - set(data.columns)

        if missing:
            raise ValueError(
                "Correlation-breakdown stat_arb strategy input is missing "
                f"{sorted(missing)}. These columns are added to the pair "
                "pseudo-instrument during alpha-factory feature prep -- "
                "see src.pipelines.alpha_factory.pipeline."
                "prepare_pair_features."
            )

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input_data(data)

        corr_window = self.parameters["corr_window"]
        entry_z = float(self.parameters["entry_z"])
        exit_z = float(self.parameters["exit_z"])
        z_window = corr_window * 4

        result = data.copy()

        leg_a_return = result["leg_a_close"].pct_change()
        leg_b_return = result["leg_b_close"].pct_change()

        rolling_corr = leg_a_return.rolling(
            window=corr_window,
            min_periods=corr_window,
        ).corr(leg_b_return)

        corr_mean = rolling_corr.rolling(
            window=z_window,
            min_periods=z_window,
        ).mean()

        corr_std = rolling_corr.rolling(
            window=z_window,
            min_periods=z_window,
        ).std(ddof=0)

        valid_corr_std = corr_std.where(corr_std > 0)

        corr_z = (rolling_corr - corr_mean) / valid_corr_std

        spread_mean = result["close"].rolling(
            window=corr_window,
            min_periods=corr_window,
        ).mean()

        result["rolling_corr"] = rolling_corr
        result["corr_z"] = corr_z

        breakdown = corr_z <= -entry_z
        dislocated_below = result["close"] < spread_mean
        dislocated_above = result["close"] > spread_mean

        target_positions = np.zeros(
            len(result),
            dtype=np.int8,
        )

        current_position = 0

        corr_z_values = corr_z.to_numpy()
        close_values = result["close"].to_numpy()
        spread_mean_values = spread_mean.to_numpy()

        for index in range(len(result)):
            z_score = corr_z_values[index]
            mean_level = spread_mean_values[index]

            if not np.isfinite(z_score) or not np.isfinite(mean_level):
                current_position = 0
                target_positions[index] = current_position
                continue

            if current_position == 0:
                if z_score <= -entry_z:
                    if close_values[index] < mean_level:
                        current_position = 1
                    elif close_values[index] > mean_level:
                        current_position = -1

            elif current_position == 1:
                if z_score >= -exit_z:
                    current_position = 0

            elif current_position == -1:
                if z_score >= -exit_z:
                    current_position = 0

            target_positions[index] = current_position

        result["raw_signal"] = np.select(
            [
                breakdown & dislocated_below,
                breakdown & dislocated_above,
            ],
            [1, -1],
            default=0,
        ).astype("int8")

        result["target_position"] = target_positions

        return result
