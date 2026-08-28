from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class CarryZScoreStrategy(BaseStrategy):
    """
    Stateful FX carry strategy based on a rolling z-score of the
    base-vs-quote short-term policy-rate differential against its own
    recent history, rather than an absolute level.

    Behavioral hypothesis
    ---------------------
    Different currency pairs carry very different typical differential
    magnitudes, so a fixed absolute threshold (as in the static carry
    strategy) treats them unevenly; standardizing the differential by its
    own rolling mean and volatility instead captures when a pair's carry
    is unusually wide or narrow relative to its own recent history -- a
    relative-carry read that should generalize more consistently across
    pairs.

    Free parameters
    ---------------
    lookback
        Rolling window used to estimate the local mean and volatility of
        the rate differential.
    entry_z
        Absolute z-score required to enter a contrarian-carry position.

    Fixed design choices
    --------------------
    - lookback is expressed in calendar-day units (24 bars/day), matching
      carry_rate_momentum, since the differential only moves on a
      policy-cycle cadence (see that strategy's docstring).
    - entry_z is set looser than the price-based z-score strategies: the
      differential's own rolling distribution is far smoother/less noisy
      than a price series, so a smaller z-score already reflects a
      meaningful relative extreme.
    - Enter long when z <= -entry_z.
    - Enter short when z >= entry_z.
    - Exit a long position when z >= 0.
    - Exit a short position when z <= 0.
    - Hold the current position between entry and exit.
    - Because the differential changes slowly, long lookbacks combined
      with this hold-until-exit state machine may produce very few state
      changes across a multi-year sample -- an honest consequence of this
      strategy's premise, not a bug.
    - Depends on an interest_rate_differential column already being
      present on the input data (CASH/FX instruments only).
    """

    family_name = "carry_zscore"
    parameter_names = ("lookback", "entry_z")
    parameter_grid = {
        "lookback": [720, 1440, 2160],
        "entry_z": [1.0, 1.5],
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

        lookback = self.parameters["lookback"]
        entry_z = float(self.parameters["entry_z"])

        result = data.copy()

        differential = result["interest_rate_differential"]

        result["rolling_mean"] = differential.rolling(
            window=lookback,
            min_periods=lookback,
        ).mean()

        result["rolling_std"] = differential.rolling(
            window=lookback,
            min_periods=lookback,
        ).std(ddof=0)

        valid_std = result["rolling_std"].where(
            result["rolling_std"] > 0
        )

        result["z_score"] = (
            differential - result["rolling_mean"]
        ) / valid_std

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
