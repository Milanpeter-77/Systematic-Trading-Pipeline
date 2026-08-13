from __future__ import annotations

import numpy as np
import pandas as pd


def build_executed_positions(
    strategy_output: pd.DataFrame,
    minimum_signal_coverage: float = 0.80,
    minimum_fill_coverage: float = 0.80,
    initial_position: int = 0,
) -> pd.DataFrame:
    """
    Convert completed-bar target positions into next-bar executed positions.

    A target produced on bar t can be executed at the open of bar t+1 only if:

    1. the signal bar has sufficient coverage;
    2. the fill bar has sufficient coverage;
    3. a valid next bar exists.

    If execution is not permitted, the previous executed position is carried.
    """
    required_columns = {
        "target_position",
        "coverage_ratio",
        "open",
    }

    missing = required_columns.difference(
        strategy_output.columns
    )

    if missing:
        raise ValueError(
            f"Execution input is missing columns: {sorted(missing)}"
        )

    if initial_position not in {-1, 0, 1}:
        raise ValueError(
            "initial_position must be -1, 0, or 1."
        )

    if not 0 < minimum_signal_coverage <= 1:
        raise ValueError(
            "minimum_signal_coverage must be in (0, 1]."
        )

    if not 0 < minimum_fill_coverage <= 1:
        raise ValueError(
            "minimum_fill_coverage must be in (0, 1]."
        )

    result = strategy_output.copy()

    result["signal_bar_eligible"] = (
        result["coverage_ratio"] >= minimum_signal_coverage
    ).astype(bool)

    result["fill_bar_eligible"] = (
        result["coverage_ratio"] >= minimum_fill_coverage
    ).astype(bool)

    result["delayed_target"] = (
        result["target_position"].shift(1)
    )

    result["delayed_signal_eligible"] = (
        result["signal_bar_eligible"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    executed_positions = np.zeros(
        len(result),
        dtype=np.int8,
    )

    execution_allowed = np.zeros(
        len(result),
        dtype=bool,
    )

    previous_position = initial_position

    delayed_targets = result["delayed_target"].to_numpy()
    delayed_signal_valid = (
        result["delayed_signal_eligible"].to_numpy()
    )
    fill_valid = result["fill_bar_eligible"].to_numpy()

    for index in range(len(result)):
        target = delayed_targets[index]

        can_execute = (
            delayed_signal_valid[index]
            and fill_valid[index]
            and np.isfinite(target)
        )

        if can_execute:
            previous_position = int(target)
            execution_allowed[index] = True

        executed_positions[index] = previous_position

    result["execution_allowed"] = execution_allowed
    result["executed_position"] = executed_positions

    result["position_change"] = (
        result["executed_position"]
        .diff()
        .fillna(
            result["executed_position"] - initial_position
        )
    )

    result["trade_executed"] = (
        result["position_change"] != 0
    )

    return result