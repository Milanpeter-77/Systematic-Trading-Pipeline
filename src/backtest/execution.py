from __future__ import annotations

import numpy as np
import pandas as pd


def build_executed_positions(
    strategy_output: pd.DataFrame,
    initial_position: int = 0,
) -> pd.DataFrame:
    """
    Convert completed-bar target positions into next-bar executed positions.

    A target produced on bar t can be executed at the open of bar t+1 only
    if a valid next bar exists. If execution is not permitted (the first
    bar, where no prior target exists), the previous executed position is
    carried.
    """
    required_columns = {
        "target_position",
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

    result = strategy_output.copy()

    result["delayed_target"] = (
        result["target_position"].shift(1)
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

    for index in range(len(result)):
        target = delayed_targets[index]

        can_execute = np.isfinite(target)

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
