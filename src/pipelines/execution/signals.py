from __future__ import annotations

import pandas as pd

from src.factory.candidate import CandidateSpec
from src.factory.registry import create_strategy


def compute_target_position(
    candidate: CandidateSpec,
    market_data: pd.DataFrame,
) -> int:
    """
    Compute the current target position for one deployed candidate.

    Reuses create_strategy() + BaseStrategy.generate_positions() -- the
    exact code path already used in backtesting (see
    src.backtest.engine.run_candidate_backtest) -- and returns the
    target_position from market_data's most recent (completed) bar. Never
    reimplement strategy logic independently here: doing so would let live
    behavior silently diverge from what was actually validated in the
    alpha factory, defeating the entire point of keeping strategy code
    shared across research and execution.

    generate_positions() validates market_data's required schema itself via
    BaseStrategy.validate_input_data().
    """
    strategy = create_strategy(
        family=candidate.family,
        parameters=candidate.parameters,
    )

    strategy_output = strategy.generate_positions(market_data)

    return int(strategy_output["target_position"].iloc[-1])
