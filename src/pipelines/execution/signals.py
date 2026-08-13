from __future__ import annotations

import pandas as pd

from src.factory.candidate import CandidateSpec


def compute_target_position(
    candidate: CandidateSpec,
    market_data: pd.DataFrame,
) -> int:
    """
    Compute the current target position for one deployed candidate.

    CRITICAL: this MUST call src.factory.registry.create_strategy() to
    instantiate the strategy and then call its generate_positions(data)
    method -- the exact same code path already used in backtesting (see
    src.backtest.engine.run_candidate_backtest and
    src.strategies.base.BaseStrategy). This function must NEVER
    reimplement strategy logic independently: doing so would let live
    behavior silently diverge from what was actually validated in the
    alpha factory, defeating the entire point of keeping strategy code
    shared across research and execution.

    Raises:
        NotImplementedError: always -- this is an unimplemented stub.
    """
    raise NotImplementedError(
        "Target-position computation is not yet implemented."
    )
