from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.factory.candidate import CandidateSpec


@dataclass
class BacktestResult:
    """
    Complete output of one candidate backtest.
    """

    candidate: CandidateSpec
    timeseries: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]

    def metrics_record(self) -> dict[str, Any]:
        """
        Return a flat record suitable for a results table.
        """
        return {
            **self.candidate.to_dict(),
            **self.metrics,
        }