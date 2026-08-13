from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract interface shared by every strategy family.

    Strategies generate target positions using information available at the
    close of each bar. They do not apply execution lags, transaction costs,
    or portfolio-return calculations.
    """

    family_name: str
    parameter_names: tuple[str, ...]

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """
        Validate the supplied parameter names.

        Family-specific numerical restrictions should be implemented by
        overriding this method and calling super().
        """
        supplied = set(self.parameters)
        expected = set(self.parameter_names)

        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)

            raise ValueError(
                f"Invalid parameters for {self.family_name}. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )

    @abstractmethod
    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate indicators, signals, and target positions.

        The returned target_position is the desired position after observing
        the completed bar. The backtester must lag this target before applying
        it to returns.
        """
        raise NotImplementedError

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> None:
        """
        Validate the minimum market-data inputs required by the strategies.
        """
        required_columns = {
            "open",
            "high",
            "low",
            "close",
            "coverage_ratio",
        }

        missing = required_columns.difference(data.columns)

        if missing:
            raise ValueError(
                f"Strategy input is missing columns: {sorted(missing)}"
            )

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                "Strategy data must use a DatetimeIndex."
            )

        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "Strategy data must be chronologically sorted."
            )

        if data.index.has_duplicates:
            raise ValueError(
                "Strategy data contains duplicate timestamps."
            )