from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def format_parameter_value(value: Any) -> str:
    """
    Convert a parameter value into a stable candidate-ID component.
    """
    if isinstance(value, float):
        text = f"{value:g}"
        return text.replace(".", "p").replace("-", "m")

    return str(value)


@dataclass(frozen=True)
class CandidateSpec:
    """
    Complete and reproducible specification of one strategy candidate.
    """

    family: str
    symbol: str
    parameters: dict[str, int | float]

    @property
    def candidate_id(self) -> str:
        """
        Generate a stable identifier independent of dictionary insertion order.
        """
        parameter_text = "__".join(
            f"{name}-{format_parameter_value(value)}"
            for name, value in sorted(self.parameters.items())
        )

        return f"{self.family}__{self.symbol}__{parameter_text}"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the candidate specification into a flat serializable record.
        """
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "symbol": self.symbol,
            **self.parameters,
        }


def candidate_from_record(record: dict[str, Any]) -> CandidateSpec:
    """
    Reconstruct one CandidateSpec from a candidate-table record.

    Shared by the alpha factory (reconstructing candidates from
    results/candidates.csv or an in-memory candidate frame) and, eventually,
    the execution package (reconstructing a promoted candidate from
    config/deployed_strategies.yml). The record must contain only
    candidate_id/family/symbol/parameter columns -- results/gate/
    final_survivors.csv also carries ~50 extra metric/pass-flag columns
    (layer1_pass, oos_net_sharpe, etc.) that this function would otherwise
    misinterpret as strategy parameters, so reconstruction should look up
    each candidate's row in results/candidates.csv instead, and use
    final_survivors.csv only to confirm gate-pass membership.
    """
    parameters = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "candidate_id",
            "family",
            "symbol",
        }
        and pd.notna(value)
    }

    normalized_parameters: dict[
        str,
        Any,
    ] = {}

    for key, value in (
        parameters.items()
    ):
        if (
            isinstance(value, float)
            and value.is_integer()
        ):
            normalized_parameters[key] = (
                int(value)
            )
        else:
            normalized_parameters[key] = (
                value
            )

    candidate = CandidateSpec(
        family=record["family"],
        symbol=record["symbol"],
        parameters=normalized_parameters,
    )

    expected_id = record.get(
        "candidate_id"
    )

    if (
        expected_id is not None
        and candidate.candidate_id
        != expected_id
    ):
        raise RuntimeError(
            "Candidate reconstruction produced "
            f"{candidate.candidate_id}, expected "
            f"{expected_id}."
        )

    return candidate