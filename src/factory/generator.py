from __future__ import annotations

from itertools import product

import pandas as pd

from src.factory.candidate import CandidateSpec
from src.factory.registry import STRATEGY_REGISTRY


def expand_parameter_grid(
    parameter_grid: dict[str, list[int | float]],
) -> list[dict[str, int | float]]:
    """
    Expand a dictionary of parameter values into a Cartesian product.
    """
    if not parameter_grid:
        raise ValueError("Parameter grid cannot be empty.")

    parameter_names = list(parameter_grid)

    parameter_values = [
        parameter_grid[name]
        for name in parameter_names
    ]

    if any(not values for values in parameter_values):
        raise ValueError(
            "Every strategy parameter must contain at least one value."
        )

    return [
        dict(zip(parameter_names, combination))
        for combination in product(*parameter_values)
    ]


def generate_candidates(
    symbols: list[str],
    family_symbol_overrides: dict[str, list[str]] | None = None,
) -> list[CandidateSpec]:
    """
    Expand every enabled registered strategy across its parameter grid and symbols.

    family_symbol_overrides lets a family use its own symbol universe
    instead of `symbols` -- e.g. stat_arb trades precomputed pair-spread
    pseudo-instruments, not the same tradable instruments every other
    family crosses with `symbols`. Families not present in the override
    dict (or when it's None) use `symbols` as before.
    """
    if not symbols:
        raise ValueError("At least one symbol is required.")

    overrides = family_symbol_overrides or {}

    candidates: list[CandidateSpec] = []

    for family, strategy_class in STRATEGY_REGISTRY.items():
        if not strategy_class.enabled:
            continue

        family_symbols = overrides.get(family, symbols)

        if not family_symbols:
            continue

        parameter_combinations = expand_parameter_grid(
            strategy_class.parameter_grid
        )

        for parameters in parameter_combinations:
            # Instantiation validates parameter names and numerical rules.
            strategy_class(**parameters)

            for symbol in family_symbols:
                candidate = CandidateSpec(
                    family=family,
                    symbol=symbol,
                    parameters=parameters.copy(),
                )

                candidates.append(candidate)

    candidate_ids = [
        candidate.candidate_id
        for candidate in candidates
    ]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(
            "Candidate generation produced duplicate candidate IDs."
        )

    return candidates


def candidates_to_frame(
    candidates: list[CandidateSpec],
) -> pd.DataFrame:
    """
    Convert candidate specifications into a tabular registry.
    """
    records = [
        candidate.to_dict()
        for candidate in candidates
    ]

    candidates_frame = pd.DataFrame(records)

    if not candidates_frame.empty:
        candidates_frame = candidates_frame.sort_values(
            ["family", "symbol", "candidate_id"]
        ).reset_index(drop=True)

    return candidates_frame


def candidates_to_frames_by_family(
    candidates: list[CandidateSpec],
) -> dict[str, pd.DataFrame]:
    """
    Convert candidate specifications into one tabular registry per family.

    Each family has its own parameter names, so a single combined table
    would carry parameter columns that are meaningless (NaN) for every
    other family. Splitting by family keeps each table limited to that
    family's own columns.
    """
    candidates_frame = candidates_to_frame(candidates)

    frames_by_family: dict[str, pd.DataFrame] = {}

    for family, group in candidates_frame.groupby("family"):
        family_frame = group.dropna(axis="columns", how="all").reset_index(
            drop=True
        )

        for column in family_frame.columns:
            if column in {"candidate_id", "family", "symbol"}:
                continue

            # Columns with NaN elsewhere (other families) are upcast to
            # float by the combined frame -- restore int for parameters
            # whose grid only ever holds whole numbers.
            if (
                pd.api.types.is_float_dtype(family_frame[column])
                and (family_frame[column] % 1 == 0).all()
            ):
                family_frame[column] = family_frame[column].astype(int)

        frames_by_family[family] = family_frame

    return frames_by_family