from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.factory.candidate import CandidateSpec
from src.factory.registry import create_strategy


def load_strategy_config(
    config_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Load the strategy-family configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Strategy configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not config:
        raise ValueError(
            f"Strategy configuration is empty or invalid: {config_path}"
        )

    return config


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
    strategy_config: dict[str, dict[str, Any]],
) -> list[CandidateSpec]:
    """
    Expand all enabled families across parameter grids and symbols.
    """
    if not symbols:
        raise ValueError("At least one symbol is required.")

    candidates: list[CandidateSpec] = []

    for family, family_config in strategy_config.items():
        if not family_config.get("enabled", True):
            continue

        parameter_grid = family_config.get("parameters")

        if not isinstance(parameter_grid, dict):
            raise ValueError(
                f"Family '{family}' has no valid parameter grid."
            )

        parameter_combinations = expand_parameter_grid(
            parameter_grid
        )

        for parameters in parameter_combinations:
            # Instantiation validates parameter names and numerical rules.
            create_strategy(
                family=family,
                parameters=parameters,
            )

            for symbol in symbols:
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