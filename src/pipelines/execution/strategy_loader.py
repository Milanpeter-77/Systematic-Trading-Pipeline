from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.factory.candidate import CandidateSpec, candidate_from_record
from src.factory.promotion import load_deployed_candidate_ids


def load_deployed_strategies(
    strategies_config_path: str | Path,
    final_survivors_path: str | Path,
    candidates_directory: str | Path,
) -> list[CandidateSpec]:
    """
    Load and reconstruct every strategy promoted to live execution.

    Reads the `deployed_strategies` list of candidate_id strings from
    strategies_config_path, verifies each id is present in
    final_survivors_path (confirming it actually passed the validation
    gate), then reconstructs each as a CandidateSpec from its row in
    candidates_directory/<family>.csv -- never from final_survivors_path
    directly, which carries ~50 extra metric/pass-flag columns that
    candidate_from_record() would otherwise misinterpret as strategy
    parameters.

    Raises:
        ValueError: if any deployed candidate_id is not present in
            final_survivors_path.
    """
    candidates_directory = Path(candidates_directory)

    deployed_ids = load_deployed_candidate_ids(strategies_config_path)

    if not deployed_ids:
        return []

    final_survivors = pd.read_csv(final_survivors_path)
    survivors_by_id = final_survivors.set_index("candidate_id")

    missing_ids = sorted(
        set(deployed_ids) - set(survivors_by_id.index)
    )

    if missing_ids:
        raise ValueError(
            "deployed_strategies contains candidate_id(s) not present in "
            f"{final_survivors_path} (not confirmed to have passed the "
            f"validation gate): {missing_ids}"
        )

    family_frames: dict[str, pd.DataFrame] = {}
    candidates: list[CandidateSpec] = []

    for candidate_id in deployed_ids:
        family = survivors_by_id.loc[candidate_id, "family"]

        if family not in family_frames:
            family_frames[family] = pd.read_csv(
                candidates_directory / f"{family}.csv"
            ).set_index("candidate_id", drop=False)

        record = family_frames[family].loc[candidate_id].to_dict()
        candidates.append(candidate_from_record(record))

    return candidates
