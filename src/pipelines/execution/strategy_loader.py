from __future__ import annotations

from pathlib import Path

from src.factory.candidate import CandidateSpec


def load_deployed_strategies(config_path: Path) -> list[CandidateSpec]:
    """
    Load and reconstruct every strategy promoted to live execution.

    Intended behavior once implemented: read the `deployed_strategies` list
    of candidate_id strings from config/deployed_strategies.yml, verify each
    id is present in results/gate/final_survivors.csv (to confirm it
    actually passed the validation gate), then reconstruct each as a
    CandidateSpec via src.factory.candidate.candidate_from_record().

    OPEN CAVEAT: candidate_from_record() expects a record containing only
    candidate_id/family/symbol/parameter columns. results/gate/
    final_survivors.csv also carries ~50 extra metric/pass-flag columns
    (layer1_pass, oos_net_sharpe, etc.), which candidate_from_record() would
    otherwise misinterpret as strategy parameters. The eventual
    implementation should look up each promoted candidate_id's row in
    results/candidates.csv (which has the clean schema) for reconstruction,
    and use final_survivors.csv only to confirm gate-pass membership.

    Raises:
        NotImplementedError: always -- this is an unimplemented stub.
    """
    raise NotImplementedError(
        "Deployed-strategy loading is not yet implemented."
    )
