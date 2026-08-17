from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_deployed_candidate_ids(
    strategies_config_path: str | Path,
) -> list[str]:
    """
    Read the current deployed_strategies list, preserving file order.
    """
    strategies_config_path = Path(strategies_config_path)

    with strategies_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return list(config.get("deployed_strategies") or [])


def find_new_survivors(
    final_survivors_path: str | Path,
    strategies_config_path: str | Path,
) -> list[str]:
    """
    Find candidate_ids that passed the gate but are not yet deployed.
    """
    survivors = pd.read_csv(final_survivors_path)
    survivor_ids = set(survivors["candidate_id"])

    deployed_ids = set(
        load_deployed_candidate_ids(strategies_config_path)
    )

    return sorted(survivor_ids - deployed_ids)


def write_deployed_candidate_ids(
    strategies_config_path: str | Path,
    candidate_ids: list[str],
) -> None:
    """
    Rewrite the deployed_strategies list, preserving the rest of the file.

    Everything before the `deployed_strategies:` key (the header comment) is
    kept verbatim -- only the list itself is regenerated, in the given
    order.
    """
    strategies_config_path = Path(strategies_config_path)

    original_text = strategies_config_path.read_text(encoding="utf-8")
    header, separator, _ = original_text.partition("deployed_strategies:")

    if not separator:
        raise ValueError(
            f"{strategies_config_path} has no 'deployed_strategies:' key "
            "to update."
        )

    if candidate_ids:
        body = "deployed_strategies:\n" + "\n".join(
            f'  - "{candidate_id}"'
            for candidate_id in candidate_ids
        )
    else:
        body = "deployed_strategies: []"

    strategies_config_path.write_text(
        header + body + "\n",
        encoding="utf-8",
    )


def promote_survivors(
    final_survivors_path: str | Path,
    strategies_config_path: str | Path,
    write: bool = False,
) -> list[str]:
    """
    Find newly-passed candidates and, if requested, append them.

    Existing deployed_strategies entries keep their original order; new
    entries are appended at the end, sorted for determinism.
    """
    new_ids = find_new_survivors(
        final_survivors_path,
        strategies_config_path,
    )

    if write and new_ids:
        current_ids = load_deployed_candidate_ids(
            strategies_config_path
        )

        write_deployed_candidate_ids(
            strategies_config_path,
            current_ids + new_ids,
        )

    return new_ids


def main() -> None:
    """
    CLI: python -m src.factory.promotion [--write]
    """
    project_root = Path(__file__).resolve().parents[2]
    final_survivors_path = (
        project_root / "results" / "validation" / "final_survivors.csv"
    )
    strategies_config_path = project_root / "config" / "strategies.yml"

    parser = argparse.ArgumentParser(
        description=(
            "Report (and optionally append) candidates that passed the "
            "validation gate but are not yet listed in "
            "config/strategies.yml."
        )
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Append newly-found candidates to config/strategies.yml.",
    )
    args = parser.parse_args()

    new_ids = promote_survivors(
        final_survivors_path,
        strategies_config_path,
        write=args.write,
    )

    if not new_ids:
        print("No new survivors to promote.")
        return

    verb = "Added" if args.write else "Found"
    print(f"{verb} {len(new_ids)} new survivor(s) not yet deployed:")

    for candidate_id in new_ids:
        print(f"  + {candidate_id}")

    if not args.write:
        print(
            "\nRe-run with --write to add these to config/strategies.yml"
        )


if __name__ == "__main__":
    main()
