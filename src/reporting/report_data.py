from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ReportData:
    candidate_metrics: pd.DataFrame
    layer1_results: pd.DataFrame
    layer2_results: pd.DataFrame
    layer3_results: pd.DataFrame
    layer4_results: pd.DataFrame
    final_funnel: pd.DataFrame
    final_survivors: pd.DataFrame


def load_report_data(
    project_root: str | Path,
) -> ReportData:
    """
    Load all result tables required by the reporting layer.
    """
    project_root = Path(project_root)

    backtest_directory = (
        project_root / "results" / "backtests"
    )

    validation_directory = (
        project_root / "results" / "validation"
    )

    required_paths = {
        "candidate_metrics": (
            backtest_directory
            / "candidate_metrics.csv"
        ),
        "layer1_results": (
            validation_directory
            / "layer1_results.csv"
        ),
        "layer2_results": (
            validation_directory
            / "layer2_results.csv"
        ),
        "layer3_results": (
            validation_directory
            / "layer3_results.csv"
        ),
        "layer4_results": (
            validation_directory
            / "layer4_results.csv"
        ),
        "final_funnel": (
            validation_directory
            / "final_funnel.csv"
        ),
        "final_survivors": (
            validation_directory
            / "final_survivors.csv"
        ),
    }

    missing_paths = [
        path
        for path in required_paths.values()
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing reporting inputs:\n"
            + "\n".join(
                str(path)
                for path in missing_paths
            )
        )

    return ReportData(
        candidate_metrics=pd.read_csv(
            required_paths[
                "candidate_metrics"
            ]
        ),
        layer1_results=pd.read_csv(
            required_paths[
                "layer1_results"
            ]
        ),
        layer2_results=pd.read_csv(
            required_paths[
                "layer2_results"
            ]
        ),
        layer3_results=pd.read_csv(
            required_paths[
                "layer3_results"
            ]
        ),
        layer4_results=pd.read_csv(
            required_paths[
                "layer4_results"
            ]
        ),
        final_funnel=pd.read_csv(
            required_paths[
                "final_funnel"
            ]
        ),
        final_survivors=pd.read_csv(
            required_paths[
                "final_survivors"
            ]
        ),
    )