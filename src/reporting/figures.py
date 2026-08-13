from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reporting.report_data import ReportData
from src.reporting.summaries import (
    build_cumulative_funnel,
    build_failure_summary,
)


def save_figure(
    figure: plt.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_validation_funnel(
    report_data: ReportData,
    output_path: str | Path,
) -> Path:
    funnel = build_cumulative_funnel(
        report_data.final_funnel
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.bar(
        funnel["stage"],
        funnel["survivors"],
    )

    axis.set_title(
        "Strategy Validation Funnel"
    )

    axis.set_xlabel(
        "Validation stage"
    )

    axis.set_ylabel(
        "Surviving candidates"
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    output_path = Path(
        output_path
    )

    save_figure(
        figure,
        output_path,
    )

    return output_path


def plot_independent_pass_rates(
    report_data: ReportData,
    output_path: str | Path,
) -> Path:
    pass_rates = pd.Series(
        {
            "Layer 1": (
                report_data.final_funnel[
                    "layer1_pass"
                ].mean()
            ),
            "Layer 2": (
                report_data.final_funnel[
                    "layer2_pass"
                ].mean()
            ),
            "Layer 3": (
                report_data.final_funnel[
                    "layer3_pass"
                ].mean()
            ),
            "Layer 4": (
                report_data.final_funnel[
                    "layer4_pass"
                ].mean()
            ),
        }
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        pass_rates.index,
        pass_rates.values,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_title(
        "Independent Pass Rate by Validation Layer"
    )

    axis.set_ylabel(
        "Pass rate"
    )

    output_path = Path(
        output_path
    )

    save_figure(
        figure,
        output_path,
    )

    return output_path


def plot_failure_reasons(
    report_data: ReportData,
    output_path: str | Path,
    maximum_reasons: int = 12,
) -> Path:
    failure_summary = (
        build_failure_summary(
            report_data
        )
    )

    aggregated = (
        failure_summary
        .groupby("failure_reason")
        ["candidate_failures"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(maximum_reasons)
        .sort_values(
            ascending=True
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.barh(
        aggregated.index,
        aggregated.values,
    )

    axis.set_title(
        "Most Frequent Candidate Failure Reasons"
    )

    axis.set_xlabel(
        "Number of failures"
    )

    output_path = Path(
        output_path
    )

    save_figure(
        figure,
        output_path,
    )

    return output_path


def plot_full_sample_sharpe_distribution(
    report_data: ReportData,
    output_path: str | Path,
) -> Path:
    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    for family, group in (
        report_data.candidate_metrics
        .groupby("family")
    ):
        axis.hist(
            group["net_sharpe"].dropna(),
            bins=12,
            alpha=0.5,
            label=family,
        )

    axis.axvline(
        0,
        linestyle="--",
    )

    axis.set_title(
        "Full-Sample Net Sharpe Distribution"
    )

    axis.set_xlabel(
        "Net Sharpe"
    )

    axis.set_ylabel(
        "Candidate count"
    )

    axis.legend()

    output_path = Path(
        output_path
    )

    save_figure(
        figure,
        output_path,
    )

    return output_path


def plot_oos_vs_walkforward_sharpe(
    report_data: ReportData,
    output_path: str | Path,
) -> Path:
    comparison = (
        report_data.layer1_results[
            [
                "candidate_id",
                "oos_net_sharpe",
            ]
        ]
        .merge(
            report_data.layer2_results[
                [
                    "candidate_id",
                    "walkforward_net_sharpe",
                ]
            ],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.scatter(
        comparison[
            "oos_net_sharpe"
        ],
        comparison[
            "walkforward_net_sharpe"
        ],
    )

    values = comparison[
        [
            "oos_net_sharpe",
            "walkforward_net_sharpe",
        ]
    ].to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        axis.set_title(
            "OOS versus Walk-Forward Net Sharpe"
        )

        axis.set_xlabel(
            "Layer 1 OOS net Sharpe"
        )

        axis.set_ylabel(
            "Layer 2 walk-forward net Sharpe"
        )

        axis.text(
            0.5,
            0.5,
            "No finite Sharpe pairs available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

        output_path = Path(
            output_path
        )

        save_figure(
            figure,
            output_path,
        )

        return output_path

    minimum_value = np.nanmin(
        values
    )

    maximum_value = np.nanmax(
        values
    )

    axis.plot(
        [
            minimum_value,
            maximum_value,
        ],
        [
            minimum_value,
            maximum_value,
        ],
        linestyle="--",
    )

    axis.set_title(
        "OOS versus Walk-Forward Net Sharpe"
    )

    axis.set_xlabel(
        "Layer 1 OOS net Sharpe"
    )

    axis.set_ylabel(
        "Layer 2 walk-forward net Sharpe"
    )

    output_path = Path(
        output_path
    )

    save_figure(
        figure,
        output_path,
    )

    return output_path


def save_report_figures(
    report_data: ReportData,
    output_directory: str | Path,
) -> dict[str, Path]:
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "validation_funnel": (
            plot_validation_funnel(
                report_data,
                output_directory
                / "validation_funnel.png",
            )
        ),
        "independent_pass_rates": (
            plot_independent_pass_rates(
                report_data,
                output_directory
                / "independent_pass_rates.png",
            )
        ),
        "failure_reasons": (
            plot_failure_reasons(
                report_data,
                output_directory
                / "failure_reasons.png",
            )
        ),
        "sharpe_distribution": (
            plot_full_sample_sharpe_distribution(
                report_data,
                output_directory
                / "full_sample_sharpe_distribution.png",
            )
        ),
        "oos_walkforward_sharpe": (
            plot_oos_vs_walkforward_sharpe(
                report_data,
                output_directory
                / "oos_vs_walkforward_sharpe.png",
            )
        ),
    }

    return paths