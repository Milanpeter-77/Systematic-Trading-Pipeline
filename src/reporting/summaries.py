from __future__ import annotations

from typing import Any

import pandas as pd

from src.reporting.report_data import ReportData


def build_cumulative_funnel(
    final_funnel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build cumulative survivor counts across all validation layers.
    """
    required_columns = {
        "candidate_id",
        "layer1_pass",
        "layer2_pass",
        "layer3_pass",
        "layer4_pass",
        "final_pass",
    }

    missing = required_columns.difference(
        final_funnel.columns
    )

    if missing:
        raise ValueError(
            f"Final funnel missing columns: "
            f"{sorted(missing)}"
        )

    candidate_count = len(final_funnel)

    cumulative_layer1 = (
        final_funnel["layer1_pass"]
    )

    cumulative_layer2 = (
        cumulative_layer1
        & final_funnel["layer2_pass"]
    )

    cumulative_layer3 = (
        cumulative_layer2
        & final_funnel["layer3_pass"]
    )

    cumulative_layer4 = (
        cumulative_layer3
        & final_funnel["layer4_pass"]
    )

    return pd.DataFrame(
        {
            "stage": [
                "Generated",
                "Layer 1",
                "Layer 2",
                "Layer 3",
                "Layer 4",
            ],
            "survivors": [
                candidate_count,
                int(cumulative_layer1.sum()),
                int(cumulative_layer2.sum()),
                int(cumulative_layer3.sum()),
                int(cumulative_layer4.sum()),
            ],
        }
    )


def build_independent_gate_summary(
    final_funnel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize independent pass counts for each validation layer.
    """
    records = []

    for layer_number in range(1, 5):
        column = f"layer{layer_number}_pass"

        records.append(
            {
                "layer": f"Layer {layer_number}",
                "independent_passes": int(
                    final_funnel[column].sum()
                ),
                "independent_failures": int(
                    (~final_funnel[column]).sum()
                ),
                "independent_pass_rate": float(
                    final_funnel[column].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def explode_failure_reasons(
    results: pd.DataFrame,
    layer_name: str,
) -> pd.DataFrame:
    """
    Convert semicolon-separated failure reasons into one row per reason.
    """
    failed = results.loc[
        ~results[
            f"{layer_name}_pass"
        ],
        [
            "candidate_id",
            "family",
            "symbol",
            "failure_reason",
        ],
    ].copy()

    if failed.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "family",
                "symbol",
                "layer",
                "failure_reason",
            ]
        )

    failed["failure_reason"] = (
        failed["failure_reason"]
        .fillna("")
        .str.split(";")
    )

    failed = failed.explode(
        "failure_reason"
    )

    failed = failed.loc[
        failed["failure_reason"].ne("")
    ].copy()

    failed["layer"] = layer_name

    return failed


def build_failure_summary(
    report_data: ReportData,
) -> pd.DataFrame:
    """
    Combine failure reasons across all four layers.
    """
    frames = [
        explode_failure_reasons(
            report_data.layer1_results,
            "layer1",
        ),
        explode_failure_reasons(
            report_data.layer2_results,
            "layer2",
        ),
        explode_failure_reasons(
            report_data.layer3_results,
            "layer3",
        ),
        explode_failure_reasons(
            report_data.layer4_results,
            "layer4",
        ),
    ]

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "failure_reason",
                "candidate_failures",
            ]
        )

    return (
        combined
        .groupby(
            [
                "layer",
                "failure_reason",
            ]
        )
        .size()
        .rename("candidate_failures")
        .reset_index()
        .sort_values(
            [
                "layer",
                "candidate_failures",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )


def build_family_symbol_summary(
    report_data: ReportData,
) -> pd.DataFrame:
    """
    Summarize candidate counts and pass rates by family and symbol.
    """
    merged = (
        report_data.candidate_metrics
        .merge(
            report_data.final_funnel,
            on=[
                "candidate_id",
                "family",
                "symbol",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    return (
        merged
        .groupby(
            [
                "family",
                "symbol",
            ]
        )
        .agg(
            candidate_count=(
                "candidate_id",
                "count",
            ),
            median_full_sample_sharpe=(
                "net_sharpe",
                "median",
            ),
            layer1_passes=(
                "layer1_pass",
                "sum",
            ),
            layer2_passes=(
                "layer2_pass",
                "sum",
            ),
            layer3_passes=(
                "layer3_pass",
                "sum",
            ),
            layer4_passes=(
                "layer4_pass",
                "sum",
            ),
            final_survivors=(
                "final_pass",
                "sum",
            ),
        )
        .reset_index()
    )


def build_report_summary(
    report_data: ReportData,
) -> dict[str, Any]:
    """
    Build a compact machine-readable report summary.
    """
    funnel = build_cumulative_funnel(
        report_data.final_funnel
    )

    best_full_sample = (
        report_data.candidate_metrics
        .sort_values(
            "net_sharpe",
            ascending=False,
        )
        .iloc[0]
    )

    summary: dict[str, Any] = {
        "official_candidate_count": int(
            len(
                report_data.final_funnel
            )
        ),
        "final_survivor_count": int(
            report_data.final_funnel[
                "final_pass"
            ].sum()
        ),
        "final_survival_rate": float(
            report_data.final_funnel[
                "final_pass"
            ].mean()
        ),
        "best_full_sample_candidate": str(
            best_full_sample[
                "candidate_id"
            ]
        ),
        "best_full_sample_net_sharpe": float(
            best_full_sample[
                "net_sharpe"
            ]
        ),
        "funnel": funnel.to_dict(
            orient="records"
        ),
    }

    return summary