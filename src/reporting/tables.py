from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.report_data import ReportData
from src.reporting.summaries import (
    build_cumulative_funnel,
    build_failure_summary,
    build_family_symbol_summary,
    build_independent_gate_summary,
)


def format_percentage(
    series: pd.Series,
    decimals: int = 1,
) -> pd.Series:
    return series.map(
        lambda value: (
            ""
            if pd.isna(value)
            else f"{value:.{decimals}%}"
        )
    )


def format_decimal(
    series: pd.Series,
    decimals: int = 2,
) -> pd.Series:
    return series.map(
        lambda value: (
            ""
            if pd.isna(value)
            else f"{value:.{decimals}f}"
        )
    )


def build_candidate_performance_table(
    report_data: ReportData,
    maximum_rows: int = 10,
) -> pd.DataFrame:
    """
    Build a compact table of the strongest candidates by full-sample Sharpe.
    """
    table = (
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
        .sort_values(
            "net_sharpe",
            ascending=False,
        )
        .head(maximum_rows)
        [
            [
                "candidate_id",
                "net_annual_return",
                "net_sharpe",
                "net_max_drawdown",
                "trade_count",
                "layer1_pass",
                "layer2_pass",
                "layer3_pass",
                "layer4_pass",
                "final_pass",
            ]
        ]
        .copy()
    )

    table["net_annual_return"] = (
        format_percentage(
            table["net_annual_return"]
        )
    )

    table["net_sharpe"] = (
        format_decimal(
            table["net_sharpe"]
        )
    )

    table["net_max_drawdown"] = (
        format_percentage(
            table["net_max_drawdown"]
        )
    )

    return table


def build_final_survivor_table(
    report_data: ReportData,
) -> pd.DataFrame:
    """
    Build a concise final-survivor table.
    """
    if report_data.final_survivors.empty:
        return pd.DataFrame(
            {
                "result": [
                    (
                        "No candidate passed all four "
                        "validation layers."
                    )
                ]
            }
        )

    desired_columns = [
        "candidate_id",
        "net_annual_return",
        "net_sharpe",
        "net_max_drawdown",
        "oos_net_sharpe",
        "walkforward_net_sharpe",
        "synthetic_terminal_nav",
        "permutation_p_value",
        "dsr_probability",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column
        in report_data.final_survivors.columns
    ]

    table = (
        report_data.final_survivors[
            available_columns
        ]
        .copy()
    )

    percentage_columns = [
        "net_annual_return",
        "net_max_drawdown",
    ]

    for column in percentage_columns:
        if column in table.columns:
            table[column] = format_percentage(
                table[column]
            )

    decimal_columns = [
        "net_sharpe",
        "oos_net_sharpe",
        "walkforward_net_sharpe",
        "synthetic_terminal_nav",
        "permutation_p_value",
        "dsr_probability",
    ]

    for column in decimal_columns:
        if column in table.columns:
            table[column] = format_decimal(
                table[column],
                decimals=3,
            )

    return table


def save_report_tables(
    report_data: ReportData,
    output_directory: str | Path,
) -> dict[str, Path]:
    """
    Build and save all main reporting tables.
    """
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables = {
        "cumulative_funnel": (
            build_cumulative_funnel(
                report_data.final_funnel
            )
        ),
        "independent_gate_summary": (
            build_independent_gate_summary(
                report_data.final_funnel
            )
        ),
        "failure_summary": (
            build_failure_summary(
                report_data
            )
        ),
        "family_symbol_summary": (
            build_family_symbol_summary(
                report_data
            )
        ),
        "top_candidate_performance": (
            build_candidate_performance_table(
                report_data
            )
        ),
        "final_survivors": (
            build_final_survivor_table(
                report_data
            )
        ),
    }

    output_paths: dict[
        str,
        Path,
    ] = {}

    for table_name, table in tables.items():
        path = (
            output_directory
            / f"{table_name}.csv"
        )

        table.to_csv(
            path,
            index=False,
        )

        output_paths[
            table_name
        ] = path

    return output_paths