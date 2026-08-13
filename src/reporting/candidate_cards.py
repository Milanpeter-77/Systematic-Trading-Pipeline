from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.result import BacktestResult


def create_candidate_card(
    result: BacktestResult,
    gate_row: pd.Series,
    output_path: str | Path,
) -> Path:
    """
    Create a compact candidate diagnostic figure.
    """
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(10, 7)
    )

    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=[
            2,
            1,
            1,
        ],
    )

    nav_axis = figure.add_subplot(
        grid[0]
    )

    drawdown_axis = figure.add_subplot(
        grid[1]
    )

    text_axis = figure.add_subplot(
        grid[2]
    )

    result.timeseries[
        "net_nav"
    ].plot(
        ax=nav_axis
    )

    nav_axis.set_title(
        result.candidate.candidate_id
    )

    nav_axis.set_ylabel(
        "Net NAV"
    )

    result.timeseries[
        "net_drawdown"
    ].plot(
        ax=drawdown_axis
    )

    drawdown_axis.set_ylabel(
        "Drawdown"
    )

    drawdown_axis.set_xlabel(
        "Date"
    )

    text_axis.axis(
        "off"
    )

    text_lines = [
        (
            f"Net Sharpe: "
            f"{result.metrics['net_sharpe']:.2f}"
        ),
        (
            f"Annual return: "
            f"{result.metrics['net_annual_return']:.1%}"
        ),
        (
            f"Maximum drawdown: "
            f"{result.metrics['net_max_drawdown']:.1%}"
        ),
        (
            f"Trades: "
            f"{result.metrics['trade_count']}"
        ),
        (
            f"Layer decisions: "
            f"L1={bool(gate_row['layer1_pass'])}, "
            f"L2={bool(gate_row['layer2_pass'])}, "
            f"L3={bool(gate_row['layer3_pass'])}, "
            f"L4={bool(gate_row['layer4_pass'])}"
        ),
        (
            f"Final pass: "
            f"{bool(gate_row['final_pass'])}"
        ),
    ]

    text_axis.text(
        0.01,
        0.95,
        "\n".join(text_lines),
        va="top",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path