from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factory.registry import STRATEGY_REGISTRY

RESULTS_DIRECTORY = PROJECT_ROOT / "results"
CONFIG_DIRECTORY = PROJECT_ROOT / "config"
DASHBOARD_DATA_DIRECTORY = PROJECT_ROOT / "dashboard" / "data"

VALIDATION_LAYERS = (1, 2, 3, 4)

MARKET_DATA_SUMMARY_COLUMNS = [
    "symbol",
    "asset_class",
    "currency",
    "exchange",
    "tick_size",
    "expected_session",
    "row_count",
    "rows_after_cleaning",
    "start_timestamp",
    "end_timestamp",
    "duplicate_timestamp_rows",
    "invalid_ohlc_count",
    "extreme_return_count",
    "extreme_range_count",
    "unexpected_gap_count",
    "largest_time_gap_minutes",
]


def export_market_data_summary() -> None:
    """
    Extend results/data_quality/summary.csv with per-symbol currency/exchange
    from config/instruments.yml.
    """
    summary = pd.read_csv(RESULTS_DIRECTORY / "data_quality" / "summary.csv")

    with (CONFIG_DIRECTORY / "instruments.yml").open() as file:
        instruments = yaml.safe_load(file)

    summary["currency"] = summary["symbol"].map(
        lambda symbol: instruments.get(symbol, {}).get("currency")
    )
    summary["exchange"] = summary["symbol"].map(
        lambda symbol: instruments.get(symbol, {}).get("exchange")
    )

    summary[MARKET_DATA_SUMMARY_COLUMNS].to_csv(
        DASHBOARD_DATA_DIRECTORY / "market_data_summary.csv", index=False
    )


def _strategy_description(strategy_class: type) -> str | None:
    """
    First paragraph of the strategy class docstring, collapsed to one line.
    """
    docstring = inspect.getdoc(strategy_class)

    if not docstring:
        return None

    first_paragraph = docstring.split("\n\n", 1)[0]

    return " ".join(first_paragraph.split())


def export_strategies() -> None:
    """
    One entry per strategy family, combining the class registry (name,
    description, parameters, enabled) with candidate/validation/deployment
    counts read from results/ and config/.
    """
    final_survivors = pd.read_csv(RESULTS_DIRECTORY / "validation" / "final_survivors.csv")

    with (CONFIG_DIRECTORY / "strategies.yml").open() as file:
        deployed_candidate_ids = yaml.safe_load(file).get("deployed_strategies") or []

    strategies = []

    for family_name, strategy_class in STRATEGY_REGISTRY.items():
        candidates_path = RESULTS_DIRECTORY / "candidates" / f"{family_name}.csv"
        candidate_count = (
            len(pd.read_csv(candidates_path)) if candidates_path.exists() else 0
        )

        validated_count = int((final_survivors["family"] == family_name).sum())

        family_deployed_ids = [
            candidate_id
            for candidate_id in deployed_candidate_ids
            if candidate_id.startswith(f"{family_name}__")
        ]

        if not strategy_class.enabled:
            status = "disabled"
        elif family_deployed_ids:
            status = "deployed"
        elif validated_count > 0:
            status = "validated"
        else:
            status = "candidate"

        strategies.append(
            {
                "name": family_name,
                "description": _strategy_description(strategy_class),
                "enabled": strategy_class.enabled,
                "parameter_grid": strategy_class.parameter_grid,
                "candidate_count": candidate_count,
                "validated_count": validated_count,
                "deployed_candidate_ids": family_deployed_ids,
                "status": status,
            }
        )

    with (DASHBOARD_DATA_DIRECTORY / "strategies.json").open("w", encoding="utf-8") as file:
        json.dump(strategies, file, indent=2)


def export_validation_results() -> None:
    """
    Pipeline-wide summary (verbatim copy of results/report/report_summary.json)
    plus a per-candidate table joined from final_funnel.csv, candidate_metrics.csv,
    and each layer's own failure_reason column.
    """
    with (RESULTS_DIRECTORY / "report" / "report_summary.json").open() as file:
        summary = json.load(file)

    merged = pd.read_csv(RESULTS_DIRECTORY / "validation" / "final_funnel.csv")

    metrics = pd.read_csv(RESULTS_DIRECTORY / "backtests" / "candidate_metrics.csv")[
        ["candidate_id", "net_sharpe", "net_max_drawdown", "net_annual_return"]
    ]
    merged = merged.merge(metrics, on="candidate_id", how="left")

    failure_reason_columns = []

    for layer in VALIDATION_LAYERS:
        layer_column = f"layer{layer}_failure_reason"
        layer_frame = pd.read_csv(
            RESULTS_DIRECTORY / "validation" / f"layer{layer}_results.csv"
        )[["candidate_id", "failure_reason"]].rename(
            columns={"failure_reason": layer_column}
        )
        merged = merged.merge(layer_frame, on="candidate_id", how="left")
        failure_reason_columns.append(layer_column)

    # First non-null failure_reason across layers 1-4, in order.
    merged["failure_reason"] = merged[failure_reason_columns].bfill(axis=1).iloc[:, 0]
    merged = merged.drop(columns=failure_reason_columns)

    candidates = json.loads(merged.to_json(orient="records"))

    with (DASHBOARD_DATA_DIRECTORY / "validation_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({"summary": summary, "candidates": candidates}, file, indent=2)


def export_portfolio_performance_placeholder() -> None:
    """
    Header-only placeholder: Portfolio Construction / Risk Management / Paper
    Trading Execution (src/portfolio, src/risk, src/pipelines/execution) are
    not yet implemented, so no equity curve exists on disk to read.
    """
    pd.DataFrame(
        columns=["date", "equity", "daily_return", "cumulative_return", "drawdown"]
    ).to_csv(DASHBOARD_DATA_DIRECTORY / "portfolio_performance.csv", index=False)


def export_positions_placeholder() -> None:
    """
    Header-only placeholder shaped after OrderState
    (src/pipelines/execution/orders.py), the only forward-looking sketch of
    per-candidate position tracking in the codebase today.
    """
    pd.DataFrame(
        columns=[
            "candidate_id",
            "symbol",
            "family",
            "target_position",
            "current_position",
            "status",
            "created_at",
            "updated_at",
        ]
    ).to_csv(DASHBOARD_DATA_DIRECTORY / "positions.csv", index=False)


def main() -> None:
    DASHBOARD_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    export_market_data_summary()
    export_strategies()
    export_validation_results()
    export_portfolio_performance_placeholder()
    export_positions_placeholder()

    print(f"Dashboard data exported to {DASHBOARD_DATA_DIRECTORY}")


if __name__ == "__main__":
    main()
