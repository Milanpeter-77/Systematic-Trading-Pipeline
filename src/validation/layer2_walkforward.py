from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import run_candidate_backtest
from src.backtest.metrics import (
    calculate_drawdown,
    calculate_return_metrics,
    infer_periods_per_year,
)
from src.backtest.result import BacktestResult
from src.factory.candidate import CandidateSpec
from src.factory.parallel import get_worker_market_data, run_parallel_map
from src.validation.layer1_oos import calculate_window_metrics, generate_parameter_neighbors


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Layer2OptimizationConfig:
    scope: str
    objective: str
    minimum_training_trades: int
    fallback_policy: str
    tie_break: list[str]
    warmup_weeks: int
    mark_fallback_as_ineligible: bool


@dataclass(frozen=True)
class Layer2CriteriaConfig:
    minimum_oos_sharpe: float
    minimum_profitable_fold_share: float
    minimum_oos_trades: int
    maximum_drawdown: float
    minimum_oos_return: float


@dataclass(frozen=True)
class Layer2Config:
    train_weeks: int
    test_weeks: int
    step_weeks: int
    expanding_window: bool
    minimum_completed_folds: int
    parameter_perturbation: float
    optimization: Layer2OptimizationConfig
    criteria: Layer2CriteriaConfig


@dataclass(frozen=True)
class WalkForwardWindow:
    fold_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def _stable_parameter_json(parameters: dict[str, int | float]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _parameter_order_key(parameters: dict[str, int | float]) -> tuple[Any, ...]:
    return tuple((name, parameters[name]) for name in sorted(parameters))


def _validate_layer2_config(layer2_config: dict[str, Any]) -> Layer2Config:
    train_weeks = int(layer2_config.get("train_weeks", layer2_config.get("training_weeks", 0)))
    test_weeks = int(layer2_config.get("test_weeks", layer2_config.get("testing_weeks", 0)))
    step_weeks = int(layer2_config.get("step_weeks", 0))
    expanding_window = bool(layer2_config.get("expanding_window", False))
    minimum_completed_folds = int(
        layer2_config.get("minimum_completed_folds", layer2_config.get("minimum_test_windows", 1))
    )
    parameter_perturbation = float(layer2_config.get("parameter_perturbation", 0.20))

    optimization_raw = layer2_config.get("optimization", {})
    if not isinstance(optimization_raw, dict):
        raise ValueError("layer2.optimization must be a mapping.")

    optimization = Layer2OptimizationConfig(
        scope=str(optimization_raw.get("scope", "local_neighborhood")),
        objective=str(optimization_raw.get("objective", "net_sharpe")),
        minimum_training_trades=int(optimization_raw.get("minimum_training_trades", 30)),
        fallback_policy=str(optimization_raw.get("fallback_policy", "original_candidate")),
        tie_break=list(
            optimization_raw.get(
                "tie_break",
                ["lower_max_drawdown", "higher_net_return", "fewer_trades", "parameter_order"],
            )
        ),
        warmup_weeks=int(optimization_raw.get("warmup_weeks", 0)),
        mark_fallback_as_ineligible=bool(optimization_raw.get("mark_fallback_as_ineligible", True)),
    )

    criteria_raw = layer2_config.get("criteria", {})
    if not isinstance(criteria_raw, dict):
        raise ValueError("layer2.criteria must be a mapping.")

    criteria = Layer2CriteriaConfig(
        minimum_oos_sharpe=float(
            criteria_raw.get("minimum_oos_sharpe", layer2_config.get("minimum_aggregate_net_sharpe", 0.0))
        ),
        minimum_profitable_fold_share=float(
            criteria_raw.get(
                "minimum_profitable_fold_share",
                layer2_config.get("minimum_positive_window_fraction", 0.0),
            )
        ),
        minimum_oos_trades=int(
            criteria_raw.get("minimum_oos_trades", layer2_config.get("minimum_total_test_trades", 0))
        ),
        maximum_drawdown=float(
            criteria_raw.get("maximum_drawdown", layer2_config.get("maximum_aggregate_net_drawdown", 1.0))
        ),
        minimum_oos_return=float(
            criteria_raw.get("minimum_oos_return", layer2_config.get("minimum_aggregate_net_return", 0.0))
        ),
    )

    config = Layer2Config(
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        step_weeks=step_weeks,
        expanding_window=expanding_window,
        minimum_completed_folds=minimum_completed_folds,
        parameter_perturbation=parameter_perturbation,
        optimization=optimization,
        criteria=criteria,
    )

    if config.train_weeks <= 0:
        raise ValueError("layer2.train_weeks must be positive.")
    if config.test_weeks <= 0:
        raise ValueError("layer2.test_weeks must be positive.")
    if config.step_weeks <= 0:
        raise ValueError("layer2.step_weeks must be positive.")
    if config.minimum_completed_folds <= 0:
        raise ValueError("layer2.minimum_completed_folds must be positive.")
    if not 0 < config.parameter_perturbation < 1:
        raise ValueError("layer2.parameter_perturbation must lie in (0, 1).")

    if config.step_weeks < config.test_weeks:
        raise ValueError(
            "layer2.step_weeks must be >= layer2.test_weeks to keep test windows non-overlapping."
        )

    if config.optimization.scope != "local_neighborhood":
        raise ValueError("Unsupported layer2.optimization.scope.")
    if config.optimization.objective != "net_sharpe":
        raise ValueError("Unsupported layer2.optimization.objective.")
    if config.optimization.fallback_policy != "original_candidate":
        raise ValueError("Unsupported layer2.optimization.fallback_policy.")
    if config.optimization.minimum_training_trades < 0:
        raise ValueError("layer2.optimization.minimum_training_trades must be non-negative.")
    if config.optimization.warmup_weeks < 0:
        raise ValueError("layer2.optimization.warmup_weeks must be non-negative.")

    supported_tie_break = {
        "lower_max_drawdown",
        "higher_net_return",
        "fewer_trades",
        "parameter_order",
    }
    unsupported_tie_break = [item for item in config.optimization.tie_break if item not in supported_tie_break]
    if unsupported_tie_break:
        raise ValueError(
            "Unsupported layer2.optimization.tie_break entries: "
            f"{sorted(unsupported_tie_break)}"
        )

    if config.criteria.minimum_oos_trades < 0:
        raise ValueError("layer2.criteria.minimum_oos_trades must be non-negative.")
    if not 0 <= config.criteria.minimum_profitable_fold_share <= 1:
        raise ValueError("layer2.criteria.minimum_profitable_fold_share must lie in [0, 1].")
    if config.criteria.maximum_drawdown <= 0:
        raise ValueError("layer2.criteria.maximum_drawdown must be positive.")

    return config


def generate_walkforward_windows(
    data_index: pd.DatetimeIndex,
    train_weeks: int,
    test_weeks: int,
    step_weeks: int,
    expanding_window: bool,
) -> list[WalkForwardWindow]:
    if not isinstance(data_index, pd.DatetimeIndex):
        raise TypeError("data_index must be a DatetimeIndex.")
    if len(data_index) < 2:
        raise ValueError("At least two timestamps are required.")
    if not data_index.is_monotonic_increasing:
        raise ValueError("data_index must be chronologically sorted.")
    if data_index.has_duplicates:
        raise ValueError("data_index contains duplicate timestamps.")

    train_delta = pd.Timedelta(weeks=train_weeks)
    test_delta = pd.Timedelta(weeks=test_weeks)
    step_delta = pd.Timedelta(weeks=step_weeks)

    data_start = data_index.min()
    data_end = data_index.max()

    windows: list[WalkForwardWindow] = []
    fold_number = 1
    test_start = data_start + train_delta

    while True:
        train_start = data_start if expanding_window else test_start - train_delta
        train_end = test_start
        test_end = test_start + test_delta

        if test_end > data_end:
            break

        windows.append(
            WalkForwardWindow(
                fold_number=fold_number,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        fold_number += 1
        test_start += step_delta

    return windows


def _slice_backtest_input(market_data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if market_data.empty:
        raise ValueError("market_data cannot be empty.")
    if not isinstance(market_data.index, pd.DatetimeIndex):
        raise TypeError("market_data must use a DatetimeIndex.")
    if not market_data.index.is_monotonic_increasing:
        raise ValueError("market_data must be chronologically sorted.")

    start_position = market_data.index.searchsorted(start, side="left")
    end_position = market_data.index.searchsorted(end, side="left")

    if end_position >= len(market_data):
        end_position = len(market_data) - 1

    sliced = market_data.iloc[start_position : end_position + 1].copy()

    if sliced.empty:
        raise ValueError(f"No observations found in [{start}, {end}).")

    return sliced


def _evaluate_variant_on_window(
    variant: CandidateSpec,
    market_data: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    commission_bps_per_side: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    window_market_data = _slice_backtest_input(market_data=market_data, start=window_start, end=window_end)

    backtest = run_candidate_backtest(
        candidate=variant,
        market_data=window_market_data,
        commission_bps_per_side=commission_bps_per_side,
    )

    window_timeseries = backtest.timeseries.loc[
        (backtest.timeseries.index >= window_start)
        & (backtest.timeseries.index < window_end)
    ].copy()

    if window_timeseries.empty:
        raise ValueError(
            f"No return observations for {variant.candidate_id} in [{window_start}, {window_end})."
        )

    metrics, _ = calculate_window_metrics(window_data=window_timeseries, prefix="window")

    return metrics, window_timeseries


def _local_variants(candidate: CandidateSpec, perturbation: float) -> list[CandidateSpec]:
    neighbors = generate_parameter_neighbors(candidate=candidate, perturbation=perturbation)

    variants = {
        candidate.candidate_id: candidate,
        **{neighbor.candidate_id: neighbor for neighbor in neighbors},
    }

    return [variants[candidate_id] for candidate_id in sorted(variants)]


def _selection_sort_key(record: dict[str, Any], tie_break: list[str]) -> tuple[Any, ...]:
    key_parts: list[Any] = [-float(record["training_net_sharpe"])]

    for rule in tie_break:
        if rule == "lower_max_drawdown":
            key_parts.append(abs(float(record["training_net_max_drawdown"])))
        elif rule == "higher_net_return":
            key_parts.append(-float(record["training_net_total_return"]))
        elif rule == "fewer_trades":
            key_parts.append(int(record["training_trade_count"]))
        elif rule == "parameter_order":
            key_parts.append(_parameter_order_key(record["parameters"]))

    key_parts.append(str(record["variant_candidate_id"]))
    return tuple(key_parts)


def _select_training_variant(
    variant_training_records: list[dict[str, Any]],
    candidate: CandidateSpec,
    optimization: Layer2OptimizationConfig,
) -> tuple[dict[str, Any], bool, bool]:
    eligible = [
        record
        for record in variant_training_records
        if int(record["training_trade_count"]) >= optimization.minimum_training_trades
        and np.isfinite(float(record["training_net_sharpe"]))
    ]

    if eligible:
        selected = sorted(
            eligible,
            key=lambda record: _selection_sort_key(record=record, tie_break=optimization.tie_break),
        )[0]

        return selected, False, False

    for record in variant_training_records:
        if record["variant_candidate_id"] == candidate.candidate_id:
            return record, True, optimization.mark_fallback_as_ineligible

    raise RuntimeError("Original candidate not found in local variant set.")


def _evaluate_fold(
    candidate: CandidateSpec,
    market_data: pd.DataFrame,
    window: WalkForwardWindow,
    config: Layer2Config,
    commission_bps_per_side: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    variants = _local_variants(candidate=candidate, perturbation=config.parameter_perturbation)

    training_records: list[dict[str, Any]] = []

    for variant in variants:
        train_metrics, _ = _evaluate_variant_on_window(
            variant=variant,
            market_data=market_data,
            window_start=window.train_start,
            window_end=window.train_end,
            commission_bps_per_side=commission_bps_per_side,
        )

        training_records.append(
            {
                "variant_candidate_id": variant.candidate_id,
                "parameters": variant.parameters.copy(),
                "training_net_sharpe": float(train_metrics["window_net_sharpe"]),
                "training_net_total_return": float(train_metrics["window_net_total_return"]),
                "training_net_max_drawdown": float(train_metrics["window_net_max_drawdown"]),
                "training_trade_count": int(train_metrics["window_trade_count"]),
            }
        )

    selected_record, fallback_used, training_ineligible = _select_training_variant(
        variant_training_records=training_records,
        candidate=candidate,
        optimization=config.optimization,
    )

    selected_variant = CandidateSpec(
        family=candidate.family,
        symbol=candidate.symbol,
        parameters=selected_record["parameters"],
    )

    warmup_start = window.test_start - pd.Timedelta(weeks=config.optimization.warmup_weeks)
    if warmup_start < window.train_start:
        warmup_start = window.train_start

    _, warmup_test_timeseries = _evaluate_variant_on_window(
        variant=selected_variant,
        market_data=market_data,
        window_start=warmup_start,
        window_end=window.test_end,
        commission_bps_per_side=commission_bps_per_side,
    )

    test_timeseries = warmup_test_timeseries.loc[
        (warmup_test_timeseries.index >= window.test_start)
        & (warmup_test_timeseries.index < window.test_end)
    ].copy()

    if test_timeseries.empty:
        raise ValueError(f"No test observations for {candidate.candidate_id}, fold {window.fold_number}.")

    test_metrics, _ = calculate_window_metrics(window_data=test_timeseries, prefix="test")

    eligible_count = sum(
        1
        for record in training_records
        if int(record["training_trade_count"]) >= config.optimization.minimum_training_trades
        and np.isfinite(float(record["training_net_sharpe"]))
    )

    fold_record: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "symbol": candidate.symbol,
        **window.to_dict(),
        "original_parameters": _stable_parameter_json(candidate.parameters),
        "eligible_parameter_count": int(eligible_count),
        "selected_parameters": _stable_parameter_json(selected_record["parameters"]),
        "selected_variant_candidate_id": selected_variant.candidate_id,
        "training_objective": config.optimization.objective,
        "training_objective_value": float(selected_record["training_net_sharpe"]),
        "training_net_sharpe": float(selected_record["training_net_sharpe"]),
        "training_net_return": float(selected_record["training_net_total_return"]),
        "training_max_drawdown": float(selected_record["training_net_max_drawdown"]),
        "training_trade_count": int(selected_record["training_trade_count"]),
        "fallback_selection_used": bool(fallback_used),
        "training_ineligible": bool(training_ineligible),
        "test_net_return": float(test_metrics["test_net_total_return"]),
        "test_net_sharpe": float(test_metrics["test_net_sharpe"]),
        "test_max_drawdown": float(test_metrics["test_net_max_drawdown"]),
        "test_trade_count": int(test_metrics["test_trade_count"]),
        "positive_test_fold": bool(test_metrics["test_net_total_return"] > 0),
    }

    return fold_record, test_timeseries


def concatenate_walkforward_tests(test_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not test_frames:
        raise ValueError("At least one test frame is required.")

    combined = pd.concat(test_frames, axis=0).sort_index()

    if combined.index.has_duplicates:
        duplicate_count = int(combined.index.duplicated().sum())
        raise ValueError(f"Walk-forward test windows contain duplicate timestamps: {duplicate_count}.")

    return combined


def calculate_walkforward_aggregate_metrics(
    combined_test_data: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> dict[str, float | int]:
    if combined_test_data.empty:
        raise ValueError("combined_test_data cannot be empty.")
    if fold_results.empty:
        raise ValueError("fold_results cannot be empty.")

    periods_per_year = infer_periods_per_year(combined_test_data.index)

    aggregate_metrics: dict[str, float | int] = {}

    aggregate_metrics.update(
        calculate_return_metrics(
            returns=combined_test_data["gross_return"],
            periods_per_year=periods_per_year,
            prefix="walkforward_gross",
        )
    )

    aggregate_metrics.update(
        calculate_return_metrics(
            returns=combined_test_data["net_return"],
            periods_per_year=periods_per_year,
            prefix="walkforward_net",
        )
    )

    net_drawdown = calculate_drawdown(combined_test_data["net_return"])

    completed_folds = int(len(fold_results))
    positive_folds = int(fold_results["positive_test_fold"].sum())

    selected_by_fold = {
        str(row["fold_number"]): row["selected_parameters"]
        for row in fold_results.sort_values("fold_number").to_dict(orient="records")
    }

    parameter_unique_count = int(fold_results["selected_parameters"].nunique())
    mode_share = (
        float(fold_results["selected_parameters"].value_counts().iloc[0] / completed_folds)
        if completed_folds > 0
        else np.nan
    )

    aggregate_metrics.update(
        {
            "walkforward_window_count": completed_folds,
            "walkforward_positive_window_count": positive_folds,
            "walkforward_positive_window_fraction": (
                positive_folds / completed_folds if completed_folds > 0 else np.nan
            ),
            "walkforward_total_test_trades": int(fold_results["test_trade_count"].sum()),
            "walkforward_median_window_sharpe": float(fold_results["test_net_sharpe"].median()),
            "walkforward_worst_window_sharpe": float(fold_results["test_net_sharpe"].min()),
            "walkforward_best_window_sharpe": float(fold_results["test_net_sharpe"].max()),
            "walkforward_median_window_return": float(fold_results["test_net_return"].median()),
            "walkforward_worst_window_return": float(fold_results["test_net_return"].min()),
            "walkforward_total_transaction_cost": float(combined_test_data["transaction_cost"].sum()),
            "walkforward_aggregate_max_drawdown": float(net_drawdown["drawdown"].min()),
            "walkforward_observation_count": int(len(combined_test_data)),
            "layer2_completed_folds": completed_folds,
            "layer2_oos_total_trades": int(fold_results["test_trade_count"].sum()),
            "layer2_profitable_fold_share": (
                positive_folds / completed_folds if completed_folds > 0 else np.nan
            ),
            "layer2_parameter_selection_unique_count": parameter_unique_count,
            "layer2_parameter_selection_mode_share": mode_share,
            "layer2_selected_parameters_by_fold": json.dumps(
                selected_by_fold,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "layer2_fallback_fold_count": int(fold_results["fallback_selection_used"].sum()),
        }
    )

    return aggregate_metrics


def _deterministic_primary_failure_reason(failures: list[str]) -> str:
    return sorted(failures)[0] if failures else ""


def evaluate_layer2_criteria(
    aggregate_metrics: dict[str, float | int],
    config: Layer2Config,
) -> tuple[bool, list[str], str]:
    failures: list[str] = []

    completed_folds = int(aggregate_metrics["layer2_completed_folds"])
    oos_sharpe = float(aggregate_metrics["walkforward_net_sharpe"])
    oos_return = float(aggregate_metrics["walkforward_net_total_return"])
    profitable_fold_share = float(aggregate_metrics["layer2_profitable_fold_share"])
    oos_trades = int(aggregate_metrics["layer2_oos_total_trades"])
    oos_drawdown = float(aggregate_metrics["walkforward_aggregate_max_drawdown"])

    if completed_folds < config.minimum_completed_folds:
        failures.append("minimum_completed_folds")
    if not np.isfinite(oos_sharpe) or oos_sharpe < config.criteria.minimum_oos_sharpe:
        failures.append("minimum_oos_sharpe")
    if not np.isfinite(oos_return) or oos_return <= config.criteria.minimum_oos_return:
        failures.append("minimum_oos_return")
    if (
        not np.isfinite(profitable_fold_share)
        or profitable_fold_share < config.criteria.minimum_profitable_fold_share
    ):
        failures.append("minimum_profitable_fold_share")
    if oos_trades < config.criteria.minimum_oos_trades:
        failures.append("minimum_oos_trades")
    if not np.isfinite(oos_drawdown) or oos_drawdown < -abs(config.criteria.maximum_drawdown):
        failures.append("maximum_drawdown")

    layer2_pass = len(failures) == 0
    primary_failure_reason = _deterministic_primary_failure_reason(failures)

    return layer2_pass, failures, primary_failure_reason


def evaluate_layer2_candidate(
    result: BacktestResult,
    market_data: pd.DataFrame,
    layer2_config: dict[str, Any],
    commission_bps_per_side: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config = _validate_layer2_config(layer2_config)
    candidate = result.candidate

    windows = generate_walkforward_windows(
        data_index=market_data.index,
        train_weeks=config.train_weeks,
        test_weeks=config.test_weeks,
        step_weeks=config.step_weeks,
        expanding_window=config.expanding_window,
    )

    if not windows:
        raise ValueError(f"No valid walk-forward windows for {candidate.candidate_id}.")

    if len(windows) < config.minimum_completed_folds:
        raise ValueError(
            f"Configured minimum_completed_folds={config.minimum_completed_folds} cannot be met; only "
            f"{len(windows)} folds are available for {candidate.candidate_id}."
        )

    fold_records: list[dict[str, Any]] = []
    test_frames: list[pd.DataFrame] = []

    for window in windows:
        fold_record, test_data = _evaluate_fold(
            candidate=candidate,
            market_data=market_data,
            window=window,
            config=config,
            commission_bps_per_side=commission_bps_per_side,
        )

        fold_records.append(fold_record)

        test_data = test_data.copy()
        test_data["fold_number"] = window.fold_number
        test_frames.append(test_data)

    fold_results = pd.DataFrame(fold_records).sort_values("fold_number").reset_index(drop=True)

    combined_test_data = concatenate_walkforward_tests(test_frames=test_frames)

    aggregate_metrics = calculate_walkforward_aggregate_metrics(
        combined_test_data=combined_test_data,
        fold_results=fold_results,
    )

    layer2_pass, failures, primary_failure_reason = evaluate_layer2_criteria(
        aggregate_metrics=aggregate_metrics,
        config=config,
    )

    record: dict[str, Any] = {
        **candidate.to_dict(),
        "walkforward_first_test_start": fold_results["test_start"].min(),
        "walkforward_last_test_end": fold_results["test_end"].max(),
        "layer2_planned_folds": int(len(windows)),
        **aggregate_metrics,
        "layer2_failed_criteria": ";".join(sorted(failures)) if failures else "",
        "layer2_primary_failure_reason": primary_failure_reason,
        "layer2_pass": layer2_pass,
        "failure_reason": ";".join(sorted(failures)) if failures else "",
    }

    return record, fold_results


def _evaluate_layer2_candidate_worker(
    result: BacktestResult,
    layer2_config: dict[str, Any],
    commission_bps_per_side: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Worker-process entry point for parallel Layer 2 evaluation -- see
    _evaluate_layer1_candidate_worker in layer1_oos.py for why market data
    is resolved from the worker-local store instead of an argument.
    """
    market_data = get_worker_market_data()[result.candidate.symbol]

    return evaluate_layer2_candidate(
        result=result,
        market_data=market_data,
        layer2_config=layer2_config,
        commission_bps_per_side=commission_bps_per_side,
    )


def run_layer2_gate(
    backtest_results: dict[str, BacktestResult],
    processed_data: dict[str, pd.DataFrame],
    layer2_config: dict[str, Any],
    commission_bps_per_side: float = 0.5,
    executor: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    executor is an optional concurrent.futures.ProcessPoolExecutor (see
    src.factory.parallel) whose worker processes were already initialized
    with this run's market data. Pass None (the default) to evaluate
    candidates sequentially in-process, exactly as before.
    """
    _validate_layer2_config(layer2_config)

    candidate_records: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []

    if executor is None:
        for candidate_id, result in backtest_results.items():
            logger.debug(f"Layer 2: {candidate_id}")

            symbol = result.candidate.symbol

            if symbol not in processed_data:
                raise KeyError(f"No processed data for symbol {symbol}.")

            record, fold_results = evaluate_layer2_candidate(
                result=result,
                market_data=processed_data[symbol],
                layer2_config=layer2_config,
                commission_bps_per_side=commission_bps_per_side,
            )

            candidate_records.append(record)
            fold_frames.append(fold_results)
    else:
        worker = partial(
            _evaluate_layer2_candidate_worker,
            layer2_config=layer2_config,
            commission_bps_per_side=commission_bps_per_side,
        )

        for record, fold_results in run_parallel_map(
            executor, worker, backtest_results.values()
        ):
            candidate_records.append(record)
            fold_frames.append(fold_results)

    layer2_results = pd.DataFrame(candidate_records)
    all_fold_results = (
        pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    )

    if len(layer2_results) != len(backtest_results):
        raise RuntimeError("Layer 2 did not produce one row per official candidate.")

    if not layer2_results.empty and not layer2_results["candidate_id"].is_unique:
        raise RuntimeError("Layer 2 produced duplicate candidate IDs.")

    return layer2_results, all_fold_results
