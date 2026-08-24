from __future__ import annotations

import hashlib
import logging
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from src.backtest.metrics import infer_periods_per_year
from src.backtest.result import BacktestResult
from src.factory.parallel import run_parallel_map


logger = logging.getLogger(__name__)


def calculate_sharpe(
    returns: np.ndarray,
    periods_per_year: float,
) -> float:
    """
    Calculate an annualized Sharpe ratio without a risk-free adjustment.
    """
    returns = np.asarray(
        returns,
        dtype=float,
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if returns.size < 2:
        return np.nan

    standard_deviation = returns.std(
        ddof=1
    )

    if standard_deviation <= 0:
        return np.nan

    return float(
        returns.mean()
        / standard_deviation
        * np.sqrt(periods_per_year)
    )


def calculate_rowwise_sharpe(
    returns: np.ndarray,
    periods_per_year: float,
) -> np.ndarray:
    """
    Calculate annualized Sharpe ratios across rows of a 2D array.
    """
    returns = np.asarray(
        returns,
        dtype=float,
    )

    means = returns.mean(axis=1)

    standard_deviations = returns.std(
        axis=1,
        ddof=1,
    )

    sharpe = np.full(
        returns.shape[0],
        np.nan,
        dtype=float,
    )

    valid = standard_deviations > 0

    sharpe[valid] = (
        means[valid]
        / standard_deviations[valid]
        * np.sqrt(periods_per_year)
    )

    return sharpe

def generate_valid_circular_shifts(
    observation_count: int,
    permutation_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate random non-zero circular-shift offsets.
    """
    if observation_count < 3:
        raise ValueError(
            "At least three observations are required."
        )

    if permutation_count <= 0:
        raise ValueError(
            "permutation_count must be positive."
        )

    return rng.integers(
        low=1,
        high=observation_count,
        size=permutation_count,
    )


def run_circular_shift_permutation_test(
    timeseries: pd.DataFrame,
    permutation_count: int,
    random_seed: int,
    chunk_size: int = 100,
) -> tuple[dict[str, float | int], np.ndarray]:
    """
    Test whether observed Sharpe exceeds Sharpe ratios obtained after
    randomly shifting market returns relative to executed positions.

    Transaction costs remain fixed because the observed trading schedule
    is preserved under the null.
    """
    required_columns = {
        "executed_position",
        "open_to_open_return",
        "transaction_cost",
        "net_return",
    }

    missing = required_columns.difference(
        timeseries.columns
    )

    if missing:
        raise ValueError(
            f"Permutation input is missing columns: "
            f"{sorted(missing)}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    clean_data = timeseries.loc[
        timeseries[
            [
                "executed_position",
                "open_to_open_return",
                "transaction_cost",
                "net_return",
            ]
        ].notna().all(axis=1)
    ].copy()

    if len(clean_data) < 3:
        raise ValueError(
            "Insufficient observations for permutation testing."
        )

    positions = clean_data[
        "executed_position"
    ].to_numpy(dtype=float)

    market_returns = clean_data[
        "open_to_open_return"
    ].to_numpy(dtype=float)

    transaction_costs = clean_data[
        "transaction_cost"
    ].to_numpy(dtype=float)

    observed_returns = clean_data[
        "net_return"
    ].to_numpy(dtype=float)

    periods_per_year = infer_periods_per_year(
        clean_data.index
    )

    observed_sharpe = calculate_sharpe(
        observed_returns,
        periods_per_year,
    )

    rng = np.random.default_rng(
        random_seed
    )

    shifts = generate_valid_circular_shifts(
        observation_count=len(clean_data),
        permutation_count=permutation_count,
        rng=rng,
    )

    permutation_sharpes = np.empty(
        permutation_count,
        dtype=float,
    )

    base_indices = np.arange(
        len(clean_data)
    )

    for start in range(
        0,
        permutation_count,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            permutation_count,
        )

        chunk_shifts = shifts[start:stop]

        shifted_indices = (
            base_indices[None, :]
            - chunk_shifts[:, None]
        ) % len(clean_data)

        shifted_market_returns = (
            market_returns[
                shifted_indices
            ]
        )

        null_gross_returns = (
            positions[None, :]
            * shifted_market_returns
        )

        null_net_returns = (
            null_gross_returns
            - transaction_costs[None, :]
        )

        permutation_sharpes[start:stop] = (
            calculate_rowwise_sharpe(
                returns=null_net_returns,
                periods_per_year=periods_per_year,
            )
        )

    finite_permutations = permutation_sharpes[
        np.isfinite(permutation_sharpes)
    ]

    if finite_permutations.size == 0:
        p_value = np.nan
        permutation_mean_sharpe = np.nan
        permutation_median_sharpe = np.nan
        permutation_sharpe_p95 = np.nan
    else:
        p_value = float(
            (
                1
                + np.sum(
                    finite_permutations
                    >= observed_sharpe
                )
            )
            / (
                finite_permutations.size
                + 1
            )
        )

        permutation_mean_sharpe = float(
            np.mean(finite_permutations)
        )

        permutation_median_sharpe = float(
            np.median(finite_permutations)
        )

        permutation_sharpe_p95 = float(
            np.quantile(
                finite_permutations,
                0.95,
            )
        )

    summary = {
        "observed_net_sharpe": float(
            observed_sharpe
        ),
        "permutation_count": int(
            finite_permutations.size
        ),
        "permutation_mean_sharpe": permutation_mean_sharpe,
        "permutation_median_sharpe": permutation_median_sharpe,
        "permutation_sharpe_p95": permutation_sharpe_p95,
        "permutation_p_value": p_value,
    }

    return summary, permutation_sharpes

def calculate_sequence_max_drawdown(
    trade_returns: np.ndarray,
) -> np.ndarray:
    """
    Calculate maximum drawdown for each row of bootstrapped trade returns.
    """
    if trade_returns.ndim != 2:
        raise ValueError(
            "trade_returns must be a 2D array."
        )

    log_growth = np.log1p(
        trade_returns
    )

    cumulative_log_nav = np.cumsum(
        log_growth,
        axis=1,
    )

    zero_column = np.zeros(
        (
            cumulative_log_nav.shape[0],
            1,
        )
    )

    cumulative_log_nav = np.concatenate(
        [
            zero_column,
            cumulative_log_nav,
        ],
        axis=1,
    )

    running_max = np.maximum.accumulate(
        cumulative_log_nav,
        axis=1,
    )

    drawdown = np.exp(
        cumulative_log_nav
        - running_max
    ) - 1.0

    return drawdown.min(axis=1)


def run_trade_sequence_bootstrap(
    trades: pd.DataFrame,
    bootstrap_count: int,
    random_seed: int,
    chunk_size: int = 500,
) -> tuple[
    dict[str, float | int],
    pd.DataFrame,
]:
    """
    Bootstrap complete net trade returns with replacement.
    """
    if "net_trade_return" not in trades.columns:
        raise ValueError(
            "Trades must contain net_trade_return."
        )

    trade_returns = (
        trades["net_trade_return"]
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(trade_returns) == 0:
        raise ValueError(
            "No valid trade returns available."
        )

    if np.any(trade_returns <= -1):
        raise ValueError(
            "Trade returns at or below -100% "
            "cannot be compounded."
        )

    if bootstrap_count <= 0:
        raise ValueError(
            "bootstrap_count must be positive."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    rng = np.random.default_rng(
        random_seed
    )

    bootstrap_total_returns = np.empty(
        bootstrap_count,
        dtype=float,
    )

    bootstrap_max_drawdowns = np.empty(
        bootstrap_count,
        dtype=float,
    )

    trade_count = len(trade_returns)

    for start in range(
        0,
        bootstrap_count,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            bootstrap_count,
        )

        current_size = stop - start

        sampled_indices = rng.integers(
            low=0,
            high=trade_count,
            size=(
                current_size,
                trade_count,
            ),
        )

        sampled_returns = trade_returns[
            sampled_indices
        ]

        total_returns = np.expm1(
            np.log1p(
                sampled_returns
            ).sum(axis=1)
        )

        maximum_drawdowns = (
            calculate_sequence_max_drawdown(
                sampled_returns
            )
        )

        bootstrap_total_returns[
            start:stop
        ] = total_returns

        bootstrap_max_drawdowns[
            start:stop
        ] = maximum_drawdowns

    probability_negative_return = float(
        np.mean(
            bootstrap_total_returns < 0
        )
    )

    drawdown_p95 = float(
        np.quantile(
            np.abs(
                bootstrap_max_drawdowns
            ),
            0.95,
        )
    )

    summary = {
        "bootstrap_count": int(
            bootstrap_count
        ),
        "bootstrap_trade_count": int(
            trade_count
        ),
        "bootstrap_median_total_return": float(
            np.median(
                bootstrap_total_returns
            )
        ),
        "bootstrap_total_return_p05": float(
            np.quantile(
                bootstrap_total_returns,
                0.05,
            )
        ),
        "bootstrap_total_return_p95": float(
            np.quantile(
                bootstrap_total_returns,
                0.95,
            )
        ),
        "bootstrap_probability_negative_return": (
            probability_negative_return
        ),
        "bootstrap_median_max_drawdown": float(
            np.median(
                np.abs(
                    bootstrap_max_drawdowns
                )
            )
        ),
        "bootstrap_drawdown_p95": drawdown_p95,
    }

    distributions = pd.DataFrame(
        {
            "bootstrap_total_return": (
                bootstrap_total_returns
            ),
            "bootstrap_max_drawdown": (
                bootstrap_max_drawdowns
            ),
        }
    )

    return summary, distributions

def expected_maximum_sharpe(
    trial_count: int,
    sharpe_standard_error: float,
) -> float:
    """
    Approximate the expected maximum Sharpe under multiple trials.

    Uses the Bailey and López de Prado extreme-value approximation.
    """
    if trial_count <= 1:
        return 0.0

    if sharpe_standard_error <= 0:
        return 0.0

    euler_mascheroni = (
        0.5772156649015329
    )

    first_quantile = norm.ppf(
        1.0 - 1.0 / trial_count
    )

    second_quantile = norm.ppf(
        1.0
        - 1.0
        / (
            trial_count
            * np.e
        )
    )

    expected_maximum = (
        (
            1.0 - euler_mascheroni
        )
        * first_quantile
        + euler_mascheroni
        * second_quantile
    )

    return float(
        sharpe_standard_error
        * expected_maximum
    )


def calculate_deflated_sharpe_ratio(
    returns: pd.Series,
    trial_count: int,
) -> dict[str, float | int]:
    """
    Calculate the Deflated Sharpe Ratio probability.

    The calculation uses the non-annualized per-period Sharpe because
    the sampling-distribution formula operates at the observation level.
    """
    clean_returns = (
        returns.dropna()
        .to_numpy(dtype=float)
    )

    observation_count = len(
        clean_returns
    )

    if observation_count < 3:
        return {
            "dsr_observation_count": (
                observation_count
            ),
            "dsr_trial_count": int(
                trial_count
            ),
            "dsr_period_sharpe": np.nan,
            "dsr_benchmark_sharpe": np.nan,
            "dsr_probability": np.nan,
            "dsr_skewness": np.nan,
            "dsr_kurtosis": np.nan,
        }

    standard_deviation = clean_returns.std(
        ddof=1
    )

    if standard_deviation <= 0:
        period_sharpe = np.nan
    else:
        period_sharpe = float(
            clean_returns.mean()
            / standard_deviation
        )

    return_skewness = float(
        skew(
            clean_returns,
            bias=False,
        )
    )

    return_kurtosis = float(
        kurtosis(
            clean_returns,
            fisher=False,
            bias=False,
        )
    )

    sharpe_variance_component = (
        1.0
        - return_skewness
        * period_sharpe
        + (
            (
                return_kurtosis
                - 1.0
            )
            / 4.0
        )
        * period_sharpe**2
    )

    if sharpe_variance_component <= 0:
        sharpe_standard_error = np.nan
        benchmark_sharpe = np.nan
        dsr_probability = np.nan

    else:
        sharpe_standard_error = float(
            np.sqrt(
                sharpe_variance_component
                / (
                    observation_count
                    - 1
                )
            )
        )

        benchmark_sharpe = (
            expected_maximum_sharpe(
                trial_count=trial_count,
                sharpe_standard_error=(
                    sharpe_standard_error
                ),
            )
        )

        test_statistic = (
            (
                period_sharpe
                - benchmark_sharpe
            )
            * np.sqrt(
                observation_count
                - 1
            )
            / np.sqrt(
                sharpe_variance_component
            )
        )

        dsr_probability = float(
            norm.cdf(
                test_statistic
            )
        )

    return {
        "dsr_observation_count": int(
            observation_count
        ),
        "dsr_trial_count": int(
            trial_count
        ),
        "dsr_period_sharpe": float(
            period_sharpe
        ),
        "dsr_sharpe_standard_error": float(
            sharpe_standard_error
        ),
        "dsr_benchmark_sharpe": float(
            benchmark_sharpe
        ),
        "dsr_probability": float(
            dsr_probability
        ),
        "dsr_skewness": (
            return_skewness
        ),
        "dsr_kurtosis": (
            return_kurtosis
        ),
    }

def evaluate_layer4_criteria(
    permutation_summary: dict[
        str,
        float | int,
    ],
    bootstrap_summary: dict[
        str,
        float | int,
    ],
    dsr_summary: dict[
        str,
        float | int,
    ],
    layer4_config: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Apply the pre-specified Layer 4 statistical criteria.
    """
    failures: list[str] = []

    permutation_p_value = float(
        permutation_summary[
            "permutation_p_value"
        ]
    )

    bootstrap_drawdown_p95 = float(
        bootstrap_summary[
            "bootstrap_drawdown_p95"
        ]
    )

    probability_negative_return = float(
        bootstrap_summary[
            "bootstrap_probability_negative_return"
        ]
    )

    bootstrap_trade_count = int(
        bootstrap_summary[
            "bootstrap_trade_count"
        ]
    )

    dsr_probability = float(
        dsr_summary[
            "dsr_probability"
        ]
    )

    if (
        not np.isfinite(
            permutation_p_value
        )
        or permutation_p_value
        >= layer4_config[
            "maximum_permutation_p_value"
        ]
    ):
        failures.append(
            "permutation_p_value"
        )

    if (
        bootstrap_trade_count
        < layer4_config[
            "minimum_trades_for_bootstrap"
        ]
    ):
        failures.append(
            "bootstrap_trade_count"
        )

    if (
        not np.isfinite(
            bootstrap_drawdown_p95
        )
        or bootstrap_drawdown_p95
        > layer4_config[
            "maximum_bootstrap_drawdown_p95"
        ]
    ):
        failures.append(
            "bootstrap_drawdown_p95"
        )

    if (
        not np.isfinite(
            probability_negative_return
        )
        or probability_negative_return
        >= layer4_config[
            "maximum_probability_negative_return"
        ]
    ):
        failures.append(
            "bootstrap_probability_negative_return"
        )

    if (
        not np.isfinite(
            dsr_probability
        )
        or dsr_probability
        < layer4_config[
            "deflated_sharpe_minimum_probability"
        ]
    ):
        failures.append(
            "deflated_sharpe_probability"
        )

    return len(failures) == 0, failures

def stable_candidate_seed(candidate_id: str, base_seed: int) -> int:
    """
    Derive a per-candidate RNG seed from the candidate's own identity.

    Must be independent of any run's batch composition or process, since a
    cooldown-filtered run only tests a varying subset of candidates each
    time, and worker processes in a parallel pool don't share Python's
    per-process string hash salt. Using the built-in `hash()` or a
    position-in-batch index would make permutation/bootstrap draws (and
    therefore layer4 pass/fail) silently non-reproducible across runs.
    """
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()

    return base_seed + (int(digest, 16) % 10_000_000)


def evaluate_layer4_candidate(
    result: BacktestResult,
    layer4_config: dict[str, Any],
    candidate_seed: int,
    official_trial_count: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate one official candidate under Layer 4.
    """
    permutation_summary, permutation_distribution = (
        run_circular_shift_permutation_test(
            timeseries=result.timeseries,
            permutation_count=int(
                layer4_config[
                    "permutation_count"
                ]
            ),
            random_seed=candidate_seed,
            chunk_size=int(
                layer4_config[
                    "permutation_chunk_size"
                ]
            ),
        )
    )

    bootstrap_summary, bootstrap_distribution = (
        run_trade_sequence_bootstrap(
            trades=result.trades,
            bootstrap_count=int(
                layer4_config[
                    "bootstrap_count"
                ]
            ),
            random_seed=(
                candidate_seed + 1
            ),
            chunk_size=int(
                layer4_config[
                    "bootstrap_chunk_size"
                ]
            ),
        )
    )

    dsr_summary = (
        calculate_deflated_sharpe_ratio(
            returns=result.timeseries[
                "net_return"
            ],
            trial_count=official_trial_count,
        )
    )

    layer4_pass, failures = (
        evaluate_layer4_criteria(
            permutation_summary=(
                permutation_summary
            ),
            bootstrap_summary=(
                bootstrap_summary
            ),
            dsr_summary=dsr_summary,
            layer4_config=layer4_config,
        )
    )

    record = {
        **result.candidate.to_dict(),
        **permutation_summary,
        **bootstrap_summary,
        **dsr_summary,
        "layer4_pass": layer4_pass,
        "failure_reason": (
            ";".join(failures)
            if failures
            else ""
        ),
    }

    permutation_frame = pd.DataFrame(
        {
            "candidate_id": (
                result.candidate.candidate_id
            ),
            "permutation_number": np.arange(
                1,
                len(
                    permutation_distribution
                ) + 1,
            ),
            "permutation_net_sharpe": (
                permutation_distribution
            ),
        }
    )

    bootstrap_frame = (
        bootstrap_distribution.copy()
    )

    bootstrap_frame.insert(
        0,
        "candidate_id",
        result.candidate.candidate_id,
    )

    bootstrap_frame.insert(
        1,
        "bootstrap_number",
        np.arange(
            1,
            len(
                bootstrap_frame
            ) + 1,
        ),
    )

    return (
        record,
        permutation_frame,
        bootstrap_frame,
    )

def _evaluate_layer4_candidate_worker(
    result: BacktestResult,
    layer4_config: dict[str, Any],
    official_trial_count: int,
    base_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Worker-process entry point for parallel Layer 4 evaluation. Layer 4
    needs no market data (only each candidate's own backtest timeseries
    and trades), so unlike layers 1-3 this doesn't touch the worker-local
    market-data store.
    """
    candidate_seed = stable_candidate_seed(
        candidate_id=result.candidate.candidate_id,
        base_seed=base_seed,
    )

    return evaluate_layer4_candidate(
        result=result,
        layer4_config=layer4_config,
        candidate_seed=candidate_seed,
        official_trial_count=official_trial_count,
    )


def run_layer4_gate(
    backtest_results: dict[
        str,
        BacktestResult,
    ],
    layer4_config: dict[str, Any],
    official_trial_count: int,
    executor: Any = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate all official candidates under Layer 4.

    executor is an optional concurrent.futures.ProcessPoolExecutor (see
    src.factory.parallel). Pass None (the default) to evaluate candidates
    sequentially in-process, exactly as before.
    """
    candidate_records: list[
        dict[str, Any]
    ] = []

    permutation_frames: list[
        pd.DataFrame
    ] = []

    bootstrap_frames: list[
        pd.DataFrame
    ] = []

    base_seed = int(
        layer4_config[
            "random_seed"
        ]
    )

    sorted_candidate_ids = sorted(
        backtest_results
    )

    if executor is None:
        for candidate_id in sorted_candidate_ids:
            logger.debug(
                f"Layer 4: {candidate_id}"
            )

            result = backtest_results[
                candidate_id
            ]

            candidate_seed = stable_candidate_seed(
                candidate_id=candidate_id,
                base_seed=base_seed,
            )

            (
                record,
                permutation_frame,
                bootstrap_frame,
            ) = evaluate_layer4_candidate(
                result=result,
                layer4_config=layer4_config,
                candidate_seed=candidate_seed,
                official_trial_count=official_trial_count,
            )

            candidate_records.append(
                record
            )

            permutation_frames.append(
                permutation_frame
            )

            bootstrap_frames.append(
                bootstrap_frame
            )
    else:
        worker = partial(
            _evaluate_layer4_candidate_worker,
            layer4_config=layer4_config,
            official_trial_count=official_trial_count,
            base_seed=base_seed,
        )

        ordered_results = [
            backtest_results[candidate_id]
            for candidate_id in sorted_candidate_ids
        ]

        for (
            record,
            permutation_frame,
            bootstrap_frame,
        ) in run_parallel_map(executor, worker, ordered_results):
            candidate_records.append(
                record
            )

            permutation_frames.append(
                permutation_frame
            )

            bootstrap_frames.append(
                bootstrap_frame
            )

    layer4_results = pd.DataFrame(
        candidate_records
    )

    permutation_results = (
        pd.concat(permutation_frames, ignore_index=True)
        if permutation_frames
        else pd.DataFrame()
    )

    bootstrap_results = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame()
    )

    if len(layer4_results) != len(
        backtest_results
    ):
        raise RuntimeError(
            "Layer 4 did not produce one row "
            "per official candidate."
        )

    if not layer4_results.empty and not layer4_results[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "Layer 4 produced duplicate candidate IDs."
        )

    return (
        layer4_results,
        permutation_results,
        bootstrap_results,
    )