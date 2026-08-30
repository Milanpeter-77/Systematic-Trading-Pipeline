from __future__ import annotations

import json
import logging
import signal
import time
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.engine import run_candidate_backtest
from src.backtest.result import BacktestResult
from src.data.features.statistical import (
    compute_pair_spread,
    screen_cointegrated_pairs,
)
from src.data.retrieval import load_instrument_config
from src.factory import test_history
from src.factory.candidate import candidate_from_record
from src.factory.generator import (
    candidates_to_frame,
    candidates_to_frames_by_family,
    generate_candidates,
)
from src.factory.parallel import create_executor, get_worker_market_data, run_parallel_map
from src.factory.registry import STRATEGY_REGISTRY
from src.logging_config import configure_logging, get_current_run_id, new_run_id
from src.reporting.figures import save_report_figures
from src.reporting.report_data import load_report_data
from src.reporting.summaries import build_report_summary
from src.reporting.tables import save_report_tables
from src.validation.criteria import load_gate_config
from src.validation.layer1_oos import run_layer1_gate
from src.validation.layer2_walkforward import run_layer2_gate
from src.validation.layer3_stress import run_layer3_gate
from src.validation.layer4_statistics import run_layer4_gate


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
CANDIDATE_RESULTS_DIRECTORY = RESULTS_DIRECTORY / "candidates"
BACKTEST_RESULTS_DIRECTORY = RESULTS_DIRECTORY / "backtests"
VALIDATION_RESULTS_DIRECTORY = RESULTS_DIRECTORY / "validation"
REPORT_RESULTS_DIRECTORY = RESULTS_DIRECTORY / "report"
INSTRUMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "instruments.yml"
VALIDATION_CONFIG_PATH = PROJECT_ROOT / "config" / "validation.yml"


def _upsert_by_candidate_id(
    path: Path,
    new_rows: pd.DataFrame,
    key: str = "candidate_id",
) -> pd.DataFrame:
    """
    Merge freshly computed rows into an existing results file, keyed by
    candidate_id: a retested candidate's row is replaced, everything else
    carries forward untouched.

    This is what makes the "current status" tables (candidate_metrics.csv,
    layer{1-4}_results.csv, final_funnel.csv, final_survivors.csv) correct
    once cooldown filtering means a run only tests a subset of the full
    candidate population -- a plain overwrite would silently drop every
    candidate skipped this run. On the very first run (no existing file,
    or an empty new_rows), this degrades to a plain write of new_rows.
    """
    existing = (
        pd.read_csv(path)
        if path.exists()
        else pd.DataFrame(columns=new_rows.columns)
    )

    merged = (
        pd.concat([existing, new_rows], ignore_index=True)
        .drop_duplicates(subset=key, keep="last")
        .sort_values(key)
        .reset_index(drop=True)
    )

    merged.to_csv(path, index=False)

    return merged


def _run_diagnostics_directory() -> Path:
    """
    Directory for this run's per-run-only diagnostic outputs (neighbor
    perturbations, walk-forward windows, stress windows, permutation and
    bootstrap draws): a fresh draw is only meaningful for the candidates
    actually tested this run, so unlike the "current status" tables these
    are never merged with history -- but they must still live under a
    per-run subdirectory (keyed by the same run id used everywhere else,
    see src.logging_config.get_current_run_id) rather than a fixed
    filename, or a run that skips everything due to cooldown would
    silently overwrite the previous run's real diagnostics with an empty
    file.
    """
    directory = (
        VALIDATION_RESULTS_DIRECTORY
        / "diagnostics"
        / get_current_run_id()
    )

    directory.mkdir(parents=True, exist_ok=True)

    return directory


def _split_into_batches(
    candidate_frame: pd.DataFrame,
    batch_size: int,
) -> list[pd.DataFrame]:
    """
    Split candidate_frame into ordered, batch_size-row chunks (the last
    chunk may be smaller).

    batch_size <= 0 or batch_size >= len(candidate_frame) both degrade to
    a single batch containing the whole frame -- this must stay a strict
    generalization of the pre-batching code path, since run_alpha_factory
    relies on "one big batch" reproducing the original unbatched behavior
    exactly when candidate_batching is disabled or batch_size is large.
    """
    if candidate_frame.empty:
        return []

    if batch_size <= 0 or batch_size >= len(candidate_frame):
        return [candidate_frame]

    return [
        candidate_frame.iloc[start : start + batch_size]
        for start in range(0, len(candidate_frame), batch_size)
    ]


def prepare_pair_features(
    save_output: bool = True,
) -> list[str]:
    """
    Screen the active instrument universe for cointegrated pairs and save
    each retained pair's spread as a synthetic pseudo-instrument.

    stat_arb needs a symbol universe of its own (a pair spread, not a
    single tradable instrument) -- this gives it one, computed fresh from
    whatever real instrument data is currently in data/processed/. See
    src.data.features.statistical for the actual screening/spread math;
    this only orchestrates loading input and saving output, matching how
    prepare_all_data()/process_instrument() split those concerns for the
    data-ingestion pipeline.
    """
    instrument_config = load_instrument_config(
        config_path=INSTRUMENT_CONFIG_PATH
    )

    processed_data = {
        symbol: pd.read_parquet(
            PROCESSED_DATA_DIRECTORY / f"{symbol}_H1.parquet"
        )
        for symbol in instrument_config
    }

    pairs = screen_cointegrated_pairs(processed_data)

    pair_symbols: list[str] = []

    if save_output:
        PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for symbol_a, symbol_b in pairs:
        pair_symbol = f"{symbol_a}_{symbol_b}"
        pair_symbols.append(pair_symbol)

        spread, hedge_ratio = compute_pair_spread(
            processed_data[symbol_a]["close"],
            processed_data[symbol_b]["close"],
        )

        # Trading the spread means trading both legs, so both legs' own
        # bid-ask cost are real costs -- combine them, weighting leg B by
        # the hedge ratio (the position size actually taken in it per
        # unit of leg A) rather than treating the pair as spread-free.
        combined_spread_fraction = (
            processed_data[symbol_a]["fill_spread_fraction"]
            + abs(hedge_ratio) * processed_data[symbol_b]["fill_spread_fraction"]
        ).reindex(spread.index)

        # The backtest engine computes returns as next_open/open - 1 (a
        # price-percentage convention) -- that blows up (division by
        # ~zero) whenever a raw spread crosses zero while a position is
        # held, which a mean-reverting spread does by construction. Shift
        # by a constant to keep it safely positive: this doesn't change
        # the strategy's own z-score signal at all (close - rolling_mean
        # is invariant to a constant shift, since both shift together),
        # it only keeps the return engine's price-ratio math well-defined.
        price_like_spread = spread - spread.min() + spread.std()

        pair_frame = pd.DataFrame(
            {
                "open": price_like_spread,
                "high": price_like_spread,
                "low": price_like_spread,
                "close": price_like_spread,
                "fill_spread_fraction": combined_spread_fraction,
            }
        )
        pair_frame.index.name = "timestamp"

        if save_output:
            pair_frame.to_parquet(
                PROCESSED_DATA_DIRECTORY / f"{pair_symbol}_H1.parquet"
            )

    return pair_symbols


def load_all_processed_data(
    pair_symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load every real instrument's processed data plus any pair-spread
    pseudo-instruments (see prepare_pair_features) into one dict, keyed by
    symbol the same way real instruments already are.
    """
    instrument_config = load_instrument_config(
        config_path=INSTRUMENT_CONFIG_PATH
    )

    symbols = list(instrument_config) + list(pair_symbols or [])

    return {
        symbol: pd.read_parquet(
            PROCESSED_DATA_DIRECTORY / f"{symbol}_H1.parquet"
        )
        for symbol in symbols
    }


def build_candidate_population(
    pair_symbols: list[str] | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Generate the complete official strategy-candidate population.
    """
    instrument_config = (
        load_instrument_config(
            config_path=INSTRUMENT_CONFIG_PATH
        )
    )

    # carry (and every carry_* variant) only has an interest_rate_
    # differential column (added during ingestion, see
    # add_interest_rate_differential) for CASH/FX instruments -- every
    # other family, plus stat_arb (and every stat_arb_* variant)'s
    # pair-spread pseudo-instruments, keeps its own separate symbol
    # universe. Routing by name prefix (rather than one literal entry per
    # registered strategy) means new carry_*/stat_arb_* strategies are
    # covered automatically as they're added to STRATEGY_REGISTRY.
    fx_symbols = [
        symbol
        for symbol, settings in instrument_config.items()
        if settings["sec_type"] == "CASH"
    ]

    family_symbol_overrides: dict[str, list[str]] = {}

    if fx_symbols:
        for family in STRATEGY_REGISTRY:
            if family == "carry" or family.startswith("carry_"):
                family_symbol_overrides[family] = fx_symbols

    if pair_symbols:
        for family in STRATEGY_REGISTRY:
            if family == "stat_arb" or family.startswith("stat_arb_"):
                family_symbol_overrides[family] = pair_symbols

    candidates = generate_candidates(
        symbols=list(
            instrument_config
        ),
        family_symbol_overrides=family_symbol_overrides or None,
    )

    candidates_frame = (
        candidates_to_frame(
            candidates
        )
    )

    if candidates_frame.empty:
        raise RuntimeError(
            "Candidate generation returned no candidates."
        )

    if not candidates_frame[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "Candidate IDs are not unique."
        )

    if save_output:
        CANDIDATE_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        for family, family_frame in candidates_to_frames_by_family(
            candidates
        ).items():
            family_frame.to_csv(
                CANDIDATE_RESULTS_DIRECTORY
                / f"{family}.csv",
                index=False,
            )

    return candidates_frame


def _backtest_candidate_record(
    record: dict[str, Any],
    commission_bps_per_side: float,
) -> BacktestResult:
    """
    Worker-process entry point for parallel candidate backtesting: resolves
    this candidate's market data from the pool-initialized worker-local
    store (src.factory.parallel) instead of receiving the full
    processed_data dict as an argument, so it isn't re-pickled per task.
    """
    candidate = candidate_from_record(record)
    market_data = get_worker_market_data()[candidate.symbol]

    return run_candidate_backtest(
        candidate=candidate,
        market_data=market_data,
        commission_bps_per_side=commission_bps_per_side,
    )


def run_all_candidate_backtests(
    processed_data: dict[
        str,
        pd.DataFrame,
    ] | None = None,
    candidate_frame: pd.DataFrame | None = None,
    pair_symbols: list[str] | None = None,
    save_output: bool = True,
    executor: Any = None,
) -> tuple[
    pd.DataFrame,
    dict[str, BacktestResult],
]:
    """
    Backtest the given candidate population -- typically the subset of the
    full official population that is due for testing this run (see
    src.factory.test_history).

    executor is an optional concurrent.futures.ProcessPoolExecutor (see
    src.factory.parallel) whose worker processes were already initialized
    with this run's market data. Pass None (the default) to backtest
    candidates sequentially in-process, exactly as before.

    The returned metrics frame is merged with any prior
    results/backtests/candidate_metrics.csv history when save_output is
    True, so a candidate skipped this run still carries forward its last
    known metrics rather than disappearing from the file.
    """
    if processed_data is None:
        processed_data = load_all_processed_data(pair_symbols)

    if candidate_frame is None:
        candidate_frame = (
            build_candidate_population(
                pair_symbols=pair_symbols,
                save_output=save_output,
            )
        )

    records = candidate_frame.to_dict(orient="records")

    results: dict[
        str,
        BacktestResult,
    ] = {}

    metric_records: list[
        dict[str, Any]
    ] = []

    if executor is None:
        for record in records:
            candidate = (
                candidate_from_record(
                    record
                )
            )

            if candidate.symbol not in (
                processed_data
            ):
                raise KeyError(
                    "No processed market data for "
                    f"{candidate.symbol}."
                )

            result = run_candidate_backtest(
                candidate=candidate,
                market_data=processed_data[
                    candidate.symbol
                ],
                commission_bps_per_side=0.5,
            )

            results[
                candidate.candidate_id
            ] = result

            metric_records.append(
                result.metrics_record()
            )
    else:
        worker = partial(
            _backtest_candidate_record,
            commission_bps_per_side=0.5,
        )

        for result in run_parallel_map(executor, worker, records):
            results[result.candidate.candidate_id] = result
            metric_records.append(result.metrics_record())

    metrics_frame = pd.DataFrame(
        metric_records
    )

    if len(results) != len(
        candidate_frame
    ):
        raise RuntimeError(
            "Backtesting did not produce one "
            "result per candidate."
        )

    if not metrics_frame.empty and not metrics_frame[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "Backtest metrics contain duplicate "
            "candidate IDs."
        )

    if save_output:
        BACKTEST_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_frame = _upsert_by_candidate_id(
            BACKTEST_RESULTS_DIRECTORY
            / "candidate_metrics.csv",
            metrics_frame,
        )

    return metrics_frame, results


def run_layer1_validation(
    backtest_results: dict[
        str,
        BacktestResult,
    ],
    processed_data: dict[
        str,
        pd.DataFrame,
    ],
    save_output: bool = True,
    executor: Any = None,
    batch_label: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run chronological OOS and parameter-sensitivity validation for the
    given (cooldown-filtered) backtest results.

    The returned layer1_results is merged with prior
    results/validation/layer1_results.csv history when save_output is
    True, so a candidate skipped this run still shows its last known
    pass/fail rather than disappearing. neighbor_results (a per-run
    robustness diagnostic) covers only the candidates evaluated this run.

    batch_label, if given, is appended to the diagnostics filename so
    multiple calls within one run (one per candidate batch, see
    run_alpha_factory) don't overwrite each other's diagnostics under the
    same per-run directory.
    """
    gate_config = load_gate_config(
        VALIDATION_CONFIG_PATH
    )

    (
        layer1_results,
        neighbor_results,
    ) = run_layer1_gate(
        backtest_results=backtest_results,
        processed_data=processed_data,
        sample_config=gate_config[
            "sample"
        ],
        layer1_config=gate_config[
            "layer1"
        ],
        commission_bps_per_side=0.5,
        executor=executor,
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer1_results = _upsert_by_candidate_id(
            VALIDATION_RESULTS_DIRECTORY
            / "layer1_results.csv",
            layer1_results,
        )

        neighbor_results_filename = (
            "layer1_neighbor_results.csv"
            if batch_label is None
            else f"layer1_neighbor_results.batch_{batch_label}.csv"
        )

        neighbor_results.to_csv(
            _run_diagnostics_directory()
            / neighbor_results_filename,
            index=False,
        )

        layer1_results.loc[
            layer1_results[
                "layer1_pass"
            ]
        ].to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer1_survivors.csv",
            index=False,
        )

    return (
        layer1_results,
        neighbor_results,
    )


def run_layer2_validation(
    backtest_results: dict[
        str,
        BacktestResult,
    ],
    processed_data: dict[
        str,
        pd.DataFrame,
    ],
    save_output: bool = True,
    executor: Any = None,
    batch_label: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run rolling walk-forward re-optimization validation for the given
    (cooldown-filtered) backtest results.

    The returned layer2_results is merged with prior
    results/validation/layer2_results.csv history when save_output is
    True (see run_layer1_validation for why). window_results is a per-run
    diagnostic covering only the candidates evaluated this run.

    batch_label -- see run_layer1_validation.
    """
    gate_config = load_gate_config(
        VALIDATION_CONFIG_PATH
    )

    (
        layer2_results,
        window_results,
    ) = run_layer2_gate(
        backtest_results=backtest_results,
        processed_data=processed_data,
        layer2_config=gate_config[
            "layer2"
        ],
        commission_bps_per_side=0.5,
        executor=executor,
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer2_results = _upsert_by_candidate_id(
            VALIDATION_RESULTS_DIRECTORY
            / "layer2_results.csv",
            layer2_results,
        )

        window_results_filename = (
            "layer2_window_results.csv"
            if batch_label is None
            else f"layer2_window_results.batch_{batch_label}.csv"
        )

        window_results.to_csv(
            _run_diagnostics_directory()
            / window_results_filename,
            index=False,
        )

        layer2_results.loc[
            layer2_results[
                "layer2_pass"
            ]
        ].to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer2_survivors.csv",
            index=False,
        )

    return (
        layer2_results,
        window_results,
    )


def run_layer3_validation(
    backtest_results: dict[
        str,
        BacktestResult,
    ],
    processed_data: dict[
        str,
        pd.DataFrame,
    ],
    save_output: bool = True,
    executor: Any = None,
    batch_label: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run historical and synthetic stress validation for the given
    (cooldown-filtered) backtest results.

    The returned layer3_results is merged with prior
    results/validation/layer3_results.csv history when save_output is
    True (see run_layer1_validation for why). The historical/synthetic
    stress diagnostics are per-run and cover only the candidates evaluated
    this run.

    batch_label -- see run_layer1_validation.
    """
    gate_config = load_gate_config(
        VALIDATION_CONFIG_PATH
    )

    (
        layer3_results,
        historical_stress_results,
        synthetic_stress_results,
    ) = run_layer3_gate(
        backtest_results=backtest_results,
        processed_data=processed_data,
        layer3_config=gate_config[
            "layer3"
        ],
        executor=executor,
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer3_results = _upsert_by_candidate_id(
            VALIDATION_RESULTS_DIRECTORY
            / "layer3_results.csv",
            layer3_results,
        )

        diagnostics_directory = _run_diagnostics_directory()

        historical_stress_filename = (
            "layer3_historical_stress_results.csv"
            if batch_label is None
            else f"layer3_historical_stress_results.batch_{batch_label}.csv"
        )

        historical_stress_results.to_csv(
            diagnostics_directory
            / historical_stress_filename,
            index=False,
        )

        synthetic_stress_filename = (
            "layer3_synthetic_stress_results.csv"
            if batch_label is None
            else f"layer3_synthetic_stress_results.batch_{batch_label}.csv"
        )

        synthetic_stress_results.to_csv(
            diagnostics_directory
            / synthetic_stress_filename,
            index=False,
        )

        layer3_results.loc[
            layer3_results[
                "layer3_pass"
            ]
        ].to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer3_survivors.csv",
            index=False,
        )

    return (
        layer3_results,
        historical_stress_results,
        synthetic_stress_results,
    )


def run_layer4_validation(
    backtest_results: dict[
        str,
        BacktestResult,
    ],
    official_trial_count: int,
    save_output: bool = True,
    executor: Any = None,
    batch_label: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run permutation, bootstrap, and Deflated Sharpe validation for the
    given (cooldown-filtered) backtest results.

    official_trial_count is the size of the FULL official candidate
    population (before cooldown filtering), used as the Deflated Sharpe
    Ratio's multiple-testing correction denominator -- it must reflect the
    true universe being screened, not just the subset retested this run.
    See build_candidate_population() in run_alpha_factory() for where this
    is computed.

    The returned layer4_results is merged with prior
    results/validation/layer4_results.csv history when save_output is
    True (see run_layer1_validation for why). The permutation/bootstrap
    diagnostics are per-run and cover only the candidates evaluated this
    run -- a fresh permutation draw is only meaningful for a candidate
    actually tested this run, so these are not merged with history.

    batch_label -- see run_layer1_validation.
    """
    gate_config = load_gate_config(
        VALIDATION_CONFIG_PATH
    )

    (
        layer4_results,
        permutation_results,
        bootstrap_results,
    ) = run_layer4_gate(
        backtest_results=backtest_results,
        layer4_config=gate_config[
            "layer4"
        ],
        official_trial_count=official_trial_count,
        executor=executor,
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer4_results = _upsert_by_candidate_id(
            VALIDATION_RESULTS_DIRECTORY
            / "layer4_results.csv",
            layer4_results,
        )

        diagnostics_directory = _run_diagnostics_directory()

        permutation_results_filename = (
            "layer4_permutation_results.parquet"
            if batch_label is None
            else f"layer4_permutation_results.batch_{batch_label}.parquet"
        )

        permutation_results.to_parquet(
            diagnostics_directory
            / permutation_results_filename,
            index=False,
        )

        bootstrap_results_filename = (
            "layer4_bootstrap_results.parquet"
            if batch_label is None
            else f"layer4_bootstrap_results.batch_{batch_label}.parquet"
        )

        bootstrap_results.to_parquet(
            diagnostics_directory
            / bootstrap_results_filename,
            index=False,
        )

        layer4_results.loc[
            layer4_results[
                "layer4_pass"
            ]
        ].to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer4_survivors.csv",
            index=False,
        )

    return (
        layer4_results,
        permutation_results,
        bootstrap_results,
    )


def build_final_funnel(
    layer1_results: pd.DataFrame,
    layer2_results: pd.DataFrame,
    layer3_results: pd.DataFrame,
    layer4_results: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    save_output: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Combine all independent layer decisions and identify final survivors.
    """
    final_funnel = (
        layer1_results[
            [
                "candidate_id",
                "family",
                "symbol",
                "layer1_pass",
            ]
        ]
        .merge(
            layer2_results[
                [
                    "candidate_id",
                    "layer2_pass",
                ]
            ],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            layer3_results[
                [
                    "candidate_id",
                    "layer3_pass",
                ]
            ],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            layer4_results[
                [
                    "candidate_id",
                    "layer4_pass",
                ]
            ],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
    )

    for column in [
        "layer1_pass",
        "layer2_pass",
        "layer3_pass",
        "layer4_pass",
    ]:
        final_funnel[column] = (
            final_funnel[column]
            .fillna(False)
            .astype(bool)
        )

    final_funnel[
        "final_pass"
    ] = (
        final_funnel[
            "layer1_pass"
        ]
        & final_funnel[
            "layer2_pass"
        ]
        & final_funnel[
            "layer3_pass"
        ]
        & final_funnel[
            "layer4_pass"
        ]
    )

    expected_candidate_ids = set(
        candidate_metrics[
            "candidate_id"
        ]
    )

    funnel_candidate_ids = set(
        final_funnel[
            "candidate_id"
        ]
    )

    if (
        funnel_candidate_ids
        != expected_candidate_ids
    ):
        missing = (
            expected_candidate_ids
            - funnel_candidate_ids
        )

        unexpected = (
            funnel_candidate_ids
            - expected_candidate_ids
        )

        raise RuntimeError(
            "Final funnel candidate mismatch. "
            f"Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    if not final_funnel[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "Final funnel contains duplicate "
            "candidate IDs."
        )

    final_survivors = (
        final_funnel.loc[
            final_funnel[
                "final_pass"
            ]
        ]
        .merge(
            candidate_metrics,
            on=[
                "candidate_id",
                "family",
                "symbol",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            layer1_results[
                [
                    "candidate_id",
                    "oos_net_total_return",
                    "oos_net_sharpe",
                    "oos_trade_count",
                ]
            ],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            layer2_results[
                [
                    "candidate_id",
                    "walkforward_net_sharpe",
                    "walkforward_positive_window_fraction",
                ]
            ],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            layer3_results[
                [
                    "candidate_id",
                    "synthetic_net_max_drawdown",
                    "synthetic_terminal_nav",
                ]
            ],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            layer4_results[
                [
                    "candidate_id",
                    "permutation_p_value",
                    "bootstrap_drawdown_p95",
                    "dsr_probability",
                ]
            ],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_funnel.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "final_funnel.csv",
            index=False,
        )

        final_survivors.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "final_survivors.csv",
            index=False,
        )

    return (
        final_funnel,
        final_survivors,
    )


def build_reporting_outputs(
    save_output: bool = True,
) -> dict[str, object]:
    """
    Build final tables, figures, and summary metadata.
    """
    report_data = load_report_data(
        PROJECT_ROOT
    )

    table_directory = (
        REPORT_RESULTS_DIRECTORY
        / "tables"
    )

    figure_directory = (
        REPORT_RESULTS_DIRECTORY
        / "figures"
    )

    if save_output:
        table_paths = save_report_tables(
            report_data=report_data,
            output_directory=table_directory,
        )

        figure_paths = save_report_figures(
            report_data=report_data,
            output_directory=figure_directory,
        )
    else:
        table_paths = {}
        figure_paths = {}

    report_summary = (
        build_report_summary(
            report_data
        )
    )

    if save_output:
        REPORT_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = (
            REPORT_RESULTS_DIRECTORY
            / "report_summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report_summary,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
    else:
        summary_path = None

    return {
        "report_data": report_data,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "report_summary": report_summary,
        "summary_path": summary_path,
    }


def run_alpha_factory() -> None:
    """
    Run the complete Alpha Factory research pipeline (candidate generation
    through validation and reporting).

    Assumes market data has already been prepared by the data-ingestion
    pipeline (see src.pipelines.data_ingestion) and is available under
    data/processed/.

    Candidates already tested within their cooldown window (see
    src.factory.test_history and the `cooldown:` block in
    config/validation.yml) are skipped for backtesting and Layer 1-4
    validation this run; their most recent results simply carry forward in
    the output CSVs. Backtesting and Layer 1-4 validation optionally run
    in parallel across candidates (see src.factory.parallel and the
    `parallelism:` block in config/validation.yml).

    Each stage below is a small, self-contained, banner-delimited block.
    Comment out a whole stage to skip it while testing.
    """
    logger.info("=" * 72)
    logger.info("ALPHA FACTORY PIPELINE")
    logger.info("=" * 72)

    run_id = new_run_id()

    config = load_gate_config(VALIDATION_CONFIG_PATH)
    cooldown_config = config["cooldown"]
    parallelism_config = config["parallelism"]

    # ======================================================================
    # STAGE 1/6 -- FEATURE ENGINEERING (cointegrated-pair screening)
    # ======================================================================
    logger.info("[1/6] Screening for cointegrated pairs...")

    pair_symbols = prepare_pair_features(save_output=True)

    logger.info(
        f"Found {len(pair_symbols)} cointegrated pair(s): {pair_symbols}"
    )

    # ======================================================================
    # STAGE 2/6 -- STRATEGY CANDIDATE GENERATION
    # ======================================================================
    logger.info("[2/6] Generating strategy candidates...")

    candidate_population = (
        build_candidate_population(
            pair_symbols=pair_symbols,
            save_output=True,
        )
    )

    official_trial_count = len(candidate_population)

    logger.info(
        f"Generated "
        f"{official_trial_count} "
        f"official candidates."
    )

    # ----------------------------------------------------------------------
    # COOLDOWN FILTER -- only candidates not tested within the configured
    # cooldown period are due for backtesting/validation this run.
    # ----------------------------------------------------------------------
    conn = None
    due_population = candidate_population
    skipped_count = 0

    if cooldown_config["enabled"]:
        db_path = PROJECT_ROOT / cooldown_config["db_path"]
        conn = test_history.get_connection(db_path)

        due_population, skipped_population = test_history.candidates_due_for_testing(
            conn=conn,
            candidate_frame=candidate_population,
            cooldown_days=float(cooldown_config["period_days"]),
        )

        skipped_count = len(skipped_population)

    logger.info(
        f"{len(due_population)} of {official_trial_count} candidates due "
        f"for testing this run ({skipped_count} still within their "
        f"{cooldown_config['period_days']}-day cooldown)."
    )

    # ======================================================================
    # STAGE 3/6 -- CANDIDATE BACKTESTS
    # Loaded once here (rather than reloaded inside each stage) so the same
    # in-memory market data can also seed the parallel worker pool below.
    # ======================================================================
    processed_data = load_all_processed_data(pair_symbols)

    max_workers = (
        int(parallelism_config["max_workers"])
        if parallelism_config["enabled"]
        else 1
    )

    executor = create_executor(
        max_workers=max_workers,
        market_data_by_symbol=processed_data,
    )

    if executor is not None:
        logger.info(
            f"Parallel execution enabled: {max_workers} worker processes."
        )

    # ------------------------------------------------------------------
    # Candidates are backtested and run through Layer 1-4 in bounded-size
    # batches, not all at once: holding every due candidate's full
    # BacktestResult (a complete multi-year hourly timeseries + trades
    # DataFrame each) in memory simultaneously for the whole run is what
    # OOM-killed this process once the candidate population grew large.
    # Each batch is independent -- no layer filters on an earlier layer's
    # pass/fail for a candidate -- so a batch can go through backtest ->
    # Layer1 -> Layer2 -> Layer3 -> Layer4 and be released before the next
    # batch starts, bounding peak memory regardless of total population
    # size. See config/validation.yml's candidate_batching block.
    # ------------------------------------------------------------------
    batching_config = config.get(
        "candidate_batching",
        {"enabled": False, "batch_size": 0},
    )

    batch_size = (
        int(batching_config.get("batch_size", 0))
        if batching_config.get("enabled", False)
        else 0
    )

    batches = _split_into_batches(due_population, batch_size)
    batch_count = len(batches)

    candidate_metrics = None
    layer1_results = layer2_results = layer3_results = layer4_results = None
    completed_batches = 0

    try:
        for batch_index, candidate_batch in enumerate(batches):
            if _shutdown_requested:
                logger.info(
                    f"Shutdown requested; stopping after "
                    f"{completed_batches}/{batch_count} batch(es) "
                    f"completed this pass."
                )
                break

            batch_label = f"{batch_index + 1:02d}"
            batch_progress = f"batch {batch_index + 1}/{batch_count}"

            logger.info(
                f"[3/6] Running candidate backtests "
                f"({batch_progress}, {len(candidate_batch)} candidates)..."
            )

            (
                candidate_metrics,
                backtest_results,
            ) = run_all_candidate_backtests(
                processed_data=processed_data,
                candidate_frame=candidate_batch,
                pair_symbols=pair_symbols,
                save_output=True,
                executor=executor,
            )

            logger.info(
                f"Completed {len(backtest_results)} candidate backtests "
                f"({batch_progress})."
            )

            # ==============================================================
            # STAGE 4/6 -- LAYER 1-4 VALIDATION (this batch)
            # ==============================================================
            logger.info(
                f"[4/6] Running Layer 1 ({batch_progress}): "
                "OOS and parameter sensitivity..."
            )

            (
                layer1_results,
                layer1_neighbor_results,
            ) = run_layer1_validation(
                backtest_results=backtest_results,
                processed_data=processed_data,
                save_output=True,
                executor=executor,
                batch_label=batch_label,
            )

            logger.info(
                f"Layer 1 passes across full tested history "
                f"({batch_progress}): "
                f"{int(layer1_results['layer1_pass'].sum())}"
            )

            logger.info(
                f"Running Layer 2 ({batch_progress}): "
                "walk-forward validation..."
            )

            (
                layer2_results,
                layer2_window_results,
            ) = run_layer2_validation(
                backtest_results=backtest_results,
                processed_data=processed_data,
                save_output=True,
                executor=executor,
                batch_label=batch_label,
            )

            logger.info(
                f"Layer 2 passes across full tested history "
                f"({batch_progress}): "
                f"{int(layer2_results['layer2_pass'].sum())}"
            )

            logger.info(
                f"Running Layer 3 ({batch_progress}): "
                "historical and synthetic stress..."
            )

            (
                layer3_results,
                layer3_historical_results,
                layer3_synthetic_results,
            ) = run_layer3_validation(
                backtest_results=backtest_results,
                processed_data=processed_data,
                save_output=True,
                executor=executor,
                batch_label=batch_label,
            )

            logger.info(
                f"Layer 3 passes across full tested history "
                f"({batch_progress}): "
                f"{int(layer3_results['layer3_pass'].sum())}"
            )

            logger.info(
                f"Running Layer 4 ({batch_progress}): "
                "statistical validation..."
            )

            (
                layer4_results,
                layer4_permutation_results,
                layer4_bootstrap_results,
            ) = run_layer4_validation(
                backtest_results=backtest_results,
                official_trial_count=official_trial_count,
                save_output=True,
                executor=executor,
                batch_label=batch_label,
            )

            logger.info(
                f"Layer 4 passes across full tested history "
                f"({batch_progress}): "
                f"{int(layer4_results['layer4_pass'].sum())}"
            )

            completed_batches += 1

            # Release this batch's BacktestResult objects (each holds a
            # full multi-year hourly timeseries + trades DataFrame) before
            # the next batch allocates its own. No gc.collect(): nothing
            # else references these objects once `del` runs, so CPython's
            # refcounting frees the underlying pandas/numpy buffers
            # immediately -- a full generational sweep would add overhead
            # roughly batch_count times per run for no benefit, since
            # there's no reference cycle here to break.
            del backtest_results

        if completed_batches == 0:
            logger.info(
                "No batches completed this pass (shutdown requested "
                "before any batch started, or no candidates were due "
                "for testing) -- skipping final funnel and reporting."
            )
        else:
            # ==================================================================
            # STAGE 5/6 -- FINAL FUNNEL
            # By the last completed batch, candidate_metrics/layer{1-4}_results
            # already hold the full upserted history across every batch of
            # this run (plus all prior runs) -- each run_layerN_validation
            # call above returns the merged, on-disk-persisted frame, not
            # just that batch's own rows. So building the funnel once here,
            # after the loop, reproduces exactly the same output as the
            # pre-batching single-pass code.
            # ==================================================================
            logger.info("[5/6] Building final funnel...")

            (
                final_funnel,
                final_survivors,
            ) = build_final_funnel(
                layer1_results=layer1_results,
                layer2_results=layer2_results,
                layer3_results=layer3_results,
                layer4_results=layer4_results,
                candidate_metrics=candidate_metrics,
                save_output=True,
            )

            # ------------------------------------------------------------------
            # Record this run's test events now that Layer 1-4 pass/fail and
            # final_pass are all known together for every candidate tested
            # this run. If anything above raised, execution never reaches
            # this point, so no events get recorded and every due candidate
            # simply stays due and is retried next run -- and if shutdown
            # broke the loop early, only the candidates from batches that
            # did complete are in final_funnel, so only those get recorded;
            # the rest correctly remain due for the next pass.
            # ------------------------------------------------------------------
            if conn is not None:
                tested_at = pd.Timestamp.now(tz="UTC").isoformat()
                due_candidate_ids = set(due_population["candidate_id"])

                events = [
                    test_history.TestEvent(
                        candidate_id=row.candidate_id,
                        run_id=run_id,
                        family=row.family,
                        symbol=row.symbol,
                        tested_at=tested_at,
                        layer1_pass=bool(row.layer1_pass),
                        layer2_pass=bool(row.layer2_pass),
                        layer3_pass=bool(row.layer3_pass),
                        layer4_pass=bool(row.layer4_pass),
                        final_pass=bool(row.final_pass),
                    )
                    for row in final_funnel.loc[
                        final_funnel["candidate_id"].isin(due_candidate_ids)
                    ].itertuples()
                ]

                test_history.record_test_events(conn, events)

                logger.info(
                    f"Recorded {len(events)} test event(s) for run {run_id}."
                )
    finally:
        if executor is not None:
            executor.shutdown()

        if conn is not None:
            conn.close()

    if completed_batches == 0:
        return

    # ======================================================================
    # STAGE 6/6 -- REPORTING OUTPUTS
    # ======================================================================
    logger.info("[6/6] Building reporting outputs...")

    reporting_outputs = (
        build_reporting_outputs(
            save_output=True
        )
    )

    logger.info("=" * 72)
    logger.info("ALPHA FACTORY COMPLETED")
    logger.info("=" * 72)

    logger.info(
        "Official candidates: "
        f"{len(final_funnel)}"
    )

    logger.info(
        "Final survivors: "
        f"{len(final_survivors)}"
    )

    logger.info(
        "Results directory: "
        f"{RESULTS_DIRECTORY}"
    )

    logger.info(
        "Report summary: "
        f"{reporting_outputs['summary_path']}"
    )


_shutdown_requested = False


def _handle_shutdown_signal(signum: int, _frame: Any) -> None:
    global _shutdown_requested

    _shutdown_requested = True

    logger.info(f"Received signal {signum}; will stop after the current pass.")


def _sleep_interruptibly(
    total_seconds: float,
    check_interval_seconds: float = 30.0,
) -> None:
    """
    Sleep in small increments so a shutdown signal is noticed within
    check_interval_seconds instead of only after the full interval elapses.
    """
    remaining = total_seconds

    while remaining > 0 and not _shutdown_requested:
        time.sleep(min(check_interval_seconds, remaining))
        remaining -= check_interval_seconds


def main() -> None:
    """
    Public command-line entry point.

    Runs one pass immediately, then -- if config/validation.yml's `loop:`
    block has enabled=true -- keeps running, sleeping poll_interval_hours
    between passes, until interrupted (Ctrl+C or SIGTERM). Each pass
    re-reads the loop config, so poll_interval_hours can be edited without
    restarting the process. Set loop.enabled=false for a single one-shot
    pass (e.g. manual testing).
    """
    log_file_path = configure_logging(PROJECT_ROOT, pipeline_name="alpha_factory")
    logger.info(f"Logging to {log_file_path}")

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    while True:
        run_alpha_factory()

        if _shutdown_requested:
            logger.info("Shutdown requested, exiting.")
            break

        loop_config = load_gate_config(VALIDATION_CONFIG_PATH)["loop"]

        if not loop_config["enabled"]:
            break

        poll_interval_hours = float(loop_config["poll_interval_hours"])
        logger.info(
            f"Pass complete. Sleeping {poll_interval_hours}h until next check."
        )
        _sleep_interruptibly(poll_interval_hours * 3600)

        if _shutdown_requested:
            logger.info("Shutdown requested, exiting.")
            break


if __name__ == "__main__":
    main()
