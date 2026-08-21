from __future__ import annotations

import json
import logging
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
from src.factory.candidate import candidate_from_record
from src.factory.generator import (
    candidates_to_frame,
    candidates_to_frames_by_family,
    generate_candidates,
)
from src.logging_config import configure_logging
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

    # carry only has an interest_rate_differential column (added during
    # ingestion, see add_interest_rate_differential) for CASH/FX
    # instruments -- every other family, plus stat_arb's pair-spread
    # pseudo-instruments, keeps its own separate symbol universe.
    fx_symbols = [
        symbol
        for symbol, settings in instrument_config.items()
        if settings["sec_type"] == "CASH"
    ]

    family_symbol_overrides: dict[str, list[str]] = {}

    if fx_symbols:
        family_symbol_overrides["carry"] = fx_symbols

    if pair_symbols:
        family_symbol_overrides["stat_arb"] = pair_symbols

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


def run_all_candidate_backtests(
    processed_data: dict[
        str,
        pd.DataFrame,
    ] | None = None,
    candidate_frame: pd.DataFrame | None = None,
    pair_symbols: list[str] | None = None,
    save_output: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, BacktestResult],
]:
    """
    Backtest the complete official candidate population.
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

    results: dict[
        str,
        BacktestResult,
    ] = {}

    metric_records: list[
        dict[str, Any]
    ] = []

    for record in (
        candidate_frame.to_dict(
            orient="records"
        )
    ):
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

        logger.debug(
            f"Backtesting "
            f"{candidate.candidate_id}"
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

    if not metrics_frame[
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

        metrics_frame.to_csv(
            BACKTEST_RESULTS_DIRECTORY
            / "candidate_metrics.csv",
            index=False,
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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run chronological OOS and parameter-sensitivity validation.
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
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer1_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer1_results.csv",
            index=False,
        )

        neighbor_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer1_neighbor_results.csv",
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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run rolling walk-forward re-optimization validation.
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
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer2_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer2_results.csv",
            index=False,
        )

        window_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer2_window_results.csv",
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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run historical and synthetic stress validation.
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
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer3_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer3_results.csv",
            index=False,
        )

        historical_stress_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer3_historical_stress_results.csv",
            index=False,
        )

        synthetic_stress_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer3_synthetic_stress_results.csv",
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
    save_output: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run permutation, bootstrap, and Deflated Sharpe validation.
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
    )

    if save_output:
        VALIDATION_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        layer4_results.to_csv(
            VALIDATION_RESULTS_DIRECTORY
            / "layer4_results.csv",
            index=False,
        )

        permutation_results.to_parquet(
            VALIDATION_RESULTS_DIRECTORY
            / "layer4_permutation_results.parquet",
            index=False,
        )

        bootstrap_results.to_parquet(
            VALIDATION_RESULTS_DIRECTORY
            / "layer4_bootstrap_results.parquet",
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

    Each stage below is a small, self-contained, banner-delimited block.
    Comment out a whole stage to skip it while testing.
    """
    logger.info("=" * 72)
    logger.info("ALPHA FACTORY PIPELINE")
    logger.info("=" * 72)

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

    logger.info(
        f"Generated "
        f"{len(candidate_population)} "
        f"official candidates."
    )

    # ======================================================================
    # STAGE 3/6 -- CANDIDATE BACKTESTS
    # Reloads processed data from disk (data/processed/*.parquet, produced
    # by the data-ingestion pipeline plus the pair spreads saved above)
    # since it is no longer fetched in-process here.
    # ======================================================================
    logger.info("[3/6] Running candidate backtests...")

    (
        candidate_metrics,
        backtest_results,
    ) = run_all_candidate_backtests(
        processed_data=None,
        candidate_frame=candidate_population,
        pair_symbols=pair_symbols,
        save_output=True,
    )

    logger.info(
        f"Completed "
        f"{len(backtest_results)} "
        f"candidate backtests."
    )

    # ======================================================================
    # STAGE 4/6 -- LAYER 1-4 VALIDATION
    # ======================================================================
    processed_data = load_all_processed_data(pair_symbols)

    logger.info(
        "[4/6] Running Layer 1: "
        "OOS and parameter sensitivity..."
    )

    (
        layer1_results,
        layer1_neighbor_results,
    ) = run_layer1_validation(
        backtest_results=backtest_results,
        processed_data=processed_data,
        save_output=True,
    )

    logger.info(
        "Layer 1 independent passes: "
        f"{int(layer1_results['layer1_pass'].sum())}"
    )

    logger.info(
        "Running Layer 2: "
        "walk-forward validation..."
    )

    (
        layer2_results,
        layer2_window_results,
    ) = run_layer2_validation(
        backtest_results=backtest_results,
        processed_data=processed_data,
        save_output=True,
    )

    logger.info(
        "Layer 2 independent passes: "
        f"{int(layer2_results['layer2_pass'].sum())}"
    )

    logger.info(
        "Running Layer 3: "
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
    )

    logger.info(
        "Layer 3 independent passes: "
        f"{int(layer3_results['layer3_pass'].sum())}"
    )

    logger.info(
        "Running Layer 4: "
        "statistical validation..."
    )

    (
        layer4_results,
        layer4_permutation_results,
        layer4_bootstrap_results,
    ) = run_layer4_validation(
        backtest_results=backtest_results,
        save_output=True,
    )

    logger.info(
        "Layer 4 independent passes: "
        f"{int(layer4_results['layer4_pass'].sum())}"
    )

    # ======================================================================
    # STAGE 5/6 -- FINAL FUNNEL
    # ======================================================================
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


def main() -> None:
    """
    Public command-line entry point.
    """
    log_file_path = configure_logging(PROJECT_ROOT)
    logger.info(f"Logging to {log_file_path}")

    run_alpha_factory()


if __name__ == "__main__":
    main()
