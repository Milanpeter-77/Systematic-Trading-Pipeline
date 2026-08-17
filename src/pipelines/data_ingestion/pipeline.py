from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.ibkr.historical import HistoricalDataClient
from src.data.retrieval import (
    DEFAULT_BACKFILL_START,
    DEFAULT_CLIENT_ID,
    DEFAULT_HOST,
    DEFAULT_PORT,
    fetch_all_instruments_backfill,
    load_instrument_config,
)
from src.data.validation import validate_and_clean_h1_data
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
QUALITY_RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "data_quality"
INSTRUMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "instruments.yml"

# config/validation.yml's sample section (in_sample_start=2020-01-01 ..
# out_of_sample_start=2024-01-01, plus Layer 2's 104-week walk-forward
# training windows) needs several YEARS of H1 history to actually pass.
# fetch_all_instruments_backfill() walks backward from now in chunks to
# cover that (see src.data.retrieval), respecting IBKR's per-request
# pacing/bar-count guidance rather than one unbounded request.
MARKET_DATA_BACKFILL_START = DEFAULT_BACKFILL_START
IBKR_HOST = DEFAULT_HOST
IBKR_PORT = DEFAULT_PORT
IBKR_CLIENT_ID = DEFAULT_CLIENT_ID


def process_instrument(
    symbol: str,
    raw_data: pd.DataFrame,
    instrument_config: dict[str, Any],
    save_output: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Validate, clean, and optionally save one already-fetched instrument's H1 bars.
    """
    cleaned_data, validation_report = validate_and_clean_h1_data(
        data=raw_data,
        symbol=symbol,
        expected_session=instrument_config["expected_session"],
    )

    complete_report = {
        "ibkr_symbol": instrument_config["ibkr_symbol"],
        "asset_class": instrument_config["asset_class"],
        "expected_session": instrument_config["expected_session"],
        "tick_size": float(instrument_config["tick_size"]),
        **validation_report,
    }

    if save_output:
        PROCESSED_DATA_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        QUALITY_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        hourly_output_path = (
            PROCESSED_DATA_DIRECTORY
            / f"{symbol}_H1.parquet"
        )

        report_output_path = (
            QUALITY_RESULTS_DIRECTORY
            / f"{symbol}_validation.json"
        )

        cleaned_data.to_parquet(
            hourly_output_path
        )

        with report_output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                complete_report,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    return cleaned_data, complete_report


def prepare_all_data(
    start: str = MARKET_DATA_BACKFILL_START,
    save_output: bool = True,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """
    Connect to IBKR, backfill, validate, clean, and save every configured instrument.

    The IBKR client connection is self-contained here: connect, fetch every
    symbol, disconnect -- nothing downstream needs IBKR awareness.
    """
    instrument_config = load_instrument_config(
        config_path=INSTRUMENT_CONFIG_PATH
    )

    logger.debug(
        f"Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT} "
        f"(client id {IBKR_CLIENT_ID})..."
    )

    client = HistoricalDataClient()

    try:
        client.connect_and_start(
            host=IBKR_HOST,
            port=IBKR_PORT,
            client_id=IBKR_CLIENT_ID,
        )

        raw_data = fetch_all_instruments_backfill(
            client=client,
            instrument_config=instrument_config,
            start=start,
            save_output=save_output,
        )
    finally:
        if client.isConnected():
            client.disconnect()

    logger.debug("Disconnected from IBKR.")

    processed_data: dict[
        str,
        pd.DataFrame,
    ] = {}

    summary_rows: list[
        dict[str, Any]
    ] = []

    for symbol, settings in (
        instrument_config.items()
    ):
        logger.debug(f"Processing {symbol}...")

        hourly_data, report = (
            process_instrument(
                symbol=symbol,
                raw_data=raw_data[symbol],
                instrument_config=settings,
                save_output=save_output,
            )
        )

        processed_data[symbol] = (
            hourly_data
        )

        summary_rows.append(
            {
                "symbol": symbol,
                "ibkr_symbol": report["ibkr_symbol"],
                "asset_class": report["asset_class"],
                "expected_session": report["expected_session"],
                "tick_size": report["tick_size"],
                "row_count": report["row_count"],
                "rows_after_cleaning": report["rows_after_cleaning"],
                "duplicate_timestamp_rows": report["duplicate_timestamp_rows"],
                "invalid_ohlc_count": report["invalid_ohlc_count"],
                "extreme_return_count": report["extreme_return_count"],
                "extreme_range_count": report["extreme_range_count"],
                "unexpected_gap_count": report["unexpected_gap_count"],
                "largest_time_gap_minutes": report["largest_time_gap_minutes"],
                "start_timestamp": report["start_timestamp"],
                "end_timestamp": report["end_timestamp"],
            }
        )

        logger.debug(
            f"Finished {symbol}: "
            f"{len(hourly_data):,} H1 bars"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    if save_output:
        QUALITY_RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary.to_csv(
            QUALITY_RESULTS_DIRECTORY
            / "summary.csv",
            index=False,
        )

    return processed_data, summary


def run_data_ingestion() -> None:
    """
    Run the data-ingestion pipeline: connect to IBKR, fetch, validate, and
    save every configured instrument's H1 bars.
    """
    logger.info("=" * 72)
    logger.info("DATA INGESTION PIPELINE")
    logger.info("=" * 72)

    logger.info("[1/1] Preparing market data...")

    processed_data, data_quality_summary = prepare_all_data(
        save_output=True
    )

    logger.info(
        f"Prepared "
        f"{len(processed_data)} "
        f"instruments."
    )

    logger.info("=" * 72)
    logger.info("DATA INGESTION COMPLETED")
    logger.info("=" * 72)
    logger.info(f"Processed data directory: {PROCESSED_DATA_DIRECTORY}")
    logger.info(f"Data quality directory: {QUALITY_RESULTS_DIRECTORY}")


def main() -> None:
    """
    Public command-line entry point.
    """
    log_file_path = configure_logging(PROJECT_ROOT)
    logger.info(f"Logging to {log_file_path}")

    run_data_ingestion()


if __name__ == "__main__":
    main()
