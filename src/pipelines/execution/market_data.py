from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"

DEFAULT_LOOKBACK_BARS = 500
DEFAULT_MAX_STALENESS_HOURS = 3.0


def get_latest_bars(
    symbol: str,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    max_staleness_hours: float = DEFAULT_MAX_STALENESS_HOURS,
    processed_data_directory: Path = PROCESSED_DATA_DIRECTORY,
) -> pd.DataFrame:
    """
    Read the most recent lookback_bars H1 bars for one instrument from the
    processed market-data store shared with the research pipeline.

    Resolves the "poll vs. stream" open decision with polling: this reads
    whatever the scheduled src.pipelines.data_ingestion job has already
    written -- it never talks to IBKR itself. lookback_bars defaults to
    500, comfortably above the largest current strategy lookback (trend's
    slow_window maxes at 240) with margin for future larger grids.

    Raises:
        FileNotFoundError: no processed data exists yet for this symbol.
        RuntimeError: the data is empty, or the newest saved bar is older
            than max_staleness_hours -- the scheduled ingestion job likely
            hasn't run recently, and trading on data this stale would be
            worse than refusing to.
    """
    data_path = processed_data_directory / f"{symbol}_H1.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"No processed market data found for {symbol}: {data_path}"
        )

    bars = pd.read_parquet(data_path)

    if bars.empty:
        raise RuntimeError(f"Processed market data for {symbol} is empty.")

    latest_timestamp = pd.Timestamp(bars.index.max())

    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize("UTC")

    staleness_hours = (
        pd.Timestamp.now(tz="UTC") - latest_timestamp
    ).total_seconds() / 3600.0

    if staleness_hours > max_staleness_hours:
        raise RuntimeError(
            f"Latest {symbol} bar is {staleness_hours:.1f} hours old "
            f"(max allowed: {max_staleness_hours}) -- the scheduled "
            "data-ingestion job likely hasn't run recently."
        )

    return bars.tail(lookback_bars)
