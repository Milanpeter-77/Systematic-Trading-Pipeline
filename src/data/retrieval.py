from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.environment import IBKR_DATA_CLIENT_ID, IBKR_HOST, IBKR_PORT

from .ibkr.contracts import build_contract
from .ibkr.historical import HistoricalDataClient, HistoricalRequest


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_IBKR_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "ibkr"
INSTRUMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "instruments.yml"

DEFAULT_HOST = IBKR_HOST
DEFAULT_PORT = IBKR_PORT  # 7497 = paper TWS. 7496 = live TWS, 4001/4002 = IB Gateway.
DEFAULT_CLIENT_ID = IBKR_DATA_CLIENT_ID
DEFAULT_BAR_SIZE = "1 hour"
DEFAULT_DURATION = "1 Y"

# Matches config/validation.yml's sample.in_sample_start -- the earliest
# history the validation gate actually needs. Only used as a floor/fallback
# now: fetch_all_instruments_backfill() auto-detects per instrument whether
# a full backfill from this date or a small incremental top-up is needed,
# so the same call is safe to make on every scheduled run (e.g. hourly).
DEFAULT_BACKFILL_START = "2020-01-01"
DEFAULT_BACKFILL_CHUNK_DURATION = "6 M"

# Safety margin subtracted from an instrument's latest saved bar before an
# incremental fetch, so a missed run (weekend, brief outage) can't silently
# skip a bar. A small chunk_duration is used for incremental fetches (as
# opposed to DEFAULT_BACKFILL_CHUNK_DURATION's 6-month chunks meant for a
# multi-year backfill) since the requested window is normally just this
# margin plus however long since the last run -- comfortably one chunk.
DEFAULT_INCREMENTAL_OVERLAP_HOURS = 72.0
DEFAULT_INCREMENTAL_CHUNK_DURATION = "1 W"

REQUIRED_INSTRUMENT_KEYS = {
    "sec_type",
    "ibkr_symbol",
    "currency",
    "exchange",
    "what_to_show",
    "use_regular_trading_hours",
    "tick_size",
    "asset_class",
    "expected_session",
}


def load_instrument_config(
    config_path: str | Path = INSTRUMENT_CONFIG_PATH,
) -> dict[str, dict[str, Any]]:
    """
    Load and validate the IBKR instrument configuration from YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Instrument configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not config:
        raise ValueError(
            f"Instrument configuration is empty or invalid: {config_path}"
        )

    for symbol, settings in config.items():
        if not isinstance(settings, dict):
            raise ValueError(
                f"Configuration for {symbol} must be a mapping."
            )

        missing_keys = REQUIRED_INSTRUMENT_KEYS.difference(settings)

        if missing_keys:
            raise ValueError(
                f"Configuration for {symbol} is missing: "
                f"{sorted(missing_keys)}"
            )

    return config


def fetch_instrument_history(
    client: HistoricalDataClient,
    symbol: str,
    instrument_config: dict[str, Any],
    duration: str = DEFAULT_DURATION,
    bar_size: str = DEFAULT_BAR_SIZE,
    request_id: int = 1001,
) -> pd.DataFrame:
    """
    Fetch one bounded window of historical bars for a single instrument.
    """
    contract = build_contract(instrument_config)

    request = HistoricalRequest(
        request_id=request_id,
        duration=duration,
        bar_size=bar_size,
        what_to_show=instrument_config["what_to_show"],
        use_regular_trading_hours=bool(
            instrument_config["use_regular_trading_hours"]
        ),
    )

    bars = client.request_historical_bars(
        contract=contract,
        request=request,
    )

    bars = bars.drop(columns=["request_id"])
    bars.insert(1, "symbol", symbol)

    return bars


def fetch_all_instruments(
    client: HistoricalDataClient,
    instrument_config: dict[str, dict[str, Any]],
    duration: str = DEFAULT_DURATION,
    save_output: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical bars for every configured instrument on one connected client.
    """
    results: dict[str, pd.DataFrame] = {}

    if save_output:
        RAW_IBKR_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for index, (symbol, settings) in enumerate(
        instrument_config.items()
    ):
        logger.debug(f"Fetching {symbol} from IBKR...")

        bars = fetch_instrument_history(
            client=client,
            symbol=symbol,
            instrument_config=settings,
            duration=duration,
            request_id=1001 + index,
        )

        results[symbol] = bars

        logger.debug(f"Fetched {symbol}: {len(bars):,} bars")

        if save_output:
            bars.to_parquet(
                RAW_IBKR_DIRECTORY / f"{symbol}_H1.parquet",
                index=False,
            )

    return results


def fetch_instrument_history_backfill(
    client: HistoricalDataClient,
    symbol: str,
    instrument_config: dict[str, Any],
    start: str | pd.Timestamp = DEFAULT_BACKFILL_START,
    bar_size: str = DEFAULT_BAR_SIZE,
    chunk_duration: str = DEFAULT_BACKFILL_CHUNK_DURATION,
    request_id_start: int = 1001,
    request_delay_seconds: float = 2.0,
    max_retries: int = 3,
    what_to_show: str | None = None,
) -> pd.DataFrame:
    """
    Fetch continuous historical bars for one instrument from `start` through
    now, walking backward through time in fixed-size chunks.

    IBKR's hard per-request duration cap is lifted for bar sizes >= 1
    minute, but it still recommends only a few thousand bars per request --
    multi-year H1 history needs chunking regardless of the cap. Walks
    backward from "now", requesting `chunk_duration` of history ending at
    each step, until a chunk's earliest bar reaches `start`, or IBKR
    reports no more history available for this contract (the data
    horizon).

    `what_to_show` overrides instrument_config["what_to_show"] -- used to
    fetch e.g. "BID_ASK" bars for the same contract/window instead of the
    instrument's configured price series.
    """
    contract = build_contract(instrument_config)

    if what_to_show is None:
        what_to_show = instrument_config["what_to_show"]

    start = pd.Timestamp(start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")

    chunks: list[pd.DataFrame] = []
    chunk_end = pd.Timestamp.now(tz="UTC")
    request_id = request_id_start

    while True:
        request = HistoricalRequest(
            request_id=request_id,
            duration=chunk_duration,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_regular_trading_hours=bool(
                instrument_config["use_regular_trading_hours"]
            ),
            end_date_time=chunk_end.strftime("%Y%m%d-%H:%M:%S"),
        )

        bars = None

        for attempt in range(max_retries):
            try:
                bars = client.request_historical_bars(
                    contract=contract,
                    request=request,
                )
                break
            except RuntimeError as error:
                # "returned no historical bars" is our own empty-DataFrame
                # message; "returned no data" is IBKR error code 162 (HMDS
                # has nothing for this window) -- both mean the same thing:
                # this chunk is past the earliest data IBKR has for this
                # contract, not a transient failure worth retrying.
                if (
                    "returned no historical bars" in str(error)
                    or "returned no data" in str(error)
                ):
                    logger.debug(
                        f"{symbol}: no more history before "
                        f"{chunk_end} -- reached the data horizon."
                    )
                    bars = None
                    break

                if attempt == max_retries - 1:
                    raise

                logger.warning(
                    f"{symbol} chunk ending {chunk_end} failed "
                    f"(attempt {attempt + 1}/{max_retries}): {error}. "
                    "Retrying..."
                )
                time.sleep(request_delay_seconds * (attempt + 1))
            except TimeoutError as error:
                if attempt == max_retries - 1:
                    raise

                logger.warning(
                    f"{symbol} chunk ending {chunk_end} timed out "
                    f"(attempt {attempt + 1}/{max_retries}): {error}. "
                    "Retrying..."
                )
                time.sleep(request_delay_seconds * (attempt + 1))

        if bars is None or bars.empty:
            break

        request_id += 1
        chunks.append(bars)

        earliest_in_chunk = bars["timestamp"].min()

        logger.debug(
            f"{symbol}: fetched chunk "
            f"[{earliest_in_chunk}, {chunk_end}], {len(bars):,} bars"
        )

        if earliest_in_chunk <= start:
            break

        chunk_end = earliest_in_chunk - pd.Timedelta(seconds=1)

        time.sleep(request_delay_seconds)

    if not chunks:
        raise RuntimeError(f"No historical bars fetched for {symbol}.")

    combined = (
        pd.concat(chunks, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    combined = combined.loc[
        combined["timestamp"] >= start
    ].reset_index(drop=True)

    combined = combined.drop(columns=["request_id"])
    combined.insert(1, "symbol", symbol)

    return combined


def compute_fill_spread_fraction(
    bid_ask_bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert BID_ASK bars into a per-bar fill_spread_fraction.

    IBKR's BID_ASK historical bars use open=time-average bid, close=
    time-average ask (verified empirically against a live request -- not
    documented precisely enough to trust blindly). A non-positive quoted
    spread is bad data, not a real zero-cost fill (config/validation.yml's
    spread_cleaning.treat_nonpositive_spread_as_missing) -- forward/
    backward-fill it from the nearest valid reading rather than leaking
    NaN into the cost model.
    """
    midpoint = (bid_ask_bars["open"] + bid_ask_bars["close"]) / 2.0
    spread = bid_ask_bars["close"] - bid_ask_bars["open"]

    fill_spread_fraction = (
        (spread / midpoint)
        .where(spread > 0)
        .ffill()
        .bfill()
    )

    return pd.DataFrame(
        {
            "timestamp": bid_ask_bars["timestamp"],
            "fill_spread_fraction": fill_spread_fraction,
        }
    )


def fetch_all_instruments_backfill(
    client: HistoricalDataClient,
    instrument_config: dict[str, dict[str, Any]],
    start: str | pd.Timestamp = DEFAULT_BACKFILL_START,
    save_output: bool = True,
    overlap_hours: float = DEFAULT_INCREMENTAL_OVERLAP_HOURS,
) -> dict[str, pd.DataFrame]:
    """
    Fetch continuous historical bars (price + spread) for every configured
    instrument, auto-detecting per instrument whether a full backfill or an
    incremental top-up is needed.

    If data/raw/ibkr/{symbol}_H1.parquet already exists, only bars from
    `overlap_hours` before its latest saved timestamp onward are fetched
    (using a small chunk_duration, since that window is normally tiny) and
    merged into the existing series. Otherwise the full range from `start`
    is fetched, as a from-scratch backfill would be. This makes it safe to
    call this same function on every scheduled run (e.g. hourly) without
    re-pulling the whole history each time -- a newly added instrument with
    no saved data yet still gets its full history on the next run.
    """
    results: dict[str, pd.DataFrame] = {}

    floor_start = pd.Timestamp(start)
    if floor_start.tzinfo is None:
        floor_start = floor_start.tz_localize("UTC")

    if save_output:
        RAW_IBKR_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for index, (symbol, settings) in enumerate(
        instrument_config.items()
    ):
        raw_path = RAW_IBKR_DIRECTORY / f"{symbol}_H1.parquet"

        existing_bars: pd.DataFrame | None = None
        symbol_start = floor_start
        chunk_duration = DEFAULT_BACKFILL_CHUNK_DURATION

        if raw_path.exists():
            existing_bars = pd.read_parquet(raw_path)

            if not existing_bars.empty:
                last_timestamp = pd.Timestamp(
                    existing_bars["timestamp"].max()
                )

                if last_timestamp.tzinfo is None:
                    last_timestamp = last_timestamp.tz_localize("UTC")

                incremental_start = last_timestamp - pd.Timedelta(
                    hours=overlap_hours
                )

                symbol_start = max(floor_start, incremental_start)
                chunk_duration = DEFAULT_INCREMENTAL_CHUNK_DURATION

        logger.info(f"Fetching {symbol} from IBKR (from {symbol_start})...")

        price_bars = fetch_instrument_history_backfill(
            client=client,
            symbol=symbol,
            instrument_config=settings,
            start=symbol_start,
            chunk_duration=chunk_duration,
            request_id_start=1001 + index * 100,
        )

        logger.info(f"Fetching {symbol} BID_ASK spread from IBKR...")

        bid_ask_bars = fetch_instrument_history_backfill(
            client=client,
            symbol=symbol,
            instrument_config=settings,
            start=symbol_start,
            chunk_duration=chunk_duration,
            request_id_start=5001 + index * 100,
            what_to_show="BID_ASK",
        )

        spread = compute_fill_spread_fraction(bid_ask_bars)

        bars = price_bars.merge(
            spread,
            on="timestamp",
            how="inner",
        )

        if existing_bars is not None and not existing_bars.empty:
            bars = (
                pd.concat([existing_bars, bars], ignore_index=True)
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .reset_index(drop=True)
            )

            # Parquet round-tripping and a fresh IBKR fetch can disagree on
            # datetime64 unit (e.g. microsecond vs. second precision) --
            # concatenating mismatched units silently degrades the column
            # to `object` dtype, which breaks any later `.dt` accessor use
            # (downstream data-quality checks in src.data.validation rely
            # on it). Force a single, consistent dtype after every merge.
            bars["timestamp"] = pd.to_datetime(
                bars["timestamp"], utc=True
            )

        results[symbol] = bars

        logger.info(
            f"{symbol}: {len(bars):,} bars total, "
            f"{bars['timestamp'].min()} to {bars['timestamp'].max()}"
        )

        if save_output:
            bars.to_parquet(
                raw_path,
                index=False,
            )

    return results
