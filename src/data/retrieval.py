from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .ibkr.contracts import build_contract
from .ibkr.historical import HistoricalDataClient, HistoricalRequest


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_IBKR_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "ibkr"
INSTRUMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "instruments.yml"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497  # paper TWS. 7496 = live TWS, 4001/4002 = IB Gateway.
DEFAULT_CLIENT_ID = 2
DEFAULT_BAR_SIZE = "1 hour"
DEFAULT_DURATION = "1 Y"

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
