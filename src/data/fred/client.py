from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.environment import FRED_API_KEY


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate --
# the same OECD series family across every currency this pipeline trades,
# for a consistent, directly-comparable policy-rate proxy. Confirmed via
# FRED's own series/search endpoint, not guessed from memory.
POLICY_RATE_SERIES_BY_CURRENCY = {
    "USD": "IRSTCI01USM156N",
    "EUR": "IRSTCI01EZM156N",
    "GBP": "IRSTCI01GBM156N",
    "AUD": "IRSTCI01AUM156N",
    "JPY": "IRSTCI01JPM156N",
}


def load_api_key() -> str:
    """
    Load the FRED API key from the FRED_API_KEY environment variable
    (see src/environment.py, which reads it from the repo-root .env).
    """
    if FRED_API_KEY:
        return FRED_API_KEY.strip()

    raise RuntimeError(
        "No FRED API key found. Set FRED_API_KEY in your .env file "
        "(see .env)."
    )


def fetch_fred_series(
    series_id: str,
    api_key: str | None = None,
    observation_start: str = "2019-01-01",
) -> pd.Series:
    """
    Fetch one FRED series as a float-valued pandas Series indexed by date.

    FRED marks missing observations with "." rather than omitting them --
    those are dropped rather than coerced, since a missing rate should
    carry forward from the last known value (via reindex/ffill wherever
    this is merged onto hourly market data) rather than read as zero.
    """
    if api_key is None:
        api_key = load_api_key()

    params = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
        }
    )

    url = f"{FRED_OBSERVATIONS_URL}?{params}"

    with urllib.request.urlopen(url, timeout=15) as response:
        payload = json.loads(response.read())

    observations = payload.get("observations", [])

    dates = []
    values = []

    for observation in observations:
        if observation["value"] == ".":
            continue

        dates.append(observation["date"])
        values.append(float(observation["value"]))

    series = pd.Series(
        values,
        index=pd.to_datetime(dates, utc=True),
        name=series_id,
    )

    series.index.name = "date"

    return series


def fetch_policy_rate(
    currency: str,
    api_key: str | None = None,
    observation_start: str = "2019-01-01",
) -> pd.Series:
    """
    Fetch a currency's short-term policy-rate proxy by 3-letter code.
    """
    try:
        series_id = POLICY_RATE_SERIES_BY_CURRENCY[currency.upper()]
    except KeyError as error:
        available = sorted(POLICY_RATE_SERIES_BY_CURRENCY)

        raise ValueError(
            f"No FRED policy-rate series configured for currency "
            f"'{currency}'. Available: {available}"
        ) from error

    return fetch_fred_series(
        series_id,
        api_key=api_key,
        observation_start=observation_start,
    )
