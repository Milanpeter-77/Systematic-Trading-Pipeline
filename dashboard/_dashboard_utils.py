from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"


def load_validation_results() -> tuple[dict, pd.DataFrame]:
    with (DATA_DIRECTORY / "validation_results.json").open() as file:
        payload = json.load(file)

    return payload["summary"], pd.DataFrame(payload["candidates"])


def load_strategies() -> pd.DataFrame:
    with (DATA_DIRECTORY / "strategies.json").open() as file:
        return pd.DataFrame(json.load(file))


def load_market_data_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIRECTORY / "market_data_summary.csv")


def load_portfolio_performance() -> pd.DataFrame:
    return pd.read_csv(DATA_DIRECTORY / "portfolio_performance.csv")


def load_positions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIRECTORY / "positions.csv")
