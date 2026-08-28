from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlc_data() -> pd.DataFrame:
    """
    A long, continuous synthetic hourly OHLC series for exercising strategy
    indicators. Long enough (3000 bars) to clear the largest grid value used
    by any price-based strategy (480, the trend family's widened slow_window).
    """
    rng = np.random.default_rng(42)
    n = 3000

    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")

    returns = rng.normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(returns))

    open_ = np.roll(close, 1)
    open_[0] = close[0]

    noise = np.abs(rng.normal(0, 0.05, n))
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise

    volume = rng.integers(1, 1000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def synthetic_ohlc_with_rate_differential(
    synthetic_ohlc_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    synthetic_ohlc_data plus a piecewise-constant interest_rate_differential
    column, mimicking FRED's monthly-cadence forward-fill (see
    src.pipelines.data_ingestion.pipeline.add_interest_rate_differential),
    long enough to exercise carry_zscore/carry_rate_momentum's largest
    calendar-day lookback (2160 bars, 90 days).
    """
    rng = np.random.default_rng(7)
    n = len(synthetic_ohlc_data)

    differential = np.zeros(n)
    level = 0.0

    for index in range(n):
        if index % (24 * 25) == 0:
            level = rng.normal(0, 1.0)
        differential[index] = level

    result = synthetic_ohlc_data.copy()
    result["interest_rate_differential"] = differential

    return result


def assert_valid_positions(
    result: pd.DataFrame,
    input_data: pd.DataFrame,
) -> None:
    """
    Shared shape/dtype assertions for any BaseStrategy.generate_positions()
    output, reused across every strategy family's unit tests.
    """
    assert "target_position" in result.columns
    assert result["target_position"].dtype == np.int8
    assert set(result["target_position"].unique()).issubset({-1, 0, 1})
    assert len(result) == len(input_data)
    assert (result.index == input_data.index).all()
