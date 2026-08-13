from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close"]


def _expected_gap_ceiling_minutes(
    timestamps: pd.Series,
    expected_session: str,
) -> pd.Series:
    """
    Compute the maximum tolerated gap, in minutes, before each bar.

    24/7 markets expect a roughly constant ~1-hour cadence. 24/5 and
    broker-defined sessions additionally tolerate the weekly Friday-close
    to Sunday/Monday-open gap without flagging it as anomalous.
    """
    if expected_session == "24/7":
        return pd.Series(90.0, index=timestamps.index)

    previous_day = timestamps.shift(1).dt.dayofweek

    return pd.Series(
        np.where(previous_day == 4, 52 * 60.0, 90.0),
        index=timestamps.index,
    )


def add_validation_flags(
    data: pd.DataFrame,
    expected_session: str,
    extreme_return_threshold: float = 0.02,
    extreme_range_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Add row-level data-quality flags without silently deleting observations.

    The extreme thresholds are diagnostic flags, not automatic deletion rules.
    """
    output = data.copy()

    output["invalid_timestamp"] = output["timestamp"].isna()

    output["duplicate_timestamp"] = output["timestamp"].duplicated(
        keep=False
    )

    output["missing_required_value"] = output[
        ["timestamp", *PRICE_COLUMNS]
    ].isna().any(axis=1)

    output["nonpositive_price"] = (
        output[PRICE_COLUMNS].le(0).any(axis=1)
    )

    output["high_below_low"] = output["high"] < output["low"]

    output["open_outside_range"] = (
        (output["open"] < output["low"])
        | (output["open"] > output["high"])
    )

    output["close_outside_range"] = (
        (output["close"] < output["low"])
        | (output["close"] > output["high"])
    )

    output["time_gap_minutes"] = (
        output["timestamp"].diff().dt.total_seconds() / 60
    )

    gap_ceiling = _expected_gap_ceiling_minutes(
        timestamps=output["timestamp"],
        expected_session=expected_session,
    )

    output["unexpected_gap"] = (
        output["time_gap_minutes"] > gap_ceiling
    )

    output["close_return"] = output["close"].pct_change(
        fill_method=None
    ).where(~output["unexpected_gap"])

    output["bar_range_fraction"] = (
        (output["high"] - output["low"])
        / output["close"].replace(0, np.nan)
    )

    output["extreme_return"] = (
        output["close_return"].abs() > extreme_return_threshold
    )

    output["extreme_range"] = (
        output["bar_range_fraction"] > extreme_range_threshold
    )

    return output


def remove_mechanically_invalid_rows(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Remove only observations that are mechanically unusable.

    Statistically unusual but logically valid observations are retained.
    Duplicate timestamps are resolved by keeping the final source row.
    """
    output = data.copy()

    invalid_mask = (
        output["invalid_timestamp"]
        | output["missing_required_value"]
        | output["nonpositive_price"]
        | output["high_below_low"]
        | output["open_outside_range"]
        | output["close_outside_range"]
    )

    removed_invalid_count = int(invalid_mask.sum())

    output = output.loc[~invalid_mask].copy()

    duplicate_count_before = int(
        output["timestamp"].duplicated(keep=False).sum()
    )

    output = output.drop_duplicates(
        subset="timestamp",
        keep="last",
    )

    output = output.sort_values("timestamp").reset_index(drop=True)

    cleaning_summary = {
        "removed_invalid_rows": removed_invalid_count,
        "duplicate_rows_identified": duplicate_count_before,
        "rows_after_cleaning": int(len(output)),
    }

    return output, cleaning_summary


def create_validation_report(
    data: pd.DataFrame,
    symbol: str,
) -> dict[str, Any]:
    """
    Create a serializable data-quality summary for one instrument.
    """
    valid_timestamps = data["timestamp"].dropna()
    time_gaps = data["time_gap_minutes"].dropna()

    report: dict[str, Any] = {
        "symbol": symbol,
        "row_count": int(len(data)),
        "start_timestamp": (
            valid_timestamps.min().isoformat()
            if not valid_timestamps.empty
            else None
        ),
        "end_timestamp": (
            valid_timestamps.max().isoformat()
            if not valid_timestamps.empty
            else None
        ),
        "invalid_timestamp_count": int(
            data["invalid_timestamp"].sum()
        ),
        "duplicate_timestamp_rows": int(
            data["duplicate_timestamp"].sum()
        ),
        "missing_required_value_count": int(
            data["missing_required_value"].sum()
        ),
        "nonpositive_price_count": int(
            data["nonpositive_price"].sum()
        ),
        "invalid_ohlc_count": int(
            (
                data["high_below_low"]
                | data["open_outside_range"]
                | data["close_outside_range"]
            ).sum()
        ),
        "extreme_return_count": int(
            data["extreme_return"].sum()
        ),
        "extreme_range_count": int(
            data["extreme_range"].sum()
        ),
        "unexpected_gap_count": int(
            data["unexpected_gap"].sum()
        ),
        "largest_time_gap_minutes": (
            float(time_gaps.max())
            if not time_gaps.empty
            else None
        ),
    }

    return report


def validate_and_clean_h1_data(
    data: pd.DataFrame,
    symbol: str,
    expected_session: str,
    extreme_return_threshold: float = 0.02,
    extreme_range_threshold: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the complete H1 validation and cleaning process on IBKR-sourced bars.
    """
    flagged = add_validation_flags(
        data=data,
        expected_session=expected_session,
        extreme_return_threshold=extreme_return_threshold,
        extreme_range_threshold=extreme_range_threshold,
    )

    validation_report = create_validation_report(
        data=flagged,
        symbol=symbol,
    )

    cleaned, cleaning_summary = remove_mechanically_invalid_rows(
        flagged
    )

    cleaned = cleaned.set_index("timestamp")
    cleaned.index.name = "timestamp"

    # An IBKR-returned hourly bar is a complete unit IBKR itself asserts
    # represents that hour, unlike an M1-aggregated bar with partial
    # minute coverage, so every surviving bar is treated as fully eligible.
    cleaned["coverage_ratio"] = 1.0

    validation_report.update(cleaning_summary)

    return cleaned, validation_report
