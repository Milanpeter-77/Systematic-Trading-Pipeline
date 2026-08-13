from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    calculate_drawdown,
    calculate_return_metrics,
    infer_periods_per_year,
)
from src.backtest.result import BacktestResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StressWindow:
    """
    Definition of one historical stress interval.

    End timestamps are inclusive.
    """

    stress_id: str
    stress_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    market_return: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stress_id": self.stress_id,
            "stress_type": self.stress_type,
            "stress_start": self.start,
            "stress_end": self.end,
            "market_return": self.market_return,
        }


def calculate_daily_market_returns(
    market_data: pd.DataFrame,
) -> pd.Series:
    """
    Construct daily open-to-open market returns from H1 data.
    """
    if "open" not in market_data.columns:
        raise ValueError(
            "market_data must contain an 'open' column."
        )

    daily_open = (
        market_data["open"]
        .resample("1D")
        .first()
        .dropna()
    )

    daily_return = daily_open.pct_change(
        fill_method=None
    )

    daily_return.name = "daily_market_return"

    return daily_return.dropna()


def calculate_rolling_market_returns(
    daily_returns: pd.Series,
    window_days: int,
) -> pd.Series:
    """
    Calculate compounded rolling market returns.
    """
    if window_days <= 1:
        raise ValueError(
            "window_days must be greater than one."
        )

    rolling_return = (
        1.0 + daily_returns
    ).rolling(
        window=window_days,
        min_periods=window_days,
    ).apply(
        np.prod,
        raw=True,
    ) - 1.0

    rolling_return.name = (
        f"rolling_{window_days}d_market_return"
    )

    return rolling_return.dropna()


def identify_worst_market_windows(
    market_data: pd.DataFrame,
    window_days: int = 30,
    window_count: int = 3,
    minimum_separation_days: int = 30,
) -> list[StressWindow]:
    """
    Identify the worst non-overlapping rolling market-return windows.

    Selection is based only on the underlying instrument return, not on
    strategy performance.
    """
    if window_count <= 0:
        raise ValueError(
            "window_count must be positive."
        )

    if minimum_separation_days < 0:
        raise ValueError(
            "minimum_separation_days cannot be negative."
        )

    daily_returns = calculate_daily_market_returns(
        market_data
    )

    rolling_returns = calculate_rolling_market_returns(
        daily_returns=daily_returns,
        window_days=window_days,
    )

    ranked = rolling_returns.sort_values(
        ascending=True
    )

    selected: list[StressWindow] = []

    for end_date, market_return in ranked.items():
        start_date = (
            end_date
            - pd.Timedelta(days=window_days - 1)
        )

        overlaps_existing = any(
            not (
                end_date
                < existing.start
                - pd.Timedelta(
                    days=minimum_separation_days
                )
                or start_date
                > existing.end
                + pd.Timedelta(
                    days=minimum_separation_days
                )
            )
            for existing in selected
        )

        if overlaps_existing:
            continue

        selected.append(
            StressWindow(
                stress_id=(
                    f"worst_market_window_"
                    f"{len(selected) + 1}"
                ),
                stress_type="worst_market_window",
                start=pd.Timestamp(start_date),
                end=(
                    pd.Timestamp(end_date)
                    + pd.Timedelta(
                        hours=23,
                        minutes=59,
                        seconds=59,
                    )
                ),
                market_return=float(market_return),
            )
        )

        if len(selected) == window_count:
            break

    if len(selected) < window_count:
        raise ValueError(
            "Unable to identify the requested number "
            "of separated worst market windows."
        )

    return selected


def build_historical_stress_windows(
    market_data: pd.DataFrame,
    layer3_config: dict[str, Any],
) -> list[StressWindow]:
    """
    Create the fixed March 2020 window and data-driven worst windows.
    """
    march_window = StressWindow(
        stress_id="march_2020",
        stress_type="fixed_historical",
        start=pd.Timestamp(
            layer3_config["march_2020_start"]
        ),
        end=pd.Timestamp(
            layer3_config["march_2020_end"]
        ),
        market_return=None,
    )

    worst_windows = identify_worst_market_windows(
        market_data=market_data,
        window_days=int(
            layer3_config["worst_window_days"]
        ),
        window_count=int(
            layer3_config["worst_window_count"]
        ),
        minimum_separation_days=int(
            layer3_config[
                "minimum_worst_window_separation_days"
            ]
        ),
    )

    return [march_window, *worst_windows]


def slice_stress_window(
    timeseries: pd.DataFrame,
    stress_window: StressWindow,
) -> pd.DataFrame:
    """
    Extract one historical stress interval.
    """
    stress_data = timeseries.loc[
        (
            timeseries.index
            >= stress_window.start
        )
        & (
            timeseries.index
            <= stress_window.end
        )
    ].copy()

    if stress_data.empty:
        raise ValueError(
            f"No observations found for stress window "
            f"{stress_window.stress_id}."
        )

    return stress_data


def calculate_stress_metrics(
    returns: pd.Series,
    prefix: str,
) -> dict[str, float | int]:
    """
    Calculate performance metrics for one stress return series.
    """
    clean_returns = returns.dropna()

    if len(clean_returns) < 2:
        return {
            f"{prefix}_observation_count": int(
                len(clean_returns)
            ),
            f"{prefix}_total_return": np.nan,
            f"{prefix}_annual_return": np.nan,
            f"{prefix}_annual_volatility": np.nan,
            f"{prefix}_sharpe": np.nan,
            f"{prefix}_max_drawdown": np.nan,
            f"{prefix}_terminal_nav": np.nan,
            f"{prefix}_worst_bar_return": np.nan,
        }

    periods_per_year = infer_periods_per_year(
        clean_returns.index
    )

    metrics = calculate_return_metrics(
        returns=clean_returns,
        periods_per_year=periods_per_year,
        prefix=prefix,
    )

    drawdown = calculate_drawdown(
        clean_returns
    )

    metrics.update(
        {
            f"{prefix}_observation_count": int(
                len(clean_returns)
            ),
            f"{prefix}_terminal_nav": float(
                drawdown["nav"].iloc[-1]
            ),
            f"{prefix}_worst_bar_return": float(
                clean_returns.min()
            ),
        }
    )

    return metrics


def evaluate_historical_stress_window(
    result: BacktestResult,
    stress_window: StressWindow,
) -> dict[str, Any]:
    """
    Evaluate one candidate inside one historical stress interval.
    """
    stress_data = slice_stress_window(
        timeseries=result.timeseries,
        stress_window=stress_window,
    )

    metrics = calculate_stress_metrics(
        returns=stress_data["net_return"],
        prefix="stress_net",
    )

    return {
        "candidate_id": result.candidate.candidate_id,
        "family": result.candidate.family,
        "symbol": result.candidate.symbol,
        **result.candidate.parameters,
        **stress_window.to_dict(),
        "first_observation": stress_data.index.min(),
        "last_observation": stress_data.index.max(),
        **metrics,
        "stress_transaction_cost": float(
            stress_data["transaction_cost"].sum()
        ),
        "stress_spread_cost": float(
            stress_data["spread_cost"].sum()
        ),
        "stress_commission_cost": float(
            stress_data["commission_cost"].sum()
        ),
    }


def build_synthetic_stress_returns(
    timeseries: pd.DataFrame,
    return_multiplier: float,
    spread_multiplier: float,
) -> pd.DataFrame:
    """
    Stress strategy P&L and bid-ask spread costs.

    Gross strategy returns are multiplied by return_multiplier.
    Spread costs are multiplied by spread_multiplier.
    Commission remains unchanged.
    """
    required_columns = {
        "gross_return",
        "spread_cost",
        "commission_cost",
    }

    missing = required_columns.difference(
        timeseries.columns
    )

    if missing:
        raise ValueError(
            f"Synthetic stress input is missing: "
            f"{sorted(missing)}"
        )

    if return_multiplier <= 0:
        raise ValueError(
            "return_multiplier must be positive."
        )

    if spread_multiplier <= 0:
        raise ValueError(
            "spread_multiplier must be positive."
        )

    stressed = timeseries.copy()

    stressed["synthetic_gross_return"] = (
        stressed["gross_return"]
        * return_multiplier
    )

    stressed["synthetic_spread_cost"] = (
        stressed["spread_cost"]
        * spread_multiplier
    )

    stressed["synthetic_commission_cost"] = (
        stressed["commission_cost"]
    )

    stressed["synthetic_transaction_cost"] = (
        stressed["synthetic_spread_cost"]
        + stressed["synthetic_commission_cost"]
    )

    stressed["synthetic_net_return"] = (
        stressed["synthetic_gross_return"]
        - stressed["synthetic_transaction_cost"]
    )

    return stressed


def evaluate_synthetic_stress(
    result: BacktestResult,
    layer3_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Evaluate the full backtest under simultaneous return and spread stress.
    """
    stressed = build_synthetic_stress_returns(
        timeseries=result.timeseries,
        return_multiplier=float(
            layer3_config[
                "return_stress_multiplier"
            ]
        ),
        spread_multiplier=float(
            layer3_config[
                "spread_stress_multiplier"
            ]
        ),
    )

    metrics = calculate_stress_metrics(
        returns=stressed[
            "synthetic_net_return"
        ],
        prefix="synthetic",
    )

    record = {
        "candidate_id": result.candidate.candidate_id,
        "family": result.candidate.family,
        "symbol": result.candidate.symbol,
        **result.candidate.parameters,
        "return_stress_multiplier": float(
            layer3_config[
                "return_stress_multiplier"
            ]
        ),
        "spread_stress_multiplier": float(
            layer3_config[
                "spread_stress_multiplier"
            ]
        ),
        **metrics,
        "synthetic_total_spread_cost": float(
            stressed[
                "synthetic_spread_cost"
            ].sum()
        ),
        "synthetic_total_commission_cost": float(
            stressed[
                "synthetic_commission_cost"
            ].sum()
        ),
        "synthetic_total_transaction_cost": float(
            stressed[
                "synthetic_transaction_cost"
            ].sum()
        ),
    }

    return record, stressed


def evaluate_layer3_criteria(
    historical_results: pd.DataFrame,
    synthetic_record: dict[str, Any],
    layer3_config: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Apply pre-specified historical and synthetic stress criteria.
    """
    failures: list[str] = []

    march_rows = historical_results.loc[
        historical_results["stress_id"]
        == "march_2020"
    ]

    if len(march_rows) != 1:
        failures.append("missing_march_2020")
        march_return = np.nan
        march_drawdown = np.nan
    else:
        march_row = march_rows.iloc[0]

        march_return = float(
            march_row[
                "stress_net_total_return"
            ]
        )

        march_drawdown = float(
            march_row[
                "stress_net_max_drawdown"
            ]
        )

        if (
            not np.isfinite(march_return)
            or march_return
            < layer3_config[
                "minimum_march_2020_net_return"
            ]
        ):
            failures.append(
                "march_2020_net_return"
            )

        if (
            not np.isfinite(march_drawdown)
            or march_drawdown
            < -abs(
                layer3_config[
                    "maximum_march_2020_drawdown"
                ]
            )
        ):
            failures.append(
                "march_2020_drawdown"
            )

    worst_rows = historical_results.loc[
        historical_results["stress_type"]
        == "worst_market_window"
    ]

    if worst_rows.empty:
        failures.append(
            "missing_worst_market_windows"
        )

        worst_historical_return = np.nan
        worst_historical_drawdown = np.nan
    else:
        worst_historical_return = float(
            worst_rows[
                "stress_net_total_return"
            ].min()
        )

        worst_historical_drawdown = float(
            worst_rows[
                "stress_net_max_drawdown"
            ].min()
        )

        if (
            not np.isfinite(
                worst_historical_return
            )
            or worst_historical_return
            < layer3_config[
                "minimum_worst_window_net_return"
            ]
        ):
            failures.append(
                "worst_window_net_return"
            )

        if (
            not np.isfinite(
                worst_historical_drawdown
            )
            or worst_historical_drawdown
            < -abs(
                layer3_config[
                    "maximum_worst_window_drawdown"
                ]
            )
        ):
            failures.append(
                "worst_window_drawdown"
            )

    synthetic_return = float(
        synthetic_record[
            "synthetic_total_return"
        ]
    )

    synthetic_drawdown = float(
        synthetic_record[
            "synthetic_max_drawdown"
        ]
    )

    synthetic_terminal_nav = float(
        synthetic_record[
            "synthetic_terminal_nav"
        ]
    )

    if (
        not np.isfinite(synthetic_return)
        or synthetic_return
        < layer3_config[
            "minimum_synthetic_total_return"
        ]
    ):
        failures.append(
            "synthetic_total_return"
        )

    if (
        not np.isfinite(synthetic_drawdown)
        or synthetic_drawdown
        < -abs(
            layer3_config[
                "maximum_synthetic_drawdown"
            ]
        )
    ):
        failures.append(
            "synthetic_drawdown"
        )

    if (
        not np.isfinite(
            synthetic_terminal_nav
        )
        or synthetic_terminal_nav
        < layer3_config[
            "minimum_synthetic_terminal_nav"
        ]
    ):
        failures.append(
            "synthetic_terminal_nav"
        )

    summary = {
        "march_2020_net_return": march_return,
        "march_2020_max_drawdown": march_drawdown,
        "worst_historical_net_return": (
            worst_historical_return
        ),
        "worst_historical_max_drawdown": (
            worst_historical_drawdown
        ),
        "synthetic_net_total_return": (
            synthetic_return
        ),
        "synthetic_net_max_drawdown": (
            synthetic_drawdown
        ),
        "synthetic_terminal_nav": (
            synthetic_terminal_nav
        ),
    }

    return len(failures) == 0, failures, summary


def evaluate_layer3_candidate(
    result: BacktestResult,
    market_data: pd.DataFrame,
    layer3_config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate one candidate under Layer 3.
    """
    historical_windows = (
        build_historical_stress_windows(
            market_data=market_data,
            layer3_config=layer3_config,
        )
    )

    historical_records = [
        evaluate_historical_stress_window(
            result=result,
            stress_window=stress_window,
        )
        for stress_window in historical_windows
    ]

    historical_results = pd.DataFrame(
        historical_records
    )

    synthetic_record, synthetic_timeseries = (
        evaluate_synthetic_stress(
            result=result,
            layer3_config=layer3_config,
        )
    )

    (
        layer3_pass,
        failures,
        summary,
    ) = evaluate_layer3_criteria(
        historical_results=historical_results,
        synthetic_record=synthetic_record,
        layer3_config=layer3_config,
    )

    record = {
        **result.candidate.to_dict(),
        "historical_stress_count": int(
            len(historical_results)
        ),
        **summary,
        "layer3_pass": layer3_pass,
        "failure_reason": (
            ";".join(failures)
            if failures
            else ""
        ),
    }

    synthetic_summary = pd.DataFrame(
        [synthetic_record]
    )

    return (
        record,
        historical_results,
        synthetic_summary,
    )


def run_layer3_gate(
    backtest_results: dict[
        str,
        BacktestResult,
    ],
    processed_data: dict[
        str,
        pd.DataFrame,
    ],
    layer3_config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate all official candidates under Layer 3.
    """
    candidate_records: list[
        dict[str, Any]
    ] = []

    historical_frames: list[
        pd.DataFrame
    ] = []

    synthetic_frames: list[
        pd.DataFrame
    ] = []

    for candidate_id, result in (
        backtest_results.items()
    ):
        logger.debug(f"Layer 3: {candidate_id}")

        symbol = result.candidate.symbol

        (
            candidate_record,
            historical_results,
            synthetic_results,
        ) = evaluate_layer3_candidate(
            result=result,
            market_data=processed_data[
                symbol
            ],
            layer3_config=layer3_config,
        )

        candidate_records.append(
            candidate_record
        )

        historical_frames.append(
            historical_results
        )

        synthetic_frames.append(
            synthetic_results
        )

    layer3_results = pd.DataFrame(
        candidate_records
    )

    all_historical_results = pd.concat(
        historical_frames,
        ignore_index=True,
    )

    all_synthetic_results = pd.concat(
        synthetic_frames,
        ignore_index=True,
    )

    if len(layer3_results) != len(
        backtest_results
    ):
        raise RuntimeError(
            "Layer 3 did not produce one row "
            "per official candidate."
        )

    if not layer3_results[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "Layer 3 produced duplicate "
            "candidate IDs."
        )

    return (
        layer3_results,
        all_historical_results,
        all_synthetic_results,
    )