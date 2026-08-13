from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import calculate_transaction_costs
from src.backtest.execution import build_executed_positions
from src.backtest.metrics import (
    calculate_backtest_metrics,
    calculate_drawdown,
)
from src.backtest.result import BacktestResult
from src.factory.candidate import CandidateSpec
from src.factory.registry import create_strategy


def extract_trades(
    backtest_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract complete position episodes from executed positions.

    Trade returns are compounded from the hourly net and gross returns
    occurring while the position is active.
    """
    data = backtest_data.copy()

    position_change = data["executed_position"].ne(
        data["executed_position"].shift()
    )

    data["trade_group"] = position_change.cumsum()

    active = data.loc[
        data["executed_position"] != 0
    ].copy()

    records: list[dict[str, object]] = []

    for _, trade_data in active.groupby("trade_group"):
        direction = int(
            trade_data["executed_position"].iloc[0]
        )

        entry_timestamp = trade_data.index[0]
        exit_timestamp = trade_data.index[-1]

        entry_price = float(
            trade_data["open"].iloc[0]
        )

        exit_price = float(
            trade_data["next_open"].iloc[-1]
        )

        gross_trade_return = float(
            (1.0 + trade_data["gross_return"]).prod()
            - 1.0
        )

        net_trade_return = float(
            (1.0 + trade_data["net_return"]).prod()
            - 1.0
        )

        holding_hours = (
            trade_data.index[-1]
            - trade_data.index[0]
        ).total_seconds() / 3600 + 1.0

        records.append(
            {
                "entry_timestamp": entry_timestamp,
                "exit_timestamp": exit_timestamp,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "holding_hours": holding_hours,
                "gross_trade_return": gross_trade_return,
                "net_trade_return": net_trade_return,
                "trade_cost": float(
                    trade_data["transaction_cost"].sum()
                ),
            }
        )

    return pd.DataFrame(records)


def run_candidate_backtest(
    candidate: CandidateSpec,
    market_data: pd.DataFrame,
    commission_bps_per_side: float = 0.5,
    minimum_signal_coverage: float = 0.80,
    minimum_fill_coverage: float = 0.80,
) -> BacktestResult:
    """
    Run one strategy candidate through the execution and cost engine.
    """
    required_market_columns = {
        "open",
        "high",
        "low",
        "close",
        "coverage_ratio",
        "fill_spread_fraction",
    }

    missing = required_market_columns.difference(
        market_data.columns
    )

    if missing:
        raise ValueError(
            f"Market data is missing columns: {sorted(missing)}"
        )

    strategy = create_strategy(
        family=candidate.family,
        parameters=candidate.parameters,
    )

    strategy_output = strategy.generate_positions(
        market_data
    )

    executed = build_executed_positions(
        strategy_output=strategy_output,
        minimum_signal_coverage=minimum_signal_coverage,
        minimum_fill_coverage=minimum_fill_coverage,
        initial_position=0,
    )

    executed["next_open"] = executed["open"].shift(-1)

    executed["open_to_open_return"] = (
        executed["next_open"] / executed["open"] - 1.0
    )

    executed["gross_return"] = (
        executed["executed_position"]
        * executed["open_to_open_return"]
    )

    cost_data = calculate_transaction_costs(
        position_change=executed["position_change"],
        fill_spread_fraction=executed[
            "fill_spread_fraction"
        ],
        commission_bps_per_side=commission_bps_per_side,
    )

    executed = executed.join(cost_data)

    executed["net_return"] = (
        executed["gross_return"]
        - executed["transaction_cost"]
    )

    # The final row has no next opening price and cannot earn a return.
    backtest_data = executed.loc[
        executed["next_open"].notna()
    ].copy()

    gross_drawdown = calculate_drawdown(
        backtest_data["gross_return"]
    )

    net_drawdown = calculate_drawdown(
        backtest_data["net_return"]
    )

    backtest_data["gross_nav"] = gross_drawdown["nav"]
    backtest_data["gross_drawdown"] = (
        gross_drawdown["drawdown"]
    )

    backtest_data["net_nav"] = net_drawdown["nav"]
    backtest_data["net_drawdown"] = (
        net_drawdown["drawdown"]
    )

    trades = extract_trades(backtest_data)

    metrics = calculate_backtest_metrics(
        backtest_data=backtest_data,
        trades=trades,
    )

    return BacktestResult(
        candidate=candidate,
        timeseries=backtest_data,
        trades=trades,
        metrics=metrics,
    )