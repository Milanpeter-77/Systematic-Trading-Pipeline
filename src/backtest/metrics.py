from __future__ import annotations

import numpy as np
import pandas as pd


def infer_periods_per_year(
    index: pd.DatetimeIndex,
) -> float:
    """
    Estimate the number of observations per calendar year from the sample.
    """
    if len(index) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    elapsed_years = (
        index[-1] - index[0]
    ).total_seconds() / (
        365.25 * 24 * 60 * 60
    )

    if elapsed_years <= 0:
        raise ValueError(
            "Backtest index has a non-positive duration."
        )

    return len(index) / elapsed_years


def calculate_drawdown(
    returns: pd.Series,
) -> pd.DataFrame:
    """
    Calculate NAV, running maximum, and drawdown.
    """
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0

    return pd.DataFrame(
        {
            "nav": nav,
            "running_max": running_max,
            "drawdown": drawdown,
        },
        index=returns.index,
    )


def calculate_return_metrics(
    returns: pd.Series,
    periods_per_year: float,
    prefix: str,
) -> dict[str, float]:
    """
    Calculate standardized return and risk metrics.
    """
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return {
            f"{prefix}_total_return": np.nan,
            f"{prefix}_annual_return": np.nan,
            f"{prefix}_annual_volatility": np.nan,
            f"{prefix}_sharpe": np.nan,
            f"{prefix}_max_drawdown": np.nan,
            f"{prefix}_calmar": np.nan,
        }

    total_return = (
        (1.0 + clean_returns).prod() - 1.0
    )

    sample_years = len(clean_returns) / periods_per_year

    if sample_years > 0 and total_return > -1:
        annual_return = (
            (1.0 + total_return) ** (1.0 / sample_years)
            - 1.0
        )
    else:
        annual_return = np.nan

    standard_deviation = clean_returns.std(ddof=1)

    annual_volatility = (
        standard_deviation * np.sqrt(periods_per_year)
        if np.isfinite(standard_deviation)
        else np.nan
    )

    mean_return = clean_returns.mean()

    sharpe = (
        mean_return / standard_deviation
        * np.sqrt(periods_per_year)
        if standard_deviation > 0
        else np.nan
    )

    drawdown_data = calculate_drawdown(clean_returns)
    max_drawdown = float(
        drawdown_data["drawdown"].min()
    )

    calmar = (
        annual_return / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    return {
        f"{prefix}_total_return": float(total_return),
        f"{prefix}_annual_return": float(annual_return),
        f"{prefix}_annual_volatility": float(
            annual_volatility
        ),
        f"{prefix}_sharpe": float(sharpe),
        f"{prefix}_max_drawdown": max_drawdown,
        f"{prefix}_calmar": float(calmar),
    }


def calculate_backtest_metrics(
    backtest_data: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Calculate candidate-level gross, net, cost, and activity metrics.
    """
    required_columns = {
        "gross_return",
        "net_return",
        "transaction_cost",
        "spread_cost",
        "commission_cost",
        "executed_position",
        "position_change",
    }

    missing = required_columns.difference(
        backtest_data.columns
    )

    if missing:
        raise ValueError(
            f"Backtest metrics input is missing: {sorted(missing)}"
        )

    periods_per_year = infer_periods_per_year(
        backtest_data.index
    )

    metrics: dict[str, float | int] = {
        "periods_per_year": float(periods_per_year),
        "observation_count": int(len(backtest_data)),
        "trade_count": int(len(trades)),
        "position_change_count": int(
            (backtest_data["position_change"] != 0).sum()
        ),
        "total_trading_sides": float(
            backtest_data["position_change"].abs().sum()
        ),
        "average_absolute_exposure": float(
            backtest_data["executed_position"].abs().mean()
        ),
        "long_exposure_fraction": float(
            (backtest_data["executed_position"] == 1).mean()
        ),
        "short_exposure_fraction": float(
            (backtest_data["executed_position"] == -1).mean()
        ),
        "flat_fraction": float(
            (backtest_data["executed_position"] == 0).mean()
        ),
        "total_transaction_cost": float(
            backtest_data["transaction_cost"].sum()
        ),
        "total_spread_cost": float(
            backtest_data["spread_cost"].sum()
        ),
        "total_commission_cost": float(
            backtest_data["commission_cost"].sum()
        ),
    }

    metrics.update(
        calculate_return_metrics(
            returns=backtest_data["gross_return"],
            periods_per_year=periods_per_year,
            prefix="gross",
        )
    )

    metrics.update(
        calculate_return_metrics(
            returns=backtest_data["net_return"],
            periods_per_year=periods_per_year,
            prefix="net",
        )
    )

    gross_profit = backtest_data[
        "gross_return"
    ].clip(lower=0).sum()

    metrics["cost_to_gross_profit_ratio"] = (
        float(
            backtest_data["transaction_cost"].sum()
            / gross_profit
        )
        if gross_profit > 0
        else np.nan
    )

    if not trades.empty:
        metrics.update(
            {
                "trade_win_rate": float(
                    (trades["net_trade_return"] > 0).mean()
                ),
                "average_net_trade_return": float(
                    trades["net_trade_return"].mean()
                ),
                "median_net_trade_return": float(
                    trades["net_trade_return"].median()
                ),
                "average_holding_hours": float(
                    trades["holding_hours"].mean()
                ),
                "maximum_holding_hours": float(
                    trades["holding_hours"].max()
                ),
            }
        )
    else:
        metrics.update(
            {
                "trade_win_rate": np.nan,
                "average_net_trade_return": np.nan,
                "median_net_trade_return": np.nan,
                "average_holding_hours": np.nan,
                "maximum_holding_hours": np.nan,
            }
        )

    return metrics