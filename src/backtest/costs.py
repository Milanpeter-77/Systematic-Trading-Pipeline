from __future__ import annotations

import numpy as np
import pandas as pd


BASIS_POINTS_PER_UNIT = 10_000.0


def commission_fraction(
    commission_bps_per_side: float,
) -> float:
    """
    Convert a per-side commission from basis points to return fraction.
    """
    if commission_bps_per_side < 0:
        raise ValueError(
            "commission_bps_per_side cannot be negative."
        )

    return commission_bps_per_side / BASIS_POINTS_PER_UNIT


def calculate_transaction_costs(
    position_change: pd.Series,
    fill_spread_fraction: pd.Series,
    commission_bps_per_side: float = 0.5,
) -> pd.DataFrame:
    """
    Calculate transaction costs for each position change.

    Parameters
    ----------
    position_change
        Change in executed position. Examples:
        0 -> +1 gives +1 and one trading side.
        +1 -> -1 gives -2 and two trading sides.
    fill_spread_fraction
        Full bid-ask spread as a fraction of the fill price.
    commission_bps_per_side
        Commission charged on each trading side.

    Returns
    -------
    pd.DataFrame
        Number of trading sides, spread cost, commission, and total cost.
    """
    if not position_change.index.equals(
        fill_spread_fraction.index
    ):
        raise ValueError(
            "position_change and fill_spread_fraction "
            "must have identical indices."
        )

    if fill_spread_fraction.isna().any():
        raise ValueError(
            "fill_spread_fraction contains missing values."
        )

    if (fill_spread_fraction < 0).any():
        raise ValueError(
            "fill_spread_fraction contains negative values."
        )

    trading_sides = position_change.abs().astype(float)

    half_spread_per_side = fill_spread_fraction / 2.0

    spread_cost = trading_sides * half_spread_per_side

    commission_per_side = commission_fraction(
        commission_bps_per_side
    )

    commission_cost = trading_sides * commission_per_side

    total_cost = spread_cost + commission_cost

    return pd.DataFrame(
        {
            "trading_sides": trading_sides,
            "spread_cost": spread_cost,
            "commission_cost": commission_cost,
            "transaction_cost": total_cost,
        },
        index=position_change.index,
    )