from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


DEFAULT_COINTEGRATION_P_VALUE_THRESHOLD = 0.05
MINIMUM_OVERLAPPING_OBSERVATIONS = 100


def screen_cointegrated_pairs(
    processed_data: dict[str, pd.DataFrame],
    p_value_threshold: float = DEFAULT_COINTEGRATION_P_VALUE_THRESHOLD,
) -> list[tuple[str, str]]:
    """
    Engle-Granger cointegration screen across every pair of instruments.

    Tests every combination of the given symbols' close-price series and
    keeps pairs whose cointegration test p-value is below threshold --
    i.e. pairs whose price relationship is statistically likely to be
    mean-reverting, the property a pairs-trading strategy depends on. No
    asset-class pre-filtering: every combination is tested and the
    statistics decide. Returns pairs as (symbol_a, symbol_b) tuples with
    symbol_a < symbol_b alphabetically, for a deterministic pair name.
    """
    symbols = sorted(processed_data)

    pairs: list[tuple[str, str]] = []

    for symbol_a, symbol_b in combinations(symbols, 2):
        aligned = pd.concat(
            [
                processed_data[symbol_a]["close"].rename("a"),
                processed_data[symbol_b]["close"].rename("b"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if len(aligned) < MINIMUM_OVERLAPPING_OBSERVATIONS:
            continue

        _, p_value, _ = coint(aligned["a"], aligned["b"])

        if p_value < p_value_threshold:
            pairs.append((symbol_a, symbol_b))

    return pairs


def compute_pair_spread(
    price_a: pd.Series,
    price_b: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Compute the Engle-Granger spread between two aligned price series.

    Estimates the hedge ratio via OLS (price_a regressed on price_b, with
    an intercept) over the full overlapping history, then returns the
    regression residual: price_a - intercept - hedge_ratio * price_b. This
    is the exact series screen_cointegrated_pairs's coint() test checked
    for stationarity, so it's the correct series for a pairs strategy to
    mean-revert against (oscillates around its own local mean, not raw 0).

    Also returns the hedge ratio itself -- the caller needs it to weight
    each leg's own bid-ask cost into a combined fill_spread_fraction for
    the pair (trading a pair spread means trading both legs, so both
    legs' spreads are real costs, unlike a single-instrument strategy).
    """
    aligned = pd.concat(
        [price_a.rename("a"), price_b.rename("b")],
        axis=1,
        join="inner",
    ).dropna()

    design = np.column_stack(
        [np.ones(len(aligned)), aligned["b"].to_numpy()]
    )

    intercept, hedge_ratio = np.linalg.lstsq(
        design, aligned["a"].to_numpy(), rcond=None
    )[0]

    spread = aligned["a"] - intercept - hedge_ratio * aligned["b"]

    return spread, float(hedge_ratio)
