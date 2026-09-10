from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, matching this repo's existing convention
    (adjust=False, min_periods=span) already used inline by every trend/
    momentum strategy that smooths a series."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(high-low, |high-prev_close|, |low-prev_close|),
    lifted verbatim from volatility/atr_breakout.py's existing inline
    formula so it isn't re-typed in every new strategy that needs it."""
    previous_close = close.shift(1)

    return pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Average True Range as a simple rolling mean of true_range --
    deliberately matches volatility/atr_breakout.py's existing (non-Wilder)
    convention exactly, so this repo doesn't end up with two inconsistent
    ATR definitions."""
    return true_range(high, low, close).rolling(
        window=window,
        min_periods=window,
    ).mean()


def bollinger_bands(
    close: pd.Series,
    window: int,
    num_std: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling mean/upper/lower Bollinger Bands, matching the close-to-close
    standard-deviation construction already used inline by volatility/
    breakout.py. Returns (mean, upper, lower)."""
    rolling_mean = close.rolling(window=window, min_periods=window).mean()
    rolling_std = close.rolling(window=window, min_periods=window).std(ddof=0)

    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std

    return rolling_mean, upper, lower


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    atr_multiple: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel: an EMA midline banded by a multiple of ATR instead
    of close-to-close standard deviation. Returns (mid_ema, upper, lower)."""
    mid_ema = ema(close, window)
    band_atr = atr(high, low, close, window)

    upper = mid_ema + atr_multiple * band_atr
    lower = mid_ema - atr_multiple * band_atr

    return mid_ema, upper, lower


def macd(
    close: pd.Series,
    fast_span: int,
    slow_span: int,
    signal_span: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Classic (raw, non-normalized) MACD: macd_line = fast EMA - slow EMA,
    signal_line = EMA of macd_line, histogram = macd_line - signal_line.

    Deliberately NOT the Percentage Price Oscillator (PPO) variant that
    divides by the slow EMA: a raw EMA difference is shift-invariant
    (EMA(x + c) - EMA(x + c) == EMA(x) - EMA(x) for a constant shift c),
    which matters for stat_arb/macd_spread.py -- it applies this to a pair
    spread that prepare_pair_features() shifts by an arbitrary constant to
    keep it positive, and a PPO-style normalization would make the signal
    depend on that arbitrary shift (the same shift-invariance concern
    stat_arb/return_zscore.py's docstring already documents for its own
    use of diff() instead of pct_change()). Callers that need a magnitude
    threshold comparable across instruments at different price levels
    (e.g. momentum/macd_histogram.py) normalize the output themselves
    rather than baking that into this shared helper.

    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = ema(close, fast_span)
    slow_ema = ema(close, slow_span)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_span)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling volume-weighted average price over `window` bars, using the
    typical price (H+L+C)/3 as the per-bar price weighted by volume."""
    typical_price = (high + low + close) / 3

    weighted_price_sum = (typical_price * volume).rolling(
        window=window,
        min_periods=window,
    ).sum()

    volume_sum = volume.rolling(window=window, min_periods=window).sum()
    valid_volume_sum = volume_sum.where(volume_sum > 0)

    return weighted_price_sum / valid_volume_sum


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index, plus/minus Directional Indicators.

    Every internal smoothing step (true range, +DM, -DM, and the final
    DX-to-ADX smoothing) uses a simple rolling mean via this module's own
    atr() helper -- a deliberate deviation from Wilder's original
    recursive smoothing, kept consistent with atr()'s already-established
    non-Wilder convention (matching volatility/atr_breakout.py) rather
    than introducing a second, different smoothing convention for this
    one indicator.

    Returns (plus_di, minus_di, adx_value).
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    smoothed_tr = atr(high, low, close, window)
    smoothed_plus_dm = plus_dm.rolling(window=window, min_periods=window).mean()
    smoothed_minus_dm = minus_dm.rolling(window=window, min_periods=window).mean()

    valid_tr = smoothed_tr.where(smoothed_tr > 0)

    plus_di = 100 * smoothed_plus_dm / valid_tr
    minus_di = 100 * smoothed_minus_dm / valid_tr

    di_sum = plus_di + minus_di
    valid_di_sum = di_sum.where(di_sum > 0)
    dx = 100 * (plus_di - minus_di).abs() / valid_di_sum

    adx_value = dx.rolling(window=window, min_periods=window).mean()

    return plus_di, minus_di, adx_value


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    acceleration_step: float,
    acceleration_max: float,
) -> pd.Series:
    """
    Wilder's Parabolic SAR, returning a direction series (+1/-1, 0 before
    the indicator is initialized on the first two bars).

    Unlike every other function in this module, SAR is a recursive
    indicator -- sar[t] depends on sar[t-1], the running trend direction,
    the trend's running extreme point, and an accelerating factor -- so it
    cannot be vectorized with .rolling()/.ewm() and is computed with one
    explicit bar-by-bar loop instead.
    """
    n = len(close)
    direction = np.zeros(n, dtype=np.int8)

    if n < 2:
        return pd.Series(direction, index=close.index)

    high_values = high.to_numpy()
    low_values = low.to_numpy()

    trend_up = close.iloc[1] > close.iloc[0]
    sar = low_values[0] if trend_up else high_values[0]
    extreme_point = high_values[1] if trend_up else low_values[1]
    acceleration = acceleration_step

    direction[1] = 1 if trend_up else -1

    for index in range(2, n):
        sar_candidate = sar + acceleration * (extreme_point - sar)

        if trend_up:
            sar_candidate = min(
                sar_candidate, low_values[index - 1], low_values[index - 2]
            )
        else:
            sar_candidate = max(
                sar_candidate, high_values[index - 1], high_values[index - 2]
            )

        if trend_up and low_values[index] < sar_candidate:
            trend_up = False
            sar = extreme_point
            extreme_point = low_values[index]
            acceleration = acceleration_step
        elif not trend_up and high_values[index] > sar_candidate:
            trend_up = True
            sar = extreme_point
            extreme_point = high_values[index]
            acceleration = acceleration_step
        else:
            sar = sar_candidate

            if trend_up and high_values[index] > extreme_point:
                extreme_point = high_values[index]
                acceleration = min(acceleration + acceleration_step, acceleration_max)
            elif not trend_up and low_values[index] < extreme_point:
                extreme_point = low_values[index]
                acceleration = min(acceleration + acceleration_step, acceleration_max)

        direction[index] = 1 if trend_up else -1

    return pd.Series(direction, index=close.index)
