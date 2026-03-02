"""
Session identification and liquidity analysis for XAUUSD trading.

All times are UTC. All functions are strictly causal (no look-ahead bias).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Session time boundaries (UTC, expressed as hours)
# ---------------------------------------------------------------------------

SESSION_DEFINITIONS: dict[str, tuple[int, int]] = {
    "asian":    (0,  8),
    "london":   (8,  16),
    "new_york": (13, 22),
    "overlap":  (13, 16),   # London / NY overlap
}


# ---------------------------------------------------------------------------
# 1. label_sessions
# ---------------------------------------------------------------------------

def label_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``session`` column to *df*.

    A candle is labelled ``'overlap'`` if it falls inside **both** the London
    and New York sessions (13:00-16:00 UTC).  Otherwise it receives the label
    of the single session it belongs to.  Candles outside every defined window
    receive ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a :class:`~pandas.DatetimeIndex` (UTC) named
        ``timestamp`` or be the index itself.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an added ``session`` column.
    """
    df = df.copy()
    hour = df.index.hour

    # Default to NaN – will be overwritten where a session matches.
    session = pd.Series(np.nan, index=df.index, dtype="object")

    # Assign sessions from broadest to most specific so that 'overlap' wins
    # when a candle falls in both London and NY.
    # Asian: 00:00 – 08:00  (hour 0..7)
    session[(hour >= 0) & (hour < 8)] = "asian"

    # London: 08:00 – 16:00  (hour 8..15)
    session[(hour >= 8) & (hour < 16)] = "london"

    # New York: 13:00 – 22:00  (hour 13..21)
    # Only label as new_york when NOT already in the overlap zone.
    session[(hour >= 16) & (hour < 22)] = "new_york"

    # Overlap: 13:00 – 16:00  (hour 13..15)
    session[(hour >= 13) & (hour < 16)] = "overlap"

    df["session"] = session
    return df


# ---------------------------------------------------------------------------
# 2. compute_session_levels
# ---------------------------------------------------------------------------

def compute_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session OHLC levels for each (date, session).

    **No look-ahead bias**: a session's levels are only populated *after* the
    session ends.  During the session, no row for that (date, session)
    combination is produced.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain a ``session`` column (call :func:`label_sessions`
        first) and have a UTC :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    pd.DataFrame
        One row per (date, session) with columns:
        ``date``, ``session``, ``session_open``, ``session_high``,
        ``session_low``, ``session_close``, ``session_end_ts``
        (the timestamp at which the session closed — useful for causal joins).
    """
    df_work = df.copy()
    df_work["date"] = df_work.index.date

    # Drop rows with no session label.
    df_work = df_work.dropna(subset=["session"])

    # Session end hours (exclusive upper bound).
    session_end_hour: dict[str, int] = {
        "asian":    8,
        "london":   16,
        "overlap":  16,
        "new_york": 22,
    }

    records: list[dict] = []

    for (date, session_name), grp in df_work.groupby(["date", "session"]):
        # Sort chronologically within the group.
        grp = grp.sort_index()

        # Determine the theoretical session-end timestamp.
        end_hour = session_end_hour[session_name]
        session_end_ts = pd.Timestamp(
            year=date.year, month=date.month, day=date.day,
            hour=end_hour, tz="UTC",
        )

        # Only emit levels if the session has fully ended within the data.
        # That means the last timestamp in *the entire DataFrame* must be
        # >= session_end_ts.
        if df.index.max() < session_end_ts:
            continue

        records.append(
            {
                "date":          date,
                "session":       session_name,
                "session_open":  grp["open"].iloc[0],
                "session_high":  grp["high"].max(),
                "session_low":   grp["low"].min(),
                "session_close": grp["close"].iloc[-1],
                "session_end_ts": session_end_ts,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "date", "session", "session_open", "session_high",
                "session_low", "session_close", "session_end_ts",
            ]
        )

    result = pd.DataFrame(records)
    result = result.sort_values(["date", "session_end_ts", "session"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 3. compute_pdl_pdh  (previous day / previous week highs & lows)
# ---------------------------------------------------------------------------

def compute_pdl_pdh(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``PDH``, ``PDL``, ``PWH``, ``PWL`` columns to *df*.

    * **PDH / PDL** – previous *calendar* day's high / low (computed over the
      entire day, 00:00-23:59 UTC).  On Monday the previous day is Friday (or
      the last trading day with data).
    * **PWH / PWL** – previous ISO-week's high / low.

    Strictly causal: the previous-day levels only become available once a new
    calendar day starts; similarly for weeks.

    Parameters
    ----------
    df : pd.DataFrame
        UTC DatetimeIndex with OHLCV columns.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the four new columns appended.
    """
    df = df.copy()
    df["_date"] = df.index.date

    # --- daily highs / lows ---------------------------------------------------
    daily_hl = (
        df.groupby("_date")
        .agg(day_high=("high", "max"), day_low=("low", "min"))
    )
    # Shift by one row so that each date gets the *previous* day's values.
    daily_hl["PDH"] = daily_hl["day_high"].shift(1)
    daily_hl["PDL"] = daily_hl["day_low"].shift(1)

    # --- weekly highs / lows --------------------------------------------------
    # ISO week: Monday = 1 … Sunday = 7.
    # We key on (iso_year, iso_week).
    df["_iso_year"] = df.index.isocalendar().year.values
    df["_iso_week"] = df.index.isocalendar().week.values

    weekly_hl = (
        df.groupby(["_iso_year", "_iso_week"])
        .agg(week_high=("high", "max"), week_low=("low", "min"))
    )
    weekly_hl["PWH"] = weekly_hl["week_high"].shift(1)
    weekly_hl["PWL"] = weekly_hl["week_low"].shift(1)

    # Map the daily values back onto each candle.
    daily_map = daily_hl[["PDH", "PDL"]]
    df["PDH"] = df["_date"].map(daily_map["PDH"])
    df["PDL"] = df["_date"].map(daily_map["PDL"])

    # Map the weekly values back onto each candle.
    week_key = list(zip(df["_iso_year"], df["_iso_week"]))
    pwh_map = weekly_hl["PWH"].to_dict()
    pwl_map = weekly_hl["PWL"].to_dict()
    df["PWH"] = [pwh_map.get(k, np.nan) for k in week_key]
    df["PWL"] = [pwl_map.get(k, np.nan) for k in week_key]

    # Clean up helper columns.
    df.drop(columns=["_date", "_iso_year", "_iso_week"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# 4. detect_liquidity_sweeps  (session-level sweeps)
# ---------------------------------------------------------------------------

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    session_levels: pd.DataFrame,
    max_lookback_days: int = 2,
) -> pd.DataFrame:
    """Detect candles that sweep a prior session high or low.

    A **high sweep** occurs when a candle's *high* exceeds a prior session
    high but the candle *closes* back below that level.

    A **low sweep** occurs when a candle's *low* goes below a prior session
    low but the candle *closes* back above that level.

    Only levels whose session has already ended (``session_end_ts`` <=
    candle timestamp) are considered, ensuring strict causality.

    Only session levels from the last ``max_lookback_days`` calendar days
    are checked for each candle, because a session-based strategy only
    cares about recent liquidity (today's Asian swept in London, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        The OHLCV DataFrame with a ``session`` column.
    session_levels : pd.DataFrame
        Output of :func:`compute_session_levels`.
    max_lookback_days : int
        Maximum number of calendar days to look back for session levels.
        Default is 2 (today + yesterday).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``sweep_type``, ``swept_session``,
        ``swept_date``, ``current_session``, ``level_value``.
    """
    if session_levels.empty:
        return pd.DataFrame(
            columns=[
                "timestamp", "sweep_type", "swept_session",
                "swept_date", "current_session", "level_value",
            ]
        )

    # Ensure session_levels is sorted deterministically (overlap and london
    # share 16:00 UTC end time, so add session name as tiebreaker).
    sl = session_levels.sort_values(["session_end_ts", "session"]).reset_index(drop=True)

    # Pre-convert to numpy for speed.
    sl_end_ts = sl["session_end_ts"].values.astype("datetime64[ns]")
    sl_session = sl["session"].values
    sl_date = sl["date"].values
    sl_high = sl["session_high"].values.astype(float)
    sl_low = sl["session_low"].values.astype(float)

    lookback_td = np.timedelta64(max_lookback_days, "D")

    records: list[dict] = []

    # Use numpy arrays for the candle data for speed.
    candle_highs = df["high"].values
    candle_lows = df["low"].values
    candle_closes = df["close"].values
    candle_sessions = df["session"].values if "session" in df.columns else [np.nan] * len(df)
    timestamps = df.index.values.astype("datetime64[ns]")

    for bar_i in range(len(df)):
        ts = timestamps[bar_i]
        candle_high = candle_highs[bar_i]
        candle_low = candle_lows[bar_i]
        candle_close = candle_closes[bar_i]
        current_session = candle_sessions[bar_i]

        # Only consider session levels that:
        # 1. Ended before this candle
        # 2. Are within max_lookback_days
        cutoff = ts - lookback_td
        mask = (sl_end_ts <= ts) & (sl_end_ts >= cutoff)

        avail_idx = np.where(mask)[0]
        if len(avail_idx) == 0:
            continue

        for i in avail_idx:
            level_high = sl_high[i]
            level_low = sl_low[i]

            # High sweep: candle pierces above session high, closes below it.
            if candle_high > level_high and candle_close < level_high:
                records.append(
                    {
                        "timestamp":       pd.Timestamp(ts, tz="UTC"),
                        "sweep_type":      "high_sweep",
                        "swept_session":   sl_session[i],
                        "swept_date":      sl_date[i],
                        "current_session": current_session,
                        "level_value":     level_high,
                    }
                )

            # Low sweep: candle pierces below session low, closes above it.
            if candle_low < level_low and candle_close > level_low:
                records.append(
                    {
                        "timestamp":       pd.Timestamp(ts, tz="UTC"),
                        "sweep_type":      "low_sweep",
                        "swept_session":   sl_session[i],
                        "swept_date":      sl_date[i],
                        "current_session": current_session,
                        "level_value":     level_low,
                    }
                )

    result = pd.DataFrame(
        records,
        columns=[
            "timestamp", "sweep_type", "swept_session",
            "swept_date", "current_session", "level_value",
        ],
    )
    return result


# ---------------------------------------------------------------------------
# 5. detect_pdl_pdh_sweeps  (daily / weekly level sweeps)
# ---------------------------------------------------------------------------

def detect_pdl_pdh_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect candles that sweep PDH / PDL / PWH / PWL levels.

    The input *df* must already contain the ``PDH``, ``PDL``, ``PWH``, ``PWL``
    columns (call :func:`compute_pdl_pdh` first).

    Sweep logic is identical to :func:`detect_liquidity_sweeps`:
    * High-side levels (PDH, PWH): candle high > level **and** close < level.
    * Low-side levels (PDL, PWL): candle low < level **and** close > level.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``PDH``, ``PDL``,
        ``PWH``, ``PWL`` and a ``session`` column.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``sweep_type``, ``level_name``,
        ``level_value``, ``current_session``.
    """
    required = {"PDH", "PDL", "PWH", "PWL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing columns: {missing}. "
            "Call compute_pdl_pdh() first."
        )

    # Define which levels are high-side vs low-side.
    high_side_levels = ["PDH", "PWH"]
    low_side_levels = ["PDL", "PWL"]

    records: list[dict] = []

    # Use numpy arrays for speed.
    candle_highs = df["high"].values
    candle_lows = df["low"].values
    candle_closes = df["close"].values
    candle_sessions = df["session"].values if "session" in df.columns else [np.nan] * len(df)
    timestamps = df.index

    for level_name in high_side_levels:
        level_values = df[level_name].values
        for i in range(len(df)):
            lv = level_values[i]
            if np.isnan(lv):
                continue
            if candle_highs[i] > lv and candle_closes[i] < lv:
                records.append(
                    {
                        "timestamp":       timestamps[i],
                        "sweep_type":      "high_sweep",
                        "level_name":      level_name,
                        "level_value":     lv,
                        "current_session": candle_sessions[i],
                    }
                )

    for level_name in low_side_levels:
        level_values = df[level_name].values
        for i in range(len(df)):
            lv = level_values[i]
            if np.isnan(lv):
                continue
            if candle_lows[i] < lv and candle_closes[i] > lv:
                records.append(
                    {
                        "timestamp":       timestamps[i],
                        "sweep_type":      "low_sweep",
                        "level_name":      level_name,
                        "level_value":     lv,
                        "current_session": candle_sessions[i],
                    }
                )

    result = pd.DataFrame(
        records,
        columns=[
            "timestamp", "sweep_type", "level_name",
            "level_value", "current_session",
        ],
    )
    return result
