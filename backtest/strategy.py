"""
Session-liquidity trading strategies for XAUUSD.

Trades based on which session's liquidity (high/low) gets swept in which
current session, combined with PDH/PDL levels.  All logic is strictly
causal -- the strategy function only ever sees data up to and including
the current bar.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .sessions import (
    compute_pdl_pdh,
    compute_session_levels,
    detect_liquidity_sweeps,
    detect_pdl_pdh_sweeps,
    label_sessions,
)


# ======================================================================
# Default parameters & parameter grid
# ======================================================================


def get_default_params() -> Dict[str, Any]:
    """Return default strategy parameters."""
    return {
        # Which session sweeps to trade
        "trade_asian_sweep_in_london": True,
        "trade_asian_sweep_in_ny": True,
        "trade_london_sweep_in_ny": True,
        "trade_pdh_sweep": True,
        "trade_pdl_sweep": True,
        # Risk management
        "rr_ratio": 2.0,
        "sl_buffer_pct": 0.001,
        "use_session_sl": True,
        # Filters
        "min_sweep_wick_pct": 0.0003,
        "max_sl_points": 15.0,
        "min_sl_points": 2.0,
        "only_first_sweep": True,
        "require_pdl_pdh_confluence": False,
        "confluence_distance_pct": 0.003,
        # Time filters
        "no_trade_hours": [],
        "no_trade_friday_after": 18,
    }


def get_param_grid() -> Dict[str, list]:
    """Return parameter grid for optimization.

    Each key maps to a list of values to try.
    """
    return {
        "rr_ratio": [1.5, 2.0, 2.5, 3.0, 4.0],
        "sl_buffer_pct": [0.0005, 0.001, 0.0015],
        "max_sl_points": [8.0, 12.0, 15.0, 20.0],
        "min_sl_points": [1.0, 2.0, 3.0],
        "min_sweep_wick_pct": [0.0001, 0.0003, 0.0005],
        "trade_asian_sweep_in_london": [True, False],
        "trade_asian_sweep_in_ny": [True, False],
        "trade_london_sweep_in_ny": [True, False],
        "trade_pdh_sweep": [True, False],
        "trade_pdl_sweep": [True, False],
        "use_session_sl": [True, False],
        "only_first_sweep": [True, False],
        "require_pdl_pdh_confluence": [True, False],
        "confluence_distance_pct": [0.001, 0.003, 0.005],
        "no_trade_friday_after": [15, 18, 21],
    }


# ======================================================================
# Session-liquidity strategy class
# ======================================================================


class SessionLiquidityStrategy:
    """XAUUSD strategy based on session-liquidity sweeps and PDH/PDL levels.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full OHLCV DataFrame with a UTC ``DatetimeIndex``.  Must include
        at least ``open``, ``high``, ``low``, ``close`` columns.
    params : dict
        Strategy parameters.  See :func:`get_default_params` for keys
        and their defaults.  Any missing key is filled from the defaults.
    """

    def __init__(self, df_full: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> None:
        # Merge caller params over defaults so every key is present.
        self.params: Dict[str, Any] = get_default_params()
        if params is not None:
            self.params.update(params)

        # ----------------------------------------------------------
        # Pre-computation (causal by construction of sessions.py)
        # ----------------------------------------------------------
        df = label_sessions(df_full)
        df = compute_pdl_pdh(df)
        self.df: pd.DataFrame = df

        session_levels = compute_session_levels(df)
        self.session_levels: pd.DataFrame = session_levels

        # Detect all sweep events across the full dataset.
        self._session_sweeps: pd.DataFrame = detect_liquidity_sweeps(df, session_levels)
        self._pdl_pdh_sweeps: pd.DataFrame = detect_pdl_pdh_sweeps(df)

        # Index sweep events by timestamp for O(1) lookup.
        self._session_sweep_idx: Dict[pd.Timestamp, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in self._session_sweeps.iterrows():
            self._session_sweep_idx[row["timestamp"]].append(row.to_dict())

        self._pdl_pdh_sweep_idx: Dict[pd.Timestamp, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in self._pdl_pdh_sweeps.iterrows():
            self._pdl_pdh_sweep_idx[row["timestamp"]].append(row.to_dict())

        # Track which levels have already been swept on each day
        # (populated at runtime inside generate_signal).
        self._swept_levels_today: Dict[dt.date, set] = defaultdict(set)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        current_idx: int,
        df_slice: pd.DataFrame,
        engine: Any,
    ) -> Optional[Dict[str, Any]]:
        """Strategy function compatible with ``BacktestEngine.run_strategy``.

        Parameters
        ----------
        current_idx : int
            Positional index of the current bar in the full DataFrame.
        df_slice : pd.DataFrame
            All bars from the start up to and including the current bar.
        engine : BacktestEngine
            The engine instance (used only for type compatibility; not
            referenced directly).

        Returns
        -------
        dict or None
            ``None`` if no trade, otherwise a dict with ``direction``,
            ``stop_loss``, ``take_profit``, and ``metadata``.
        """
        params = self.params

        # Use the strategy's enriched DataFrame (with session labels,
        # PDL/PDH) instead of df_slice which comes from the engine's
        # raw DataFrame and lacks session columns.
        current_bar = self.df.iloc[current_idx]
        ts: pd.Timestamp = self.df.index[current_idx]

        # ---- Current bar info ----------------------------------------
        bar_open: float = current_bar["open"]
        bar_high: float = current_bar["high"]
        bar_low: float = current_bar["low"]
        bar_close: float = current_bar["close"]
        current_session: str = current_bar.get("session", "")
        current_date: dt.date = ts.date()
        current_hour: int = ts.hour
        current_weekday: int = ts.weekday()  # 0=Mon … 4=Fri

        # ---- Time filters --------------------------------------------
        if current_hour in params["no_trade_hours"]:
            return None

        if current_weekday == 4 and current_hour >= params["no_trade_friday_after"]:
            return None

        # ---- Look up sweep events for this bar -----------------------
        session_sweep_events: List[Dict[str, Any]] = self._session_sweep_idx.get(ts, [])
        pdl_pdh_sweep_events: List[Dict[str, Any]] = self._pdl_pdh_sweep_idx.get(ts, [])

        if not session_sweep_events and not pdl_pdh_sweep_events:
            return None

        # ---- Collect candidate signals -------------------------------
        candidates: List[Dict[str, Any]] = []

        # --- PDH/PDL sweep signals (higher priority) ------------------
        for evt in pdl_pdh_sweep_events:
            level_name: str = evt["level_name"]
            level_value: float = evt["level_value"]
            sweep_type: str = evt["sweep_type"]

            if level_name == "PDH" and not params["trade_pdh_sweep"]:
                continue
            if level_name == "PDL" and not params["trade_pdl_sweep"]:
                continue
            # Skip PWH/PWL -- the params only cover PDH/PDL explicitly.
            if level_name not in ("PDH", "PDL"):
                continue

            signal = self._build_pdl_pdh_signal(
                sweep_type=sweep_type,
                level_name=level_name,
                level_value=level_value,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
                current_date=current_date,
                current_session=current_session,
                ts=ts,
            )
            if signal is not None:
                candidates.append(signal)

        # --- Session sweep signals ------------------------------------
        for evt in session_sweep_events:
            swept_session: str = evt["swept_session"]
            sweep_type = evt["sweep_type"]
            level_value = evt["level_value"]

            if not self._is_session_sweep_enabled(swept_session, current_session):
                continue

            signal = self._build_session_signal(
                sweep_type=sweep_type,
                swept_session=swept_session,
                level_value=level_value,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
                current_date=current_date,
                current_session=current_session,
                ts=ts,
                df_slice=self.df.iloc[: current_idx + 1],
            )
            if signal is not None:
                candidates.append(signal)

        if not candidates:
            return None

        # ---- Apply filters to each candidate -------------------------
        filtered: List[Dict[str, Any]] = []
        for cand in candidates:
            if not self._passes_filters(cand, bar_close, current_date):
                continue
            filtered.append(cand)

        if not filtered:
            return None

        # ---- Prioritize ----------------------------------------------
        best = self._select_best_signal(filtered)
        return best

    # ------------------------------------------------------------------
    # Internal: build signal dicts
    # ------------------------------------------------------------------

    def _build_pdl_pdh_signal(
        self,
        sweep_type: str,
        level_name: str,
        level_value: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        current_date: dt.date,
        current_session: str,
        ts: pd.Timestamp,
    ) -> Optional[Dict[str, Any]]:
        """Build a trade signal from a PDH or PDL sweep event."""
        params = self.params

        # PDH swept -> SHORT; PDL swept -> LONG
        if level_name == "PDH" and sweep_type == "high_sweep":
            direction = "short"
        elif level_name == "PDL" and sweep_type == "low_sweep":
            direction = "long"
        else:
            return None

        # Compute wick distance (how far price pierced beyond the level).
        if direction == "short":
            wick_distance = bar_high - level_value
        else:
            wick_distance = level_value - bar_low

        if wick_distance <= 0:
            return None

        # SL: beyond the sweep wick + buffer
        buffer = level_value * params["sl_buffer_pct"]
        if direction == "short":
            stop_loss = bar_high + buffer
            sl_distance = stop_loss - bar_close
            take_profit = bar_close - sl_distance * params["rr_ratio"]
        else:
            stop_loss = bar_low - buffer
            sl_distance = bar_close - stop_loss
            take_profit = bar_close + sl_distance * params["rr_ratio"]

        if sl_distance <= 0:
            return None

        level_key = f"{level_name}_{current_date.isoformat()}"

        return {
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": {
                "signal_type": "pdl_pdh_sweep",
                "level_name": level_name,
                "level_value": level_value,
                "sweep_type": sweep_type,
                "session": current_session,
                "wick_distance": wick_distance,
                "sl_distance": sl_distance,
                "timestamp": ts,
                "is_confluence": False,
            },
            "_sl_distance": sl_distance,
            "_wick_distance": wick_distance,
            "_level_key": level_key,
            "_priority": 0,  # highest priority
        }

    def _build_session_signal(
        self,
        sweep_type: str,
        swept_session: str,
        level_value: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        current_date: dt.date,
        current_session: str,
        ts: pd.Timestamp,
        df_slice: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """Build a trade signal from a session-liquidity sweep event."""
        params = self.params

        # Determine direction:
        # High swept (price went above, closed below) -> SHORT
        # Low swept  (price went below, closed above) -> LONG
        if sweep_type == "high_sweep":
            direction = "short"
        elif sweep_type == "low_sweep":
            direction = "long"
        else:
            return None

        # Wick distance beyond the level.
        if direction == "short":
            wick_distance = bar_high - level_value
        else:
            wick_distance = level_value - bar_low

        if wick_distance <= 0:
            return None

        # Stop loss computation.
        buffer = level_value * params["sl_buffer_pct"]

        if params["use_session_sl"]:
            # Use the sweep candle's extreme as SL reference.
            if direction == "short":
                stop_loss = bar_high + buffer
            else:
                stop_loss = bar_low - buffer
        else:
            # Use the swept session level + buffer.
            if direction == "short":
                stop_loss = level_value + wick_distance + buffer
            else:
                stop_loss = level_value - wick_distance - buffer

        # SL distance and TP.
        if direction == "short":
            sl_distance = stop_loss - bar_close
            take_profit = bar_close - sl_distance * params["rr_ratio"]
        else:
            sl_distance = bar_close - stop_loss
            take_profit = bar_close + sl_distance * params["rr_ratio"]

        if sl_distance <= 0:
            return None

        # Check for confluence with PDH/PDL.
        is_confluence = False
        if params["require_pdl_pdh_confluence"] or True:
            # Always compute confluence flag; the filter decides later
            # whether to enforce it.
            last_row = df_slice.iloc[-1]
            pdh = last_row.get("PDH", np.nan)
            pdl = last_row.get("PDL", np.nan)
            conf_thresh = level_value * params["confluence_distance_pct"]

            if not np.isnan(pdh) and abs(level_value - pdh) <= conf_thresh:
                is_confluence = True
            if not np.isnan(pdl) and abs(level_value - pdl) <= conf_thresh:
                is_confluence = True

        level_key = f"{swept_session}_{sweep_type}_{current_date.isoformat()}"

        return {
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": {
                "signal_type": "session_sweep",
                "swept_session": swept_session,
                "current_session": current_session,
                "level_value": level_value,
                "sweep_type": sweep_type,
                "wick_distance": wick_distance,
                "sl_distance": sl_distance,
                "is_confluence": is_confluence,
                "timestamp": ts,
            },
            "_sl_distance": sl_distance,
            "_wick_distance": wick_distance,
            "_level_key": level_key,
            "_priority": 1 if is_confluence else 2,
        }

    # ------------------------------------------------------------------
    # Internal: filters
    # ------------------------------------------------------------------

    def _is_session_sweep_enabled(self, swept_session: str, current_session: str) -> bool:
        """Check whether the strategy params allow this session sweep combination."""
        params = self.params

        # Asian liquidity swept in London or Overlap session.
        if swept_session == "asian" and current_session in ("london", "overlap"):
            return params["trade_asian_sweep_in_london"]

        # Asian liquidity swept in NY session.
        if swept_session == "asian" and current_session == "new_york":
            return params["trade_asian_sweep_in_ny"]

        # London liquidity swept in NY session.
        if swept_session == "london" and current_session == "new_york":
            return params["trade_london_sweep_in_ny"]

        # Overlap liquidity swept in NY session (treat overlap as London).
        if swept_session == "overlap" and current_session == "new_york":
            return params["trade_london_sweep_in_ny"]

        return False

    def _passes_filters(
        self,
        signal: Dict[str, Any],
        bar_close: float,
        current_date: dt.date,
    ) -> bool:
        """Return True if the candidate signal passes all parameter filters."""
        params = self.params
        sl_distance: float = signal["_sl_distance"]
        wick_distance: float = signal["_wick_distance"]
        level_key: str = signal["_level_key"]

        # Min/max SL distance in price points.
        if sl_distance > params["max_sl_points"]:
            return False
        if sl_distance < params["min_sl_points"]:
            return False

        # Minimum wick beyond the level.
        min_wick = bar_close * params["min_sweep_wick_pct"]
        if wick_distance < min_wick:
            return False

        # Confluence requirement (session sweeps only).
        if params["require_pdl_pdh_confluence"]:
            meta = signal["metadata"]
            if meta.get("signal_type") == "session_sweep" and not meta.get("is_confluence", False):
                return False

        # Only trade the first sweep of a given level per day.
        if params["only_first_sweep"]:
            if level_key in self._swept_levels_today[current_date]:
                return False
            self._swept_levels_today[current_date].add(level_key)

        return True

    def _select_best_signal(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the highest-priority signal when multiple are valid.

        Priority order:
        0 -- PDH/PDL sweeps
        1 -- Session sweeps with confluence
        2 -- Session sweeps without confluence
        """
        candidates.sort(key=lambda c: (c["_priority"], c["_sl_distance"]))
        best = candidates[0]

        # Strip internal keys before returning.
        return {
            "direction": best["direction"],
            "stop_loss": best["stop_loss"],
            "take_profit": best["take_profit"],
            "metadata": best["metadata"],
        }

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------

    def get_strategy_func(self):
        """Return a bound strategy function compatible with ``engine.run_strategy()``.

        Returns
        -------
        callable
            ``(current_idx, df_slice, engine) -> dict | None``
        """
        # Reset daily tracking state so a fresh run starts clean.
        self._swept_levels_today = defaultdict(set)
        return self.generate_signal

    def run(self, engine: Any = None) -> list:
        """Convenience method: run the strategy.

        Parameters
        ----------
        engine : BacktestEngine, optional
            An initialised backtest engine.  If ``None``, one is created
            from the strategy's enriched DataFrame.

        Returns
        -------
        list[Trade]
            All completed trades in chronological order.
        """
        if engine is None:
            from .engine import BacktestEngine as _BE
            engine = _BE(self.df)
        strategy_func = self.get_strategy_func()
        return engine.run_strategy(strategy_func)
