"""
Live strategy adapter.

Reuses the session-liquidity logic from the backtest module but operates
on a rolling window of recent candles fetched from the exchange.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.sessions import (
    compute_pdl_pdh,
    compute_session_levels,
    detect_liquidity_sweeps,
    detect_pdl_pdh_sweeps,
    label_sessions,
)

from .config import BotConfig

logger = logging.getLogger(__name__)


class LiveStrategy:
    """Session-liquidity strategy adapted for live trading.

    On each call to ``evaluate()``, it receives the latest kline DataFrame,
    re-computes session levels and sweeps, and returns a signal for the
    most recent *completed* bar (if any).
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._swept_levels_today: Dict[dt.date, set] = defaultdict(set)
        self._last_signal_ts: Optional[pd.Timestamp] = None

    def reset(self) -> None:
        """Reset internal state for a fresh simulation run."""
        self._swept_levels_today = defaultdict(set)
        self._last_signal_ts = None

    def _get_params(self) -> Dict[str, Any]:
        """Build strategy params dict from config."""
        return {
            "trade_asian_sweep_in_london": self.config.trade_asian_sweep_in_london,
            "trade_asian_sweep_in_ny": self.config.trade_asian_sweep_in_ny,
            "trade_london_sweep_in_ny": self.config.trade_london_sweep_in_ny,
            "trade_pdh_sweep": self.config.trade_pdh_sweep,
            "trade_pdl_sweep": self.config.trade_pdl_sweep,
            "rr_ratio": self.config.rr_ratio,
            "sl_buffer_pct": self.config.sl_buffer_pct,
            "use_session_sl": self.config.use_session_sl,
            "min_sweep_wick_pct": self.config.min_sweep_wick_pct,
            "max_sl_points": self.config.max_sl_points,
            "min_sl_points": self.config.min_sl_points,
            "only_first_sweep": self.config.only_first_sweep,
            "require_pdl_pdh_confluence": self.config.require_pdl_pdh_confluence,
            "confluence_distance_pct": self.config.confluence_distance_pct,
            "no_trade_hours": [],
            "no_trade_friday_after": self.config.no_trade_friday_after,
        }

    def evaluate(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Evaluate the latest completed candle for a trade signal.

        In live mode the very last bar is still forming, so we evaluate
        the second-to-last bar (``df.iloc[-2]``).
        """
        return self._evaluate_bar(df, bar_offset=-2)

    def evaluate_at(self, df: pd.DataFrame, bar_idx: int) -> Optional[Dict[str, Any]]:
        """Evaluate a specific bar index for a trade signal.

        Used by simulation tests so that both the backtest and the live
        strategy evaluate exactly the same bar.
        """
        return self._evaluate_bar(df, bar_idx=bar_idx)

    def _evaluate_bar(
        self,
        df: pd.DataFrame,
        bar_offset: Optional[int] = None,
        bar_idx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Core evaluation logic for a single bar."""
        if len(df) < 24:
            logger.warning("Not enough bars (%d) for strategy evaluation", len(df))
            return None

        params = self._get_params()

        # Enrich the data
        df_enriched = label_sessions(df)
        df_enriched = compute_pdl_pdh(df_enriched)
        session_levels = compute_session_levels(df_enriched)
        session_sweeps = detect_liquidity_sweeps(df_enriched, session_levels)
        pdl_pdh_sweeps = detect_pdl_pdh_sweeps(df_enriched)

        # Determine which bar to evaluate
        if bar_idx is not None:
            last_idx = bar_idx
        elif bar_offset is not None:
            last_idx = len(df_enriched) + bar_offset
        else:
            last_idx = len(df_enriched) - 2

        if last_idx < 0 or last_idx >= len(df_enriched):
            return None

        ts = df_enriched.index[last_idx]

        # Don't signal on the same bar twice
        if self._last_signal_ts is not None and ts <= self._last_signal_ts:
            return None

        current_bar = df_enriched.iloc[last_idx]
        bar_high = current_bar["high"]
        bar_low = current_bar["low"]
        bar_close = current_bar["close"]
        current_session = current_bar.get("session", "")
        current_date = ts.date()
        current_hour = ts.hour
        current_weekday = ts.weekday()

        # Time filters
        if current_hour in params.get("no_trade_hours", []):
            return None
        if current_weekday == 4 and current_hour >= params["no_trade_friday_after"]:
            return None

        # Look up sweeps on this timestamp
        sess_events = session_sweeps[session_sweeps["timestamp"] == ts].to_dict("records")
        pdl_events = pdl_pdh_sweeps[pdl_pdh_sweeps["timestamp"] == ts].to_dict("records")

        if not sess_events and not pdl_events:
            return None

        candidates: List[Dict[str, Any]] = []

        # PDH/PDL sweep signals
        for evt in pdl_events:
            signal = self._build_pdl_pdh_signal(
                evt, bar_high, bar_low, bar_close, current_date, current_session, ts, params
            )
            if signal:
                candidates.append(signal)

        # Session sweep signals
        for evt in sess_events:
            signal = self._build_session_signal(
                evt, bar_high, bar_low, bar_close, current_date, current_session,
                ts, params, df_enriched.iloc[: last_idx + 1],
            )
            if signal:
                candidates.append(signal)

        # Filter
        filtered = [c for c in candidates if self._passes_filters(c, bar_close, current_date, params)]

        if not filtered:
            return None

        # Pick best (tiebreaker: smallest SL distance = tightest risk)
        filtered.sort(key=lambda c: (c["_priority"], c["_sl_distance"]))
        best = filtered[0]

        self._last_signal_ts = ts

        logger.info(
            "SIGNAL: %s @ %s | SL=%.2f TP=%.2f | %s",
            best["direction"],
            ts,
            best["stop_loss"],
            best["take_profit"],
            best["metadata"].get("signal_type", ""),
        )

        return {
            "direction": best["direction"],
            "stop_loss": best["stop_loss"],
            "take_profit": best["take_profit"],
            "metadata": best["metadata"],
        }

    # ------------------------------------------------------------------
    # Signal builders (same logic as backtest strategy)
    # ------------------------------------------------------------------

    def _build_pdl_pdh_signal(
        self, evt, bar_high, bar_low, bar_close, current_date, current_session, ts, params,
    ) -> Optional[Dict[str, Any]]:
        level_name = evt["level_name"]
        level_value = evt["level_value"]
        sweep_type = evt["sweep_type"]

        if level_name == "PDH" and not params["trade_pdh_sweep"]:
            return None
        if level_name == "PDL" and not params["trade_pdl_sweep"]:
            return None
        if level_name not in ("PDH", "PDL"):
            return None

        if level_name == "PDH" and sweep_type == "high_sweep":
            direction = "short"
        elif level_name == "PDL" and sweep_type == "low_sweep":
            direction = "long"
        else:
            return None

        wick_distance = (bar_high - level_value) if direction == "short" else (level_value - bar_low)
        if wick_distance <= 0:
            return None

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
                "timestamp": str(ts),
            },
            "_sl_distance": sl_distance,
            "_wick_distance": wick_distance,
            "_level_key": level_key,
            "_priority": 0,
        }

    def _build_session_signal(
        self, evt, bar_high, bar_low, bar_close, current_date, current_session,
        ts, params, df_slice,
    ) -> Optional[Dict[str, Any]]:
        swept_session = evt["swept_session"]
        sweep_type = evt["sweep_type"]
        level_value = evt["level_value"]

        if not self._is_session_sweep_enabled(swept_session, current_session, params):
            return None

        direction = "short" if sweep_type == "high_sweep" else "long" if sweep_type == "low_sweep" else None
        if direction is None:
            return None

        wick_distance = (bar_high - level_value) if direction == "short" else (level_value - bar_low)
        if wick_distance <= 0:
            return None

        buffer = level_value * params["sl_buffer_pct"]

        if params["use_session_sl"]:
            stop_loss = (bar_high + buffer) if direction == "short" else (bar_low - buffer)
        else:
            if direction == "short":
                stop_loss = level_value + wick_distance + buffer
            else:
                stop_loss = level_value - wick_distance - buffer

        if direction == "short":
            sl_distance = stop_loss - bar_close
            take_profit = bar_close - sl_distance * params["rr_ratio"]
        else:
            sl_distance = bar_close - stop_loss
            take_profit = bar_close + sl_distance * params["rr_ratio"]

        if sl_distance <= 0:
            return None

        # Check confluence
        is_confluence = False
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
                "timestamp": str(ts),
            },
            "_sl_distance": sl_distance,
            "_wick_distance": wick_distance,
            "_level_key": level_key,
            "_priority": 1 if is_confluence else 2,
        }

    def _is_session_sweep_enabled(self, swept_session, current_session, params) -> bool:
        if swept_session == "asian" and current_session in ("london", "overlap"):
            return params["trade_asian_sweep_in_london"]
        if swept_session == "asian" and current_session == "new_york":
            return params["trade_asian_sweep_in_ny"]
        if swept_session == "london" and current_session == "new_york":
            return params["trade_london_sweep_in_ny"]
        if swept_session == "overlap" and current_session == "new_york":
            return params["trade_london_sweep_in_ny"]
        return False

    def _passes_filters(self, signal, bar_close, current_date, params) -> bool:
        sl_distance = signal["_sl_distance"]
        wick_distance = signal["_wick_distance"]
        level_key = signal["_level_key"]

        if sl_distance > params["max_sl_points"]:
            return False
        if sl_distance < params["min_sl_points"]:
            return False

        min_wick = bar_close * params["min_sweep_wick_pct"]
        if wick_distance < min_wick:
            return False

        if params["require_pdl_pdh_confluence"]:
            meta = signal["metadata"]
            if meta.get("signal_type") == "session_sweep" and not meta.get("is_confluence"):
                return False

        if params["only_first_sweep"]:
            if level_key in self._swept_levels_today[current_date]:
                return False
            self._swept_levels_today[current_date].add(level_key)

        return True
