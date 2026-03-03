"""
Enhanced session-liquidity strategy with multi-timeframe filters.

Approach 1: Smart Entry — after 1H sweep signal, use limit entry at a
  better price (pullback towards the sweep level). Tighter SL = better RR.

Approach 2: HTF Trend Filter — use higher-timeframe (4H/Daily) moving
  average or structure to filter signals that go against the trend.

Approach 3: Volatility-based SL — use ATR for dynamic SL sizing.
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


def get_default_mtf_params() -> Dict[str, Any]:
    """Return default enhanced strategy parameters."""
    return {
        # Which session sweeps to trade
        "trade_asian_sweep_in_london": True,
        "trade_asian_sweep_in_ny": True,
        "trade_london_sweep_in_ny": True,
        "trade_pdh_sweep": True,
        "trade_pdl_sweep": True,
        # Risk management
        "rr_ratio": 4.5,
        "sl_buffer_pct": 0.001,
        "use_session_sl": True,
        # Filters
        "min_sweep_wick_pct": 0.0003,
        "max_sl_points": 15.0,
        "min_sl_points": 3.0,
        "only_first_sweep": True,
        "require_pdl_pdh_confluence": False,
        "confluence_distance_pct": 0.003,
        # Time filters
        "no_trade_hours": [],
        "no_trade_friday_after": 18,
        # ---- Enhanced entry ----
        # Limit entry: enter at pullback towards the sweep level
        "use_limit_entry": True,
        "limit_entry_pct": 0.5,  # 0-1: 0=enter at close, 1=enter at sweep level
        "limit_fill_bars": 3,    # bars to wait for limit fill
        # ---- HTF trend filter ----
        "use_trend_filter": False,
        "trend_ma_period": 50,   # MA period on 1H for trend
        # ---- Volatility filter ----
        "use_atr_filter": False,
        "atr_period": 14,
        "atr_sl_multiplier": 1.5,
        "max_atr_multiplier": 3.0,  # skip if SL > this * ATR
        # ---- Engulfing confirmation ----
        "use_engulfing_filter": False,
        "engulfing_bars": 3,     # look for engulfing within N bars
    }


class MultiTimeframeStrategy:
    """Enhanced strategy with smart entry, trend filter, and volatility sizing."""

    def __init__(self, df_full: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = get_default_mtf_params()
        if params is not None:
            self.params.update(params)

        df = label_sessions(df_full)
        df = compute_pdl_pdh(df)

        # Pre-compute trend MA
        if self.params["use_trend_filter"]:
            period = self.params["trend_ma_period"]
            df["_trend_ma"] = df["close"].rolling(period, min_periods=period).mean()

        # Pre-compute ATR
        if self.params["use_atr_filter"]:
            period = self.params["atr_period"]
            high = df["high"]
            low = df["low"]
            close = df["close"]
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            df["_atr"] = tr.rolling(period, min_periods=period).mean()

        self.df: pd.DataFrame = df

        session_levels = compute_session_levels(df)
        self.session_levels = session_levels

        self._session_sweeps = detect_liquidity_sweeps(df, session_levels)
        self._pdl_pdh_sweeps = detect_pdl_pdh_sweeps(df)

        self._session_sweep_idx: Dict[pd.Timestamp, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in self._session_sweeps.iterrows():
            self._session_sweep_idx[row["timestamp"]].append(row.to_dict())

        self._pdl_pdh_sweep_idx: Dict[pd.Timestamp, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in self._pdl_pdh_sweeps.iterrows():
            self._pdl_pdh_sweep_idx[row["timestamp"]].append(row.to_dict())

        self._swept_levels_today: Dict[dt.date, set] = defaultdict(set)

        # Limit entry state
        self._pending_limit: Optional[Dict[str, Any]] = None
        self._pending_limit_bars: int = 0

    def generate_signal(
        self,
        current_idx: int,
        df_slice: pd.DataFrame,
        engine: Any,
    ) -> Optional[Dict[str, Any]]:
        params = self.params
        current_bar = self.df.iloc[current_idx]
        ts = self.df.index[current_idx]

        bar_open = current_bar["open"]
        bar_high = current_bar["high"]
        bar_low = current_bar["low"]
        bar_close = current_bar["close"]
        current_session = current_bar.get("session", "")
        current_date = ts.date()
        current_hour = ts.hour
        current_weekday = ts.weekday()

        # Time filters
        if current_hour in params["no_trade_hours"]:
            return None
        if current_weekday == 4 and current_hour >= params["no_trade_friday_after"]:
            return None

        # ============================================================
        # Check pending limit order
        # ============================================================
        if self._pending_limit is not None:
            self._pending_limit_bars -= 1

            filled = self._check_limit_fill(current_bar, current_idx)
            if filled is not None:
                self._pending_limit = None
                self._pending_limit_bars = 0
                return filled

            if self._pending_limit_bars <= 0:
                self._pending_limit = None
                self._pending_limit_bars = 0
                # Fall through

        # ============================================================
        # Engulfing confirmation for pending signal
        # ============================================================
        if hasattr(self, '_pending_engulfing') and self._pending_engulfing is not None:
            self._pending_engulfing_bars -= 1

            confirmed = self._check_engulfing(current_bar)
            if confirmed is not None:
                self._pending_engulfing = None
                self._pending_engulfing_bars = 0
                return confirmed

            if self._pending_engulfing_bars <= 0:
                self._pending_engulfing = None
                self._pending_engulfing_bars = 0

        # ============================================================
        # Look for new sweep signals
        # ============================================================
        session_sweep_events = self._session_sweep_idx.get(ts, [])
        pdl_pdh_sweep_events = self._pdl_pdh_sweep_idx.get(ts, [])

        if not session_sweep_events and not pdl_pdh_sweep_events:
            return None

        candidates: List[Dict[str, Any]] = []

        for evt in pdl_pdh_sweep_events:
            level_name = evt["level_name"]
            level_value = evt["level_value"]
            sweep_type = evt["sweep_type"]

            if level_name == "PDH" and not params["trade_pdh_sweep"]:
                continue
            if level_name == "PDL" and not params["trade_pdl_sweep"]:
                continue
            if level_name not in ("PDH", "PDL"):
                continue

            signal = self._build_signal(
                sweep_type=sweep_type,
                signal_type="pdl_pdh_sweep",
                level_name=level_name,
                level_value=level_value,
                swept_session="",
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
                current_date=current_date,
                current_session=current_session,
                ts=ts,
                current_idx=current_idx,
            )
            if signal is not None:
                candidates.append(signal)

        for evt in session_sweep_events:
            swept_session = evt["swept_session"]
            sweep_type = evt["sweep_type"]
            level_value = evt["level_value"]

            if not self._is_session_sweep_enabled(swept_session, current_session):
                continue

            signal = self._build_signal(
                sweep_type=sweep_type,
                signal_type="session_sweep",
                level_name="",
                level_value=level_value,
                swept_session=swept_session,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
                current_date=current_date,
                current_session=current_session,
                ts=ts,
                current_idx=current_idx,
            )
            if signal is not None:
                candidates.append(signal)

        if not candidates:
            return None

        filtered = [c for c in candidates if self._passes_filters(c, bar_close, current_date)]
        if not filtered:
            return None

        best = self._select_best(filtered)

        # ---- Apply trend filter ----
        if params["use_trend_filter"]:
            ma = current_bar.get("_trend_ma", np.nan)
            if not np.isnan(ma):
                if best["direction"] == "long" and bar_close < ma:
                    return None
                if best["direction"] == "short" and bar_close > ma:
                    return None

        # ---- Apply ATR filter ----
        if params["use_atr_filter"]:
            atr = current_bar.get("_atr", np.nan)
            if not np.isnan(atr):
                sl_dist = best["_sl_distance"]
                if sl_dist > atr * params["max_atr_multiplier"]:
                    return None

        # ---- Limit entry mode ----
        if params["use_limit_entry"]:
            self._pending_limit = best
            self._pending_limit_bars = params["limit_fill_bars"]
            return None

        # ---- Engulfing confirmation mode ----
        if params["use_engulfing_filter"]:
            self._pending_engulfing = best
            self._pending_engulfing_bars = params["engulfing_bars"]
            self._prev_bar = current_bar
            return None

        # ---- Immediate entry ----
        return self._clean_signal(best)

    def _check_limit_fill(self, bar: pd.Series, bar_idx: int) -> Optional[Dict[str, Any]]:
        """Check if current bar fills the pending limit order.

        IMPORTANT: The engine enters at the NEXT bar's open, not at the
        limit price. So we compute SL/TP from bar.close (best proxy for
        next bar's open) to avoid unrealistic R-multiples.
        """
        pending = self._pending_limit
        if pending is None:
            return None

        params = self.params
        direction = pending["direction"]
        level_value = pending["_level_value"]
        signal_close = pending["_signal_close"]

        # Limit price: between signal close and sweep level
        pct = params["limit_entry_pct"]
        if direction == "short":
            limit_price = signal_close + (level_value - signal_close) * pct
            # Did bar reach our limit price?
            if bar["high"] < limit_price:
                return None

            # Filled! Compute SL/TP from bar.close (proxy for next bar open)
            buffer = level_value * params["sl_buffer_pct"]
            stop_loss = max(bar["high"], level_value) + buffer
            entry_proxy = bar["close"]
            sl_distance = stop_loss - entry_proxy
            if sl_distance <= 0:
                return None
            take_profit = entry_proxy - sl_distance * params["rr_ratio"]

        else:  # long
            limit_price = signal_close - (signal_close - level_value) * pct
            if bar["low"] > limit_price:
                return None

            buffer = level_value * params["sl_buffer_pct"]
            stop_loss = min(bar["low"], level_value) - buffer
            entry_proxy = bar["close"]
            sl_distance = entry_proxy - stop_loss
            if sl_distance <= 0:
                return None
            take_profit = entry_proxy + sl_distance * params["rr_ratio"]

        if sl_distance > params["max_sl_points"]:
            return None
        if sl_distance < params["min_sl_points"]:
            return None

        return {
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": {
                **pending.get("metadata", {}),
                "entry_type": "limit_fill",
                "limit_price": limit_price,
                "sl_distance": sl_distance,
            },
        }

    def _check_engulfing(self, bar: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if current bar is an engulfing candle in signal direction."""
        pending = self._pending_engulfing
        if pending is None:
            return None

        direction = pending["direction"]
        prev = self._prev_bar

        if direction == "short":
            # Bearish engulfing: bar open > prev close, bar close < prev open
            is_engulfing = (bar["open"] >= prev["close"] and
                           bar["close"] <= prev["open"] and
                           bar["close"] < bar["open"])
        else:
            # Bullish engulfing: bar open < prev close, bar close > prev open
            is_engulfing = (bar["open"] <= prev["close"] and
                           bar["close"] >= prev["open"] and
                           bar["close"] > bar["open"])

        self._prev_bar = bar

        if not is_engulfing:
            return None

        # Recalculate SL/TP from engulfing bar
        params = self.params
        buffer = pending["_level_value"] * params["sl_buffer_pct"]

        if direction == "short":
            stop_loss = bar["high"] + buffer
            sl_distance = stop_loss - bar["close"]
            if sl_distance <= 0:
                return None
            take_profit = bar["close"] - sl_distance * params["rr_ratio"]
        else:
            stop_loss = bar["low"] - buffer
            sl_distance = bar["close"] - stop_loss
            if sl_distance <= 0:
                return None
            take_profit = bar["close"] + sl_distance * params["rr_ratio"]

        if sl_distance > params["max_sl_points"]:
            return None
        if sl_distance < params["min_sl_points"]:
            return None

        return {
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": {
                **pending.get("metadata", {}),
                "entry_type": "engulfing_confirm",
                "sl_distance": sl_distance,
            },
        }

    def _build_signal(
        self, sweep_type, signal_type, level_name, level_value,
        swept_session, bar_high, bar_low, bar_close,
        current_date, current_session, ts, current_idx,
    ) -> Optional[Dict[str, Any]]:
        params = self.params

        if sweep_type == "high_sweep":
            direction = "short"
        elif sweep_type == "low_sweep":
            direction = "long"
        else:
            return None

        if direction == "short":
            wick_distance = bar_high - level_value
        else:
            wick_distance = level_value - bar_low

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

        if signal_type == "pdl_pdh_sweep":
            level_key = f"{level_name}_{current_date.isoformat()}"
            priority = 0
        else:
            level_key = f"{swept_session}_{sweep_type}_{current_date.isoformat()}"
            priority = 2

        return {
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": {
                "signal_type": signal_type,
                "level_name": level_name,
                "level_value": level_value,
                "swept_session": swept_session,
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
            "_priority": priority,
            "_level_value": level_value,
            "_signal_close": bar_close,
        }

    def _is_session_sweep_enabled(self, swept_session, current_session):
        p = self.params
        if swept_session == "asian" and current_session in ("london", "overlap"):
            return p["trade_asian_sweep_in_london"]
        if swept_session == "asian" and current_session == "new_york":
            return p["trade_asian_sweep_in_ny"]
        if swept_session == "london" and current_session == "new_york":
            return p["trade_london_sweep_in_ny"]
        if swept_session == "overlap" and current_session == "new_york":
            return p["trade_london_sweep_in_ny"]
        return False

    def _passes_filters(self, signal, bar_close, current_date):
        params = self.params
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

        if params["only_first_sweep"]:
            if level_key in self._swept_levels_today[current_date]:
                return False
            self._swept_levels_today[current_date].add(level_key)

        return True

    def _select_best(self, candidates):
        candidates.sort(key=lambda c: (c["_priority"], c["_sl_distance"]))
        return candidates[0]

    def _clean_signal(self, signal):
        return {
            "direction": signal["direction"],
            "stop_loss": signal["stop_loss"],
            "take_profit": signal["take_profit"],
            "metadata": signal["metadata"],
        }

    def get_strategy_func(self):
        self._swept_levels_today = defaultdict(set)
        self._pending_limit = None
        self._pending_limit_bars = 0
        self._pending_engulfing = None
        self._pending_engulfing_bars = 0
        return self.generate_signal
