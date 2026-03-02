"""
Simulation test: verify live bot signal generation matches backtest.

Uses 6 months of historical XAUUSD data:
  - First 3 months = warmup (rolling window builds context)
  - Last 3 months = validation (every signal must match backtest EXACTLY)

Tests both:
  A. Direct mode (full data) — proves logic is identical
  B. Rolling window (200-bar sliding window) — proves real live conditions work
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.engine import BacktestEngine
from backtest.strategy import SessionLiquidityStrategy, get_default_params
from trading_bot.config import BotConfig
from trading_bot.live_strategy import LiveStrategy


# ======================================================================
# Helpers
# ======================================================================

def _make_config() -> BotConfig:
    bt = get_default_params()
    return BotConfig(
        api_key="test", api_secret="test", risk_per_trade_usd=100.0, testnet=True,
        rr_ratio=bt["rr_ratio"], sl_buffer_pct=bt["sl_buffer_pct"],
        max_sl_points=bt["max_sl_points"], min_sl_points=bt["min_sl_points"],
        min_sweep_wick_pct=bt["min_sweep_wick_pct"],
        trade_asian_sweep_in_london=bt["trade_asian_sweep_in_london"],
        trade_asian_sweep_in_ny=bt["trade_asian_sweep_in_ny"],
        trade_london_sweep_in_ny=bt["trade_london_sweep_in_ny"],
        trade_pdh_sweep=bt["trade_pdh_sweep"], trade_pdl_sweep=bt["trade_pdl_sweep"],
        use_session_sl=bt["use_session_sl"], only_first_sweep=bt["only_first_sweep"],
        require_pdl_pdh_confluence=bt["require_pdl_pdh_confluence"],
        confluence_distance_pct=bt["confluence_distance_pct"],
        no_trade_friday_after=bt["no_trade_friday_after"],
    )


def _load_data() -> pd.DataFrame:
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "backtest", "data", "xauusd_1h.csv",
    )
    if not os.path.exists(csv_path):
        pytest.skip(f"Data not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _sim_trade(df, idx, direction, sl, tp, spread=0.30, max_bars=48):
    """Simulate a trade bar-by-bar, same logic as BacktestEngine.

    Entry is at bar[idx+1].open (next bar after signal), matching the
    engine's fixed look-ahead-free entry.
    """
    entry_idx = idx + 1
    if entry_idx >= len(df):
        return None

    raw = df.iloc[entry_idx]["open"]
    ep = raw + spread / 2 if direction == "long" else raw - spread / 2
    risk = abs(ep - sl) or 1e-9
    end = min(entry_idx + 1 + max_bars, len(df))
    for i in range(entry_idx + 1, end):
        h, l = df.iloc[i]["high"], df.iloc[i]["low"]
        if direction == "long":
            sl_hit, tp_hit = l <= sl, h >= tp
        else:
            sl_hit, tp_hit = h >= sl, l <= tp
        if sl_hit and tp_hit:
            xp = sl
            pnl = (xp - ep) / risk if direction == "long" else (ep - xp) / risk
            return {"pnl_r": pnl, "exit_idx": i}
        if sl_hit:
            xp = sl
            pnl = (xp - ep) / risk if direction == "long" else (ep - xp) / risk
            return {"pnl_r": pnl, "exit_idx": i}
        if tp_hit:
            xp = tp
            pnl = (xp - ep) / risk if direction == "long" else (ep - xp) / risk
            return {"pnl_r": pnl, "exit_idx": i}
    xp = df.iloc[end - 1]["close"]
    pnl = (xp - ep) / risk if direction == "long" else (ep - xp) / risk
    return {"pnl_r": pnl, "exit_idx": end - 1}


# ======================================================================
# Shared fixtures
# ======================================================================

VALIDATION_START = pd.Timestamp("2021-06-01", tz="UTC")


@pytest.fixture(scope="module")
def data_6m():
    """Load 6 months: 3-month warmup + 3-month validation."""
    df = _load_data()
    df_6m = df["2021-03-01":"2021-09-01"]
    if len(df_6m) < 2000:
        pytest.skip(f"Not enough data ({len(df_6m)} bars)")
    return df_6m


@pytest.fixture(scope="module")
def backtest_results(data_6m):
    """Run backtest on full 6 months, collect signals and trades."""
    strategy = SessionLiquidityStrategy(data_6m)
    engine = BacktestEngine(data_6m)
    signals = []
    func = strategy.get_strategy_func()

    def intercept(idx, df_slice, eng):
        result = func(idx, df_slice, eng)
        if result is not None:
            ts = strategy.df.index[idx]
            signals.append({
                "bar_idx": idx, "bar_ts": ts,
                "direction": result["direction"],
                "stop_loss": result["stop_loss"],
                "take_profit": result["take_profit"],
            })
        return result

    trades = engine.run_strategy(intercept)
    return {"signals": signals, "trades": trades}


@pytest.fixture(scope="module")
def live_direct_results(data_6m):
    """Run live strategy in direct mode (full data, evaluate_at each bar)."""
    config = _make_config()
    live = LiveStrategy(config)
    live.reset()

    signals = []
    trades = []
    cte = -1

    for idx in range(len(data_6m)):
        if idx <= cte:
            continue
        sig = live.evaluate_at(data_6m, bar_idx=idx)
        if sig is not None:
            ts = data_6m.index[idx]
            signals.append({
                "bar_idx": idx, "bar_ts": ts,
                "direction": sig["direction"],
                "stop_loss": sig["stop_loss"],
                "take_profit": sig["take_profit"],
            })
            trade = _sim_trade(data_6m, idx, sig["direction"], sig["stop_loss"], sig["take_profit"])
            if trade is not None:
                trade["bar_idx"] = idx
                trades.append(trade)
                cte = trade["exit_idx"]

    return {"signals": signals, "trades": trades}


@pytest.fixture(scope="module")
def live_rolling_results(data_6m):
    """Run live strategy with 200-bar sliding window (real live conditions)."""
    config = _make_config()
    live = LiveStrategy(config)
    live.reset()
    window_size = 200

    signals = []
    trades = []
    cte = -1

    for bar_end in range(window_size, len(data_6m)):
        if bar_end <= cte:
            continue
        ws = max(0, bar_end - window_size)
        window = data_6m.iloc[ws : bar_end + 1]
        local_idx = bar_end - ws
        sig = live.evaluate_at(window, bar_idx=local_idx)
        if sig is not None:
            ts = data_6m.index[bar_end]
            signals.append({
                "bar_idx": bar_end, "bar_ts": ts,
                "direction": sig["direction"],
                "stop_loss": sig["stop_loss"],
                "take_profit": sig["take_profit"],
            })
            trade = _sim_trade(data_6m, bar_end, sig["direction"], sig["stop_loss"], sig["take_profit"])
            if trade is not None:
                trade["bar_idx"] = bar_end
                trades.append(trade)
                cte = trade["exit_idx"]

    return {"signals": signals, "trades": trades, "window_size": window_size}


# ======================================================================
# TEST A: Direct mode — full parity
# ======================================================================

class TestDirectModeParity:
    """evaluate_at() on full data must match backtest 100%."""

    def test_signal_count(self, backtest_results, live_direct_results):
        bt = len(backtest_results["signals"])
        live = len(live_direct_results["signals"])
        print(f"\nBacktest: {bt} signals, Live direct: {live} signals")
        assert bt == live

    def test_all_signals_identical(self, backtest_results, live_direct_results):
        bt = backtest_results["signals"]
        live = live_direct_results["signals"]
        n = min(len(bt), len(live))
        mismatches = []
        for i in range(n):
            ts_ok = bt[i]["bar_ts"] == live[i]["bar_ts"]
            dir_ok = bt[i]["direction"] == live[i]["direction"]
            sl_ok = abs(bt[i]["stop_loss"] - live[i]["stop_loss"]) < 1e-6
            tp_ok = abs(bt[i]["take_profit"] - live[i]["take_profit"]) < 1e-6
            if not (ts_ok and dir_ok and sl_ok and tp_ok):
                mismatches.append(f"#{i} ts={bt[i]['bar_ts']} bt={bt[i]['direction']} live={live[i]['direction']}")
        assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:10])

    def test_total_r_matches(self, backtest_results, live_direct_results):
        bt_r = sum(t.pnl_r for t in backtest_results["trades"])
        live_r = sum(t["pnl_r"] for t in live_direct_results["trades"])
        print(f"\nBacktest R: {bt_r:.4f}, Live direct R: {live_r:.4f}")
        assert abs(bt_r - live_r) < 0.01


# ======================================================================
# TEST B: Rolling window — validation period (last 3 months) must match
# ======================================================================

class TestRollingWindowParity:
    """After 3 months warmup, rolling window must match backtest exactly."""

    def test_validation_signal_count(self, backtest_results, live_rolling_results):
        bt_val = [s for s in backtest_results["signals"] if s["bar_ts"] >= VALIDATION_START]
        rl_val = [s for s in live_rolling_results["signals"] if s["bar_ts"] >= VALIDATION_START]
        print(f"\nValidation signals — Backtest: {len(bt_val)}, Rolling: {len(rl_val)}")
        assert len(bt_val) == len(rl_val)

    def test_validation_all_signals_identical(self, backtest_results, live_rolling_results):
        """Every signal in the validation period must be identical."""
        bt_val = {s["bar_ts"]: s for s in backtest_results["signals"] if s["bar_ts"] >= VALIDATION_START}
        rl_val = {s["bar_ts"]: s for s in live_rolling_results["signals"] if s["bar_ts"] >= VALIDATION_START}

        bt_only = set(bt_val) - set(rl_val)
        rl_only = set(rl_val) - set(bt_val)
        common = set(bt_val) & set(rl_val)

        assert not bt_only, f"Signals only in backtest: {sorted(bt_only)}"
        assert not rl_only, f"Signals only in rolling: {sorted(rl_only)}"

        mismatches = []
        for ts in sorted(common):
            b, r = bt_val[ts], rl_val[ts]
            if (b["direction"] != r["direction"]
                or abs(b["stop_loss"] - r["stop_loss"]) > 1e-6
                or abs(b["take_profit"] - r["take_profit"]) > 1e-6):
                mismatches.append(
                    f"{ts}: dir bt={b['direction']} rl={r['direction']} "
                    f"SL_diff={abs(b['stop_loss']-r['stop_loss']):.8f}"
                )

        assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:10])

    def test_validation_total_r_identical(self, backtest_results, live_rolling_results, data_6m):
        """Total R in validation period must match."""
        bt_val_trades = [
            t for t in backtest_results["trades"] if t.entry_time >= VALIDATION_START
        ]
        rl_val_trades = [
            t for t in live_rolling_results["trades"]
            if data_6m.index[t["bar_idx"]] >= VALIDATION_START
        ]

        bt_r = sum(t.pnl_r for t in bt_val_trades)
        rl_r = sum(t["pnl_r"] for t in rl_val_trades)

        print(f"\nValidation R — Backtest: {bt_r:.4f}, Rolling: {rl_r:.4f}")
        assert abs(bt_r - rl_r) < 0.01


# ======================================================================
# TEST C: Full report
# ======================================================================

class TestSimulationReport:
    def test_print_report(self, data_6m, backtest_results, live_direct_results, live_rolling_results):
        bt = BacktestEngine.compute_metrics(backtest_results["trades"])

        bt_val_sigs = [s for s in backtest_results["signals"] if s["bar_ts"] >= VALIDATION_START]
        rl_val_sigs = [s for s in live_rolling_results["signals"] if s["bar_ts"] >= VALIDATION_START]
        bt_val_set = {s["bar_ts"] for s in bt_val_sigs}
        rl_val_set = {s["bar_ts"] for s in rl_val_sigs}
        matched = bt_val_set & rl_val_set

        bt_val_trades = [t for t in backtest_results["trades"] if t.entry_time >= VALIDATION_START]
        rl_val_trades = [t for t in live_rolling_results["trades"]
                         if data_6m.index[t["bar_idx"]] >= VALIDATION_START]
        bt_val_r = sum(t.pnl_r for t in bt_val_trades)
        rl_val_r = sum(t["pnl_r"] for t in rl_val_trades)

        warmup_bars = len(data_6m[data_6m.index < VALIDATION_START])

        print("\n" + "=" * 70)
        print("  6-MONTH SIMULATION: 3mo warmup + 3mo validation")
        print("=" * 70)
        print(f"  Data: {data_6m.index[0]} → {data_6m.index[-1]} ({len(data_6m)} bars)")
        print(f"  Warmup: {warmup_bars} bars | Validation from: {VALIDATION_START}")
        print("-" * 70)
        print(f"  {'Metric':<35} {'Backtest':>15} {'Live Rolling':>15}")
        print("-" * 70)
        print(f"  {'Total signals (6mo)':<35} {len(backtest_results['signals']):>15} {len(live_rolling_results['signals']):>15}")
        print(f"  {'Validation signals (3mo)':<35} {len(bt_val_sigs):>15} {len(rl_val_sigs):>15}")
        print(f"  {'Signal match rate':<35} {'':>15} {len(matched)/len(bt_val_set)*100 if bt_val_set else 100:>14.1f}%")
        print(f"  {'Validation total R':<35} {bt_val_r:>15.4f} {rl_val_r:>15.4f}")
        print(f"  {'R difference':<35} {'':>15} {abs(bt_val_r-rl_val_r):>15.6f}")
        print("-" * 70)
        print(f"  {'Full 6mo total R':<35} {bt['total_r']:>15.4f} {'':>15}")
        print(f"  {'Full 6mo win rate':<35} {bt['win_rate']:>14.2f}% {'':>15}")
        print(f"  {'Full 6mo EV per trade':<35} {bt['expectancy_r']:>15.4f} {'':>15}")
        print(f"  {'Full 6mo profit factor':<35} {bt['profit_factor']:>15.4f} {'':>15}")
        print(f"  {'Full 6mo max drawdown':<35} {bt['max_drawdown_r']:>15.4f} {'':>15}")
        print("=" * 70)
        assert True
