"""
Simulation test: verify live bot signal generation matches backtest.

This test loads historical XAUUSD data and runs BOTH the backtest strategy
and the live strategy bar-by-bar on the same data. It then compares:
  1. Signal-level parity (same bar → same direction, SL, TP)
  2. Trade outcome parity (simulated P&L matches)
  3. Aggregate metrics (total R, win rate, expectancy)

The live strategy is tested in TWO modes:
  A. "Direct" mode — evaluate_at() with full pre-enriched data
  B. "Rolling window" mode — sliding window of N bars (proves live conditions work)
"""

from __future__ import annotations

import sys
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.engine import BacktestEngine, Trade
from backtest.strategy import SessionLiquidityStrategy, get_default_params
from backtest.sessions import (
    label_sessions,
    compute_pdl_pdh,
    compute_session_levels,
    detect_liquidity_sweeps,
    detect_pdl_pdh_sweeps,
)
from trading_bot.config import BotConfig
from trading_bot.live_strategy import LiveStrategy


# ======================================================================
# Helpers
# ======================================================================

def _make_config(**overrides) -> BotConfig:
    """Build a BotConfig with default strategy params matching backtest defaults."""
    bt_params = get_default_params()
    defaults = {
        "api_key": "test",
        "api_secret": "test",
        "risk_per_trade_usd": 100.0,
        "testnet": True,
        "rr_ratio": bt_params["rr_ratio"],
        "sl_buffer_pct": bt_params["sl_buffer_pct"],
        "max_sl_points": bt_params["max_sl_points"],
        "min_sl_points": bt_params["min_sl_points"],
        "min_sweep_wick_pct": bt_params["min_sweep_wick_pct"],
        "trade_asian_sweep_in_london": bt_params["trade_asian_sweep_in_london"],
        "trade_asian_sweep_in_ny": bt_params["trade_asian_sweep_in_ny"],
        "trade_london_sweep_in_ny": bt_params["trade_london_sweep_in_ny"],
        "trade_pdh_sweep": bt_params["trade_pdh_sweep"],
        "trade_pdl_sweep": bt_params["trade_pdl_sweep"],
        "use_session_sl": bt_params["use_session_sl"],
        "only_first_sweep": bt_params["only_first_sweep"],
        "require_pdl_pdh_confluence": bt_params["require_pdl_pdh_confluence"],
        "confluence_distance_pct": bt_params["confluence_distance_pct"],
        "no_trade_friday_after": bt_params["no_trade_friday_after"],
    }
    defaults.update(overrides)
    return BotConfig(**defaults)


def _load_data() -> pd.DataFrame:
    """Load XAUUSD historical data."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "backtest", "data", "xauusd_1h.csv",
    )
    if not os.path.exists(csv_path):
        pytest.skip(f"Historical data file not found at {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _simulate_trade_on_bars(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    stop_loss: float,
    take_profit: float,
    spread: float = 0.30,
    max_bars: int = 48,
) -> Dict[str, Any]:
    """Simulate a single trade bar-by-bar (same logic as BacktestEngine)."""
    raw_entry = df.iloc[entry_idx]["open"]
    if direction == "long":
        entry_price = raw_entry + spread / 2.0
    else:
        entry_price = raw_entry - spread / 2.0

    risk_r = abs(entry_price - stop_loss)
    if risk_r == 0:
        risk_r = 1e-9

    end_idx = min(entry_idx + 1 + max_bars, len(df))

    for i in range(entry_idx + 1, end_idx):
        bar_high = df.iloc[i]["high"]
        bar_low = df.iloc[i]["low"]

        if direction == "long":
            sl_hit = bar_low <= stop_loss
            tp_hit = bar_high >= take_profit
        else:
            sl_hit = bar_high >= stop_loss
            tp_hit = bar_low <= take_profit

        if sl_hit and tp_hit:
            exit_price = stop_loss
            break
        elif sl_hit:
            exit_price = stop_loss
            break
        elif tp_hit:
            exit_price = take_profit
            break
    else:
        last_idx = end_idx - 1
        exit_price = df.iloc[last_idx]["close"]
        i = last_idx

    if direction == "long":
        raw_pnl = exit_price - entry_price
    else:
        raw_pnl = entry_price - exit_price

    pnl_r = raw_pnl / risk_r
    return {
        "pnl_r": pnl_r,
        "exit_idx": i,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


# ======================================================================
# Fixtures — compute once, share across all tests
# ======================================================================

@pytest.fixture(scope="module")
def data_3m():
    """Load 3 months of XAUUSD data for testing."""
    df = _load_data()
    start = "2021-06-01"
    end = "2021-09-01"
    df_slice = df[start:end]
    if len(df_slice) < 200:
        pytest.skip(f"Not enough data ({len(df_slice)} bars)")
    return df_slice


@pytest.fixture(scope="module")
def backtest_results(data_3m):
    """Run the backtest strategy and collect signals + trades."""
    df = data_3m
    strategy = SessionLiquidityStrategy(df)
    engine = BacktestEngine(df)

    signals: List[Dict[str, Any]] = []
    original_func = strategy.get_strategy_func()

    def intercepting_func(current_idx, df_slice, eng):
        result = original_func(current_idx, df_slice, eng)
        if result is not None:
            ts = strategy.df.index[current_idx]
            signals.append({
                "bar_idx": current_idx,
                "bar_ts": ts,
                "direction": result["direction"],
                "stop_loss": result["stop_loss"],
                "take_profit": result["take_profit"],
                "signal_type": result.get("metadata", {}).get("signal_type", ""),
            })
        return result

    trades = engine.run_strategy(intercepting_func)
    metrics = BacktestEngine.compute_metrics(trades)
    return {"signals": signals, "trades": trades, "metrics": metrics}


@pytest.fixture(scope="module")
def live_direct_results(data_3m):
    """Run the live strategy in direct (full-data) mode and collect signals + trades.

    This pre-enriches data ONCE and then uses evaluate_at() on each bar,
    avoiding the O(n²) cost of re-enriching for every bar.
    """
    df = data_3m
    config = _make_config()
    live = LiveStrategy(config)
    live.reset()

    signals = []
    trades_data = []
    current_trade_exit_idx = -1

    for idx in range(len(df)):
        if idx <= current_trade_exit_idx:
            continue

        sig = live.evaluate_at(df, bar_idx=idx)
        if sig is not None:
            ts = df.index[idx]
            sig_record = {
                "bar_idx": idx,
                "bar_ts": ts,
                "direction": sig["direction"],
                "stop_loss": sig["stop_loss"],
                "take_profit": sig["take_profit"],
                "signal_type": sig.get("metadata", {}).get("signal_type", ""),
            }
            signals.append(sig_record)

            trade = _simulate_trade_on_bars(
                df, idx, sig["direction"], sig["stop_loss"], sig["take_profit"]
            )
            trade["bar_idx"] = idx
            trades_data.append(trade)
            current_trade_exit_idx = trade["exit_idx"]

    return {"signals": signals, "trades": trades_data}


@pytest.fixture(scope="module")
def live_rolling_results(data_3m):
    """Run the live strategy with a sliding window (simulating real live conditions)."""
    df = data_3m
    window_size = 200
    config = _make_config()
    live = LiveStrategy(config)
    live.reset()

    signals = []
    trades_data = []
    current_trade_exit_idx = -1

    for bar_end in range(window_size, len(df)):
        if bar_end <= current_trade_exit_idx:
            continue

        window_start = max(0, bar_end - window_size)
        df_window = df.iloc[window_start : bar_end + 1]
        local_idx = bar_end - window_start

        sig = live.evaluate_at(df_window, bar_idx=local_idx)

        if sig is not None:
            ts = df.index[bar_end]
            sig_record = {
                "bar_idx": bar_end,
                "bar_ts": ts,
                "direction": sig["direction"],
                "stop_loss": sig["stop_loss"],
                "take_profit": sig["take_profit"],
            }
            signals.append(sig_record)

            trade = _simulate_trade_on_bars(
                df, bar_end, sig["direction"], sig["stop_loss"], sig["take_profit"]
            )
            trade["bar_idx"] = bar_end
            trades_data.append(trade)
            current_trade_exit_idx = trade["exit_idx"]

    return {"signals": signals, "trades": trades_data, "window_size": window_size}


# ======================================================================
# TEST A: Direct mode — evaluate_at() on full data
# ======================================================================

class TestDirectModeSignalParity:
    """Test that LiveStrategy.evaluate_at() on full data matches backtest signals."""

    def test_signal_count_matches(self, backtest_results, live_direct_results):
        """Live strategy should detect the same number of signals."""
        bt_n = len(backtest_results["signals"])
        live_n = len(live_direct_results["signals"])
        print(f"\nBacktest signals: {bt_n}")
        print(f"Live (direct) signals: {live_n}")
        assert live_n == bt_n

    def test_signal_directions_match(self, backtest_results, live_direct_results):
        """Each signal should have the same direction."""
        bt = backtest_results["signals"]
        live = live_direct_results["signals"]
        n = min(len(bt), len(live))
        mismatches = []
        for i in range(n):
            if bt[i]["direction"] != live[i]["direction"]:
                mismatches.append(
                    f"#{i}: bt={bt[i]['direction']}@{bt[i]['bar_ts']} "
                    f"vs live={live[i]['direction']}@{live[i]['bar_ts']}"
                )
        assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:10])

    def test_sl_tp_values_match(self, backtest_results, live_direct_results):
        """SL and TP values should match exactly."""
        bt = backtest_results["signals"]
        live = live_direct_results["signals"]
        n = min(len(bt), len(live))
        sl_diffs = []
        tp_diffs = []
        for i in range(n):
            sl_diff = abs(bt[i]["stop_loss"] - live[i]["stop_loss"])
            tp_diff = abs(bt[i]["take_profit"] - live[i]["take_profit"])
            if sl_diff > 1e-6:
                sl_diffs.append((i, sl_diff))
            if tp_diff > 1e-6:
                tp_diffs.append((i, tp_diff))
        assert not sl_diffs, f"SL mismatches: {sl_diffs[:10]}"
        assert not tp_diffs, f"TP mismatches: {tp_diffs[:10]}"

    def test_total_r_matches(self, backtest_results, live_direct_results):
        """Total R from simulated trades should match backtest."""
        bt_total_r = sum(t.pnl_r for t in backtest_results["trades"])
        live_total_r = sum(t["pnl_r"] for t in live_direct_results["trades"])
        print(f"\nBacktest total R: {bt_total_r:.4f}")
        print(f"Live (direct) total R: {live_total_r:.4f}")
        assert abs(bt_total_r - live_total_r) < 0.01

    def test_bar_timestamps_align(self, backtest_results, live_direct_results):
        """Signal timestamps should align between backtest and live."""
        bt = backtest_results["signals"]
        live = live_direct_results["signals"]
        n = min(len(bt), len(live))
        mismatches = []
        for i in range(n):
            if bt[i]["bar_ts"] != live[i]["bar_ts"]:
                mismatches.append(f"#{i}: bt={bt[i]['bar_ts']} vs live={live[i]['bar_ts']}")
        assert not mismatches, f"{len(mismatches)} timestamp mismatches:\n" + "\n".join(mismatches[:10])


# ======================================================================
# TEST B: Rolling window mode
# ======================================================================

class TestRollingWindowSimulation:
    """Test that a sliding-window simulation produces comparable results."""

    def test_signal_match_rate_above_95pct(self, backtest_results, live_rolling_results):
        """Rolling window should match >=95% of backtest signals (in valid range)."""
        bt = backtest_results["signals"]
        window_size = live_rolling_results["window_size"]

        valid_bt = {s["bar_ts"] for s in bt if s["bar_idx"] >= window_size}
        live_set = {s["bar_ts"] for s in live_rolling_results["signals"]}

        matched = valid_bt & live_set
        rate = len(matched) / len(valid_bt) if valid_bt else 1.0

        print(f"\nBacktest signals (valid range): {len(valid_bt)}")
        print(f"Live (rolling) signals: {len(live_set)}")
        print(f"Matched: {len(matched)}")
        print(f"Match rate: {rate:.2%}")

        assert rate >= 0.95, f"Match rate {rate:.2%} < 95%"

    def test_total_r_within_10pct(self, backtest_results, live_rolling_results, data_3m):
        """Rolling window total R should be within 10% of backtest (valid range)."""
        bt_trades = backtest_results["trades"]
        window_size = live_rolling_results["window_size"]
        df = data_3m

        bt_valid = [
            t for t in bt_trades
            if df.index.get_indexer([t.entry_time], method="nearest")[0] >= window_size
        ]
        bt_r = sum(t.pnl_r for t in bt_valid)
        live_r = sum(t["pnl_r"] for t in live_rolling_results["trades"])

        print(f"\nBacktest total R (valid): {bt_r:.4f} ({len(bt_valid)} trades)")
        print(f"Live (rolling) total R: {live_r:.4f} ({len(live_rolling_results['trades'])} trades)")

        if abs(bt_r) > 1.0:
            pct = abs(bt_r - live_r) / abs(bt_r)
            print(f"Difference: {pct:.2%}")
            assert pct < 0.10, f"Divergence {pct:.2%} > 10%"
        else:
            assert abs(bt_r - live_r) < 5.0


# ======================================================================
# TEST C: Full comparative report
# ======================================================================

class TestSimulationReport:
    """Run a full simulation and print comparative metrics."""

    def test_print_report(self, data_3m, backtest_results, live_direct_results, live_rolling_results):
        """Print a side-by-side comparison of backtest vs live simulation."""
        bt = backtest_results["metrics"]

        live_r = [t["pnl_r"] for t in live_direct_results["trades"]]
        live_total_r = sum(live_r)
        live_count = len(live_r)
        live_wins = sum(1 for r in live_r if r > 0)
        live_losses = sum(1 for r in live_r if r < 0)
        live_wr = (live_wins / live_count * 100) if live_count else 0.0
        live_ev = np.mean(live_r) if live_r else 0.0

        roll_r = [t["pnl_r"] for t in live_rolling_results["trades"]]
        roll_total_r = sum(roll_r)
        roll_count = len(roll_r)
        roll_wins = sum(1 for r in roll_r if r > 0)
        roll_wr = (roll_wins / roll_count * 100) if roll_count else 0.0
        roll_ev = np.mean(roll_r) if roll_r else 0.0

        df = data_3m
        print("\n" + "=" * 75)
        print("       SIMULATION REPORT: Backtest vs Live Bot")
        print("=" * 75)
        print(f"  Data range: {df.index[0]} → {df.index[-1]}")
        print(f"  Total bars: {len(df)}")
        print("-" * 75)
        print(f"  {'Metric':<30} {'Backtest':>13} {'Live Direct':>13} {'Live Rolling':>13}")
        print("-" * 75)
        print(f"  {'Total trades':<30} {bt['total_trades']:>13} {live_count:>13} {roll_count:>13}")
        print(f"  {'Wins':<30} {bt['wins']:>13} {live_wins:>13} {roll_wins:>13}")
        print(f"  {'Losses':<30} {bt['losses']:>13} {live_losses:>13} {roll_count - roll_wins:>13}")
        print(f"  {'Win rate %':<30} {bt['win_rate']:>13.2f} {live_wr:>13.2f} {roll_wr:>13.2f}")
        print(f"  {'Total R':<30} {bt['total_r']:>13.4f} {live_total_r:>13.4f} {roll_total_r:>13.4f}")
        print(f"  {'Expectancy (EV) R':<30} {bt['expectancy_r']:>13.4f} {live_ev:>13.4f} {roll_ev:>13.4f}")
        print(f"  {'Profit factor':<30} {bt['profit_factor']:>13.4f} {'—':>13} {'—':>13}")
        print(f"  {'Max drawdown R':<30} {bt['max_drawdown_r']:>13.4f} {'—':>13} {'—':>13}")
        print(f"  {'Sharpe ratio':<30} {bt['sharpe_ratio']:>13.4f} {'—':>13} {'—':>13}")
        print("=" * 75)

        # Signal parity summary
        bt_sigs = {s["bar_ts"] for s in backtest_results["signals"]}
        live_sigs = {s["bar_ts"] for s in live_direct_results["signals"]}
        print(f"\n  Direct mode signal parity: {len(bt_sigs & live_sigs)}/{len(bt_sigs)} "
              f"({len(bt_sigs & live_sigs)/len(bt_sigs)*100:.1f}%)")

        roll_sigs = {s["bar_ts"] for s in live_rolling_results["signals"]}
        ws = live_rolling_results["window_size"]
        valid_bt = {s["bar_ts"] for s in backtest_results["signals"] if s["bar_idx"] >= ws}
        matched_roll = valid_bt & roll_sigs
        if valid_bt:
            print(f"  Rolling window signal parity: {len(matched_roll)}/{len(valid_bt)} "
                  f"({len(matched_roll)/len(valid_bt)*100:.1f}%)")

        print()
        assert True
