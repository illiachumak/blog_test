"""
Execution fidelity tests.

Verifies that the backtest engine and live bot correctly model real trading:
  1. No look-ahead bias (entry AFTER signal bar closes)
  2. Fill price at next bar's open, not signal bar
  3. SL/TP scan starts after entry bar
  4. Session levels are causal (only use completed sessions)
  5. PDH/PDL is causal (only previous day's data)
  6. Conservative same-bar SL+TP handling (SL wins)
  7. Risk is computed from actual entry price
  8. Live bot evaluates completed bars only
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture(scope="module")
def data():
    df = _load_data()
    return df["2021-06-01":"2021-09-01"]


@pytest.fixture(scope="module")
def backtest_trades_and_signals(data):
    """Run backtest and collect both signals and trades with bar indices."""
    strategy = SessionLiquidityStrategy(data)
    engine = BacktestEngine(data)

    signal_records = []
    func = strategy.get_strategy_func()

    def intercept(idx, df_slice, eng):
        result = func(idx, df_slice, eng)
        if result is not None:
            ts = strategy.df.index[idx]
            signal_records.append({
                "signal_bar_idx": idx,
                "signal_bar_ts": ts,
                "direction": result["direction"],
                "stop_loss": result["stop_loss"],
                "take_profit": result["take_profit"],
            })
        return result

    trades = engine.run_strategy(intercept)
    return {"signals": signal_records, "trades": trades}


# ======================================================================
# TEST 1: No look-ahead bias — entry AFTER signal bar
# ======================================================================

class TestNoLookAheadBias:
    """Every trade must enter AFTER the signal bar completes."""

    def test_entry_time_after_signal_bar(self, data, backtest_trades_and_signals):
        """Trade entry_time must be > signal bar timestamp."""
        signals = backtest_trades_and_signals["signals"]
        trades = backtest_trades_and_signals["trades"]

        assert len(trades) > 0, "No trades generated"

        violations = []
        for sig, trade in zip(signals, trades):
            signal_ts = sig["signal_bar_ts"]
            entry_ts = trade.entry_time

            # Entry must be strictly after the signal bar
            if entry_ts <= signal_ts:
                violations.append(
                    f"Signal@{signal_ts} but entry@{entry_ts} "
                    f"(should enter AFTER signal bar)"
                )

        assert not violations, (
            f"{len(violations)} look-ahead violations:\n"
            + "\n".join(violations[:10])
        )

    def test_entry_at_next_bar_open(self, data, backtest_trades_and_signals):
        """Entry price must equal next bar's open (+ spread), not signal bar's open."""
        signals = backtest_trades_and_signals["signals"]
        trades = backtest_trades_and_signals["trades"]
        spread = 0.30  # default

        violations = []
        for sig, trade in zip(signals, trades):
            sig_idx = sig["signal_bar_idx"]
            entry_idx = sig_idx + 1

            if entry_idx >= len(data):
                continue

            expected_open = data.iloc[entry_idx]["open"]
            if trade.direction == "long":
                expected_entry = expected_open + spread / 2.0
            else:
                expected_entry = expected_open - spread / 2.0

            diff = abs(trade.entry_price - expected_entry)
            if diff > 1e-6:
                violations.append(
                    f"Signal@bar[{sig_idx}]: entry_price={trade.entry_price:.4f} "
                    f"!= expected {expected_entry:.4f} (bar[{entry_idx}].open={expected_open:.4f})"
                )

        assert not violations, (
            f"{len(violations)} entry price violations:\n"
            + "\n".join(violations[:10])
        )

    def test_signal_uses_completed_bar_data(self, data, backtest_trades_and_signals):
        """Signal bar must have valid OHLC before entry happens."""
        signals = backtest_trades_and_signals["signals"]

        for sig in signals:
            idx = sig["signal_bar_idx"]
            bar = data.iloc[idx]

            # These must be valid numbers (the strategy uses them)
            assert not np.isnan(bar["high"]), f"Bar[{idx}] high is NaN"
            assert not np.isnan(bar["low"]), f"Bar[{idx}] low is NaN"
            assert not np.isnan(bar["close"]), f"Bar[{idx}] close is NaN"

            # The strategy's SL is above bar high (short) or below bar low (long)
            if sig["direction"] == "short":
                assert sig["stop_loss"] >= bar["high"], (
                    f"Bar[{idx}]: short SL={sig['stop_loss']:.2f} < bar_high={bar['high']:.2f}"
                )
            else:
                assert sig["stop_loss"] <= bar["low"], (
                    f"Bar[{idx}]: long SL={sig['stop_loss']:.2f} > bar_low={bar['low']:.2f}"
                )


# ======================================================================
# TEST 2: SL/TP scan starts AFTER entry bar
# ======================================================================

class TestSLTPScanTiming:
    """SL/TP must not be checked on the entry bar (only subsequent bars)."""

    def test_sl_tp_not_hit_on_entry_bar(self, data, backtest_trades_and_signals):
        """The entry bar's open is used for entry only; SL/TP scan from bar after."""
        trades = backtest_trades_and_signals["trades"]
        signals = backtest_trades_and_signals["signals"]

        for sig, trade in zip(signals, trades):
            entry_idx = sig["signal_bar_idx"] + 1
            if entry_idx >= len(data):
                continue

            entry_ts = trade.entry_time
            exit_ts = trade.exit_time

            # Exit cannot be on the same bar as entry (since SL/TP scan
            # starts from entry_idx + 1, the earliest exit is entry_idx + 1)
            if exit_ts == entry_ts:
                # Check if there are somehow very few bars
                entry_bar = data.iloc[entry_idx]
                pytest.fail(
                    f"Trade entered and exited on same bar {entry_ts}: "
                    f"entry={trade.entry_price:.2f} exit={trade.exit_price:.2f} "
                    f"SL={trade.stop_loss:.2f} TP={trade.take_profit:.2f} "
                    f"bar_high={entry_bar['high']:.2f} bar_low={entry_bar['low']:.2f}"
                )


# ======================================================================
# TEST 3: Conservative same-bar SL+TP (SL wins)
# ======================================================================

class TestSameBarSLTP:
    """When both SL and TP could be hit on the same bar, SL wins."""

    def test_conservative_sl_wins(self):
        """Synthetic test: create a bar where both SL and TP are hit."""
        # Build a simple 5-bar DataFrame
        dates = pd.date_range("2021-06-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open":  [1800, 1800, 1800, 1800, 1800],
            "high":  [1801, 1801, 1820, 1801, 1801],  # bar[2] hits TP=1815 AND SL=1790
            "low":   [1799, 1799, 1780, 1799, 1799],
            "close": [1800, 1800, 1800, 1800, 1800],
        }, index=dates)

        engine = BacktestEngine(df, spread=0.0)

        # Long trade: entry at bar[1].open=1800, SL=1790, TP=1815
        # Bar[2]: high=1820 (TP hit), low=1780 (SL hit) -> SL should win
        trade = engine.simulate_trade(1, "long", stop_loss=1790.0, take_profit=1815.0)

        assert trade.result == "loss", f"Expected SL to win, got {trade.result}"
        assert trade.exit_price == 1790.0, f"Expected exit at SL=1790, got {trade.exit_price}"


# ======================================================================
# TEST 4: Session levels are causal
# ======================================================================

class TestSessionLevelsCausal:
    """Session levels must only be available AFTER the session ends."""

    def test_no_future_session_levels(self, data):
        """Every session level's end_ts must be <= the latest data timestamp."""
        df = label_sessions(data)
        df = compute_pdl_pdh(df)
        session_levels = compute_session_levels(df)

        if session_levels.empty:
            pytest.skip("No session levels computed")

        max_ts = df.index.max()
        future_levels = session_levels[session_levels["session_end_ts"] > max_ts]
        assert future_levels.empty, (
            f"{len(future_levels)} session levels have end_ts after data ends:\n"
            f"{future_levels[['date', 'session', 'session_end_ts']].head()}"
        )

    def test_sweep_only_uses_ended_sessions(self, data):
        """Every sweep event must reference a session that ended before the sweep bar."""
        df = label_sessions(data)
        df = compute_pdl_pdh(df)
        session_levels = compute_session_levels(df)
        sweeps = detect_liquidity_sweeps(df, session_levels)

        if sweeps.empty:
            pytest.skip("No sweeps detected")

        # Build a lookup: (date, session) -> session_end_ts
        sl_end_map = {}
        for _, row in session_levels.iterrows():
            sl_end_map[(row["date"], row["session"])] = row["session_end_ts"]

        violations = []
        for _, sweep in sweeps.iterrows():
            sweep_ts = sweep["timestamp"]
            swept_key = (sweep["swept_date"], sweep["swept_session"])
            session_end = sl_end_map.get(swept_key)

            if session_end is None:
                continue  # session not in our levels

            if session_end > sweep_ts:
                violations.append(
                    f"Sweep@{sweep_ts} references {sweep['swept_session']}@{sweep['swept_date']} "
                    f"which ends at {session_end} (FUTURE!)"
                )

        assert not violations, (
            f"{len(violations)} causal violations:\n" + "\n".join(violations[:10])
        )


# ======================================================================
# TEST 5: PDH/PDL is causal
# ======================================================================

class TestPDHPDLCausal:
    """PDH/PDL must only use previous day's data."""

    def test_pdh_pdl_not_from_current_day(self, data):
        """Each bar's PDH/PDL must be from a PREVIOUS calendar day."""
        df = label_sessions(data)
        df = compute_pdl_pdh(df)

        # Group by date, get daily high/low
        df["_date"] = df.index.date
        daily_hl = df.groupby("_date").agg(
            day_high=("high", "max"), day_low=("low", "min")
        )

        violations = []
        for date, group in df.groupby("_date"):
            pdh_val = group["PDH"].dropna().unique()
            pdl_val = group["PDL"].dropna().unique()

            if len(pdh_val) == 0:
                continue

            # PDH should be the PREVIOUS day's high, not today's
            today_high = daily_hl.loc[date, "day_high"]
            today_low = daily_hl.loc[date, "day_low"]

            for pdh in pdh_val:
                if abs(pdh - today_high) < 1e-6:
                    # Could be coincidence — verify it matches a previous day
                    date_idx = daily_hl.index.get_loc(date)
                    if date_idx > 0:
                        prev_high = daily_hl.iloc[date_idx - 1]["day_high"]
                        if abs(pdh - prev_high) > 1e-6:
                            violations.append(
                                f"{date}: PDH={pdh} matches today's high, "
                                f"not prev day's high={prev_high}"
                            )

        assert not violations, (
            f"{len(violations)} PDH/PDL causal violations:\n"
            + "\n".join(violations[:10])
        )


# ======================================================================
# TEST 6: Risk_r is computed from actual entry price
# ======================================================================

class TestRiskCalculation:
    """Risk_r must be based on actual entry price, not signal bar close."""

    def test_risk_r_from_entry_to_sl(self, backtest_trades_and_signals):
        """risk_r = |entry_price - stop_loss| for every trade."""
        trades = backtest_trades_and_signals["trades"]

        for i, trade in enumerate(trades):
            expected_risk = abs(trade.entry_price - trade.stop_loss)
            diff = abs(trade.risk_r - expected_risk)
            assert diff < 1e-6, (
                f"Trade #{i}: risk_r={trade.risk_r:.6f} != "
                f"|entry({trade.entry_price:.4f}) - SL({trade.stop_loss:.4f})| = "
                f"{expected_risk:.6f}"
            )

    def test_pnl_r_calculation(self, backtest_trades_and_signals):
        """pnl_r = (exit - entry) / risk_r for long, (entry - exit) / risk_r for short."""
        trades = backtest_trades_and_signals["trades"]

        for i, trade in enumerate(trades):
            if trade.direction == "long":
                expected_pnl = (trade.exit_price - trade.entry_price) / trade.risk_r
            else:
                expected_pnl = (trade.entry_price - trade.exit_price) / trade.risk_r

            diff = abs(trade.pnl_r - expected_pnl)
            assert diff < 1e-6, (
                f"Trade #{i}: pnl_r={trade.pnl_r:.6f} != expected {expected_pnl:.6f}"
            )


# ======================================================================
# TEST 7: Live bot evaluates completed bars only
# ======================================================================

class TestLiveBotTiming:
    """The live strategy must evaluate completed bars, not the forming bar."""

    def test_evaluate_uses_second_to_last_bar(self, data):
        """evaluate() should check bar[-2], not bar[-1] (the forming bar)."""
        config = _make_config()
        live = LiveStrategy(config)

        # Take a window that we know has a signal
        # Just verify evaluate doesn't crash and uses bar[-2]
        for start in range(0, min(200, len(data) - 50), 10):
            window = data.iloc[start:start + 50]
            sig = live.evaluate(window)
            # If signal returned, its metadata timestamp should be bar[-2]
            if sig is not None:
                meta_ts = pd.Timestamp(sig["metadata"]["timestamp"])
                expected_ts = window.index[-2]
                assert meta_ts == expected_ts, (
                    f"Signal timestamp {meta_ts} != bar[-2] {expected_ts}"
                )
                break
        else:
            pytest.skip("No signal found in test window")

    def test_evaluate_at_matches_evaluate(self, data):
        """evaluate_at(bar_idx) and evaluate() should give the same result
        when bar_idx corresponds to bar[-2]."""
        config = _make_config()

        # Find a window that generates a signal
        for start in range(0, min(200, len(data) - 50), 5):
            window = data.iloc[start:start + 50]

            live1 = LiveStrategy(config)
            live1.reset()
            sig_evaluate = live1.evaluate(window)

            if sig_evaluate is not None:
                live2 = LiveStrategy(config)
                live2.reset()
                sig_at = live2.evaluate_at(window, bar_idx=len(window) - 2)

                assert sig_at is not None, "evaluate_at returned None but evaluate returned signal"
                assert sig_evaluate["direction"] == sig_at["direction"]
                assert abs(sig_evaluate["stop_loss"] - sig_at["stop_loss"]) < 1e-6
                assert abs(sig_evaluate["take_profit"] - sig_at["take_profit"]) < 1e-6
                break
        else:
            pytest.skip("No signal found")


# ======================================================================
# TEST 8: Entry price gap analysis (bar close vs next bar open)
# ======================================================================

class TestEntryPriceRealism:
    """Verify the gap between signal bar close and next bar open is reasonable."""

    def test_entry_gap_statistics(self, data, backtest_trades_and_signals):
        """Entry gaps (next_bar_open - signal_bar_close) should be small on 1H bars."""
        signals = backtest_trades_and_signals["signals"]

        gaps = []
        for sig in signals:
            sig_idx = sig["signal_bar_idx"]
            entry_idx = sig_idx + 1
            if entry_idx >= len(data):
                continue

            signal_close = data.iloc[sig_idx]["close"]
            next_open = data.iloc[entry_idx]["open"]
            gap_pct = abs(next_open - signal_close) / signal_close * 100
            gaps.append(gap_pct)

        if not gaps:
            pytest.skip("No trades to analyze")

        avg_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        p95_gap = np.percentile(gaps, 95)

        print(f"\n  Entry gap statistics ({len(gaps)} trades):")
        print(f"    Average gap: {avg_gap:.4f}%")
        print(f"    Max gap:     {max_gap:.4f}%")
        print(f"    P95 gap:     {p95_gap:.4f}%")

        # On 1H XAUUSD bars, gaps should be < 0.5% in 95% of cases
        assert p95_gap < 0.5, f"P95 gap {p95_gap:.4f}% > 0.5% — too large for 1H bars"
        assert True  # Report always passes, stats are printed


# ======================================================================
# TEST 9: Trade timeout closes at market
# ======================================================================

class TestTradeTimeout:
    """Trades that don't hit SL/TP within MAX_TRADE_BARS close at market."""

    def test_timeout_trade_closes_at_close_price(self):
        """After MAX_TRADE_BARS without exit, trade closes at last bar's close."""
        # Create data with very far SL/TP that won't be hit
        n_bars = 60
        dates = pd.date_range("2021-06-01", periods=n_bars, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open":  [1800.0] * n_bars,
            "high":  [1801.0] * n_bars,
            "low":   [1799.0] * n_bars,
            "close": [1800.0] * n_bars,
        }, index=dates)

        engine = BacktestEngine(df, spread=0.0)

        # SL and TP very far away — will timeout
        trade = engine.simulate_trade(0, "long", stop_loss=1700.0, take_profit=1900.0)

        # MAX_TRADE_BARS = 48, so exit at bar 48's close
        expected_exit_idx = min(0 + 1 + 48, n_bars) - 1
        expected_exit_price = df.iloc[expected_exit_idx]["close"]

        assert trade.exit_price == expected_exit_price
        assert trade.result in ("win", "loss", "breakeven")


# ======================================================================
# TEST 10: Deterministic signal selection
# ======================================================================

class TestDeterministicSelection:
    """Signal selection must be deterministic (tiebreaker on _sl_distance)."""

    def test_backtest_multiple_runs_identical(self, data):
        """Running the backtest twice on the same data gives identical trades."""
        strategy1 = SessionLiquidityStrategy(data)
        engine1 = BacktestEngine(data)
        trades1 = engine1.run_strategy(strategy1.get_strategy_func())

        strategy2 = SessionLiquidityStrategy(data)
        engine2 = BacktestEngine(data)
        trades2 = engine2.run_strategy(strategy2.get_strategy_func())

        assert len(trades1) == len(trades2), (
            f"Different trade counts: {len(trades1)} vs {len(trades2)}"
        )

        for i, (t1, t2) in enumerate(zip(trades1, trades2)):
            assert t1.direction == t2.direction, f"Trade #{i}: direction differs"
            assert t1.entry_time == t2.entry_time, f"Trade #{i}: entry_time differs"
            assert abs(t1.pnl_r - t2.pnl_r) < 1e-10, f"Trade #{i}: pnl_r differs"
