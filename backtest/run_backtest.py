#!/usr/bin/env python3
"""
XAUUSD Session-Liquidity Strategy Backtest

Main entry point that:
1. Loads real XAUUSD 1H data (from ejtraderLabs GitHub, originally Kaggle)
2. Runs optimization to find best parameters
3. Produces a comprehensive report with the best configuration

Strategy Logic:
- Identifies liquidity sweeps: when price pierces beyond a session's
  high/low and then closes back (indicating a "sweep" / "stop hunt")
- Trades the reversal after sweep events with specific rules per session:
  * Asian session highs/lows swept during London or NY session
  * London session highs/lows swept during NY session
  * Previous Day Low (PDL) sweeps
- Uses R:R based risk management with defined SL/TP levels
- All logic is strictly causal (no look-ahead bias)

Data Source:
  import kagglehub
  path = kagglehub.dataset_download("novandraanugrah/xauusd-gold-price-historical-data-2004-2024")
  (Downloaded from GitHub mirror: ejtraderLabs/historical-data)
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from backtest.data_generator import load_data, download_raw_data, RAW_CSV
from backtest.engine import BacktestEngine, compute_equity_curve
from backtest.optimizer import (
    composite_score,
    print_report,
    run_full_optimization,
    smart_search,
)
from backtest.strategy import SessionLiquidityStrategy, get_default_params


def main() -> None:
    """Run the full backtest pipeline."""
    print("=" * 60)
    print("  XAUUSD SESSION-LIQUIDITY BACKTEST")
    print("  Real Data | No Look-Ahead Bias | 5-Year Period")
    print("=" * 60)

    # ---- Data --------------------------------------------------------
    if not RAW_CSV.exists():
        download_raw_data()

    df = load_data()
    # Use last 5 years of data
    cutoff = df.index.max() - pd.DateOffset(years=5)
    df = df[df.index >= cutoff]

    print(f"\nData: {len(df):,} bars")
    print(f"Period: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # ---- Quick baseline with default params --------------------------
    print("\n" + "-" * 60)
    print("Baseline (default parameters):")
    print("-" * 60)

    strategy = SessionLiquidityStrategy(df)
    engine = BacktestEngine(strategy.df)
    trades = strategy.run(engine)
    baseline_metrics = BacktestEngine.compute_metrics(trades)

    print(f"  EV = {baseline_metrics['expectancy_r']:.3f}R")
    print(f"  Win Rate = {baseline_metrics['win_rate']:.1f}%")
    print(f"  Total R = {baseline_metrics['total_r']:.1f}")
    print(f"  Trades = {baseline_metrics['total_trades']}")

    # ---- Optimization ------------------------------------------------
    print("\n" + "-" * 60)
    print("Running parameter optimization...")
    print("-" * 60)

    result = run_full_optimization(last_n_years=5)

    # ---- Equity curve stats ------------------------------------------
    best_trades = result["trades"]
    if best_trades:
        eq = compute_equity_curve(best_trades)
        print(f"\nEquity Curve:")
        print(f"  Start: 0.0R")
        print(f"  End:   {eq.iloc[-1]:.1f}R")
        print(f"  Peak:  {eq.max():.1f}R")

        # Yearly breakdown
        print(f"\nYearly P&L Breakdown:")
        for t in best_trades:
            t._year = t.exit_time.year  # type: ignore[attr-defined]

        yearly: dict = {}
        for t in best_trades:
            y = t.exit_time.year
            if y not in yearly:
                yearly[y] = {"trades": 0, "wins": 0, "total_r": 0.0}
            yearly[y]["trades"] += 1
            if t.result == "win":
                yearly[y]["wins"] += 1
            yearly[y]["total_r"] += t.pnl_r

        for year in sorted(yearly):
            stats = yearly[year]
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
            print(
                f"  {year}: {stats['trades']:4d} trades, "
                f"WR={wr:.1f}%, "
                f"R={stats['total_r']:+.1f}"
            )

    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
