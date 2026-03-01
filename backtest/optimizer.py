"""
Optimizer for the XAUUSD session-liquidity strategy.

Performs systematic parameter search to find configurations that
maximize a composite score balancing EV, win rate, drawdown,
and trade frequency.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_generator import load_data
from .engine import BacktestEngine, compute_equity_curve
from .sessions import (
    compute_pdl_pdh,
    compute_session_levels,
    detect_liquidity_sweeps,
    detect_pdl_pdh_sweeps,
    label_sessions,
)
from .strategy import SessionLiquidityStrategy, get_default_params, get_param_grid


def composite_score(metrics: dict) -> float:
    """Combine multiple objectives into a single optimization score.

    Higher is better.  Returns 0 if expectancy is non-positive.
    """
    if metrics["expectancy_r"] <= 0:
        return 0.0
    if metrics["total_trades"] < 30:
        return 0.0

    ev = metrics["expectancy_r"]
    wr = metrics["win_rate"] / 100.0
    calmar = min(metrics["calmar_ratio"], 10.0) / 10.0
    trade_freq = min(metrics["avg_trades_per_month"], 20.0) / 20.0
    pf = min(metrics["profit_factor"], 3.0) / 3.0

    score = (
        ev * 0.35
        + wr * 0.15
        + calmar * 0.10
        + trade_freq * 0.20
        + pf * 0.20
    )

    # Penalties
    if metrics["max_drawdown_r"] > 30:
        score *= 0.5
    if metrics["total_trades"] < 50:
        score *= 0.3

    return round(score, 6)


def _run_single_config(
    df: pd.DataFrame,
    params: dict,
) -> dict:
    """Run a single backtest configuration and return metrics + score."""
    try:
        strategy = SessionLiquidityStrategy(df, params)
        engine = BacktestEngine(strategy.df)
        trades = strategy.run(engine)
        metrics = BacktestEngine.compute_metrics(trades)
        score = composite_score(metrics)

        # Add trade breakdown by signal type
        breakdown = {}
        if trades:
            for t in trades:
                sig_type = t.metadata.get("signal_type", "unknown")
                if sig_type not in breakdown:
                    breakdown[sig_type] = {"count": 0, "wins": 0}
                breakdown[sig_type]["count"] += 1
                if t.result == "win":
                    breakdown[sig_type]["wins"] += 1

        return {
            "params": params,
            "metrics": metrics,
            "score": score,
            "breakdown": breakdown,
            "trades": trades,
        }
    except Exception as e:
        return {
            "params": params,
            "metrics": BacktestEngine.compute_metrics([]),
            "score": 0.0,
            "breakdown": {},
            "trades": [],
            "error": str(e),
        }


def smart_search(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Systematic parameter optimization.

    1. Start with default params.
    2. Test one parameter at a time (coordinate descent).
    3. Do focused refinement around the best found.
    """
    base_params = get_default_params()
    grid = get_param_grid()

    results: list[dict] = []
    best_score = -1.0
    best_params = dict(base_params)

    # Phase 1: Baseline
    if verbose:
        print("=" * 60)
        print("PHASE 1: Baseline with default parameters")
        print("=" * 60)

    result = _run_single_config(df, base_params)
    results.append({"config_id": 0, **_flatten_result(result, base_params)})
    best_score = result["score"]
    if verbose:
        _print_short_result(0, result)

    config_id = 1

    # Phase 2: Coordinate descent - vary one param at a time
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Coordinate descent (one param at a time)")
        print("=" * 60)

    for param_name, values in grid.items():
        if verbose:
            print(f"\n--- Varying: {param_name} ---")

        for val in values:
            if val == best_params.get(param_name):
                continue  # Skip current best

            test_params = dict(best_params)
            test_params[param_name] = val

            result = _run_single_config(df, test_params)
            results.append({"config_id": config_id, **_flatten_result(result, test_params)})

            if verbose:
                _print_short_result(config_id, result, param_name, val)

            if result["score"] > best_score:
                best_score = result["score"]
                best_params = dict(test_params)
                if verbose:
                    print(f"    *** NEW BEST: score={best_score:.4f}")

            config_id += 1

    # Phase 3: Combo search with best individual params
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 3: Combo refinement around best params")
        print("=" * 60)

    # Test key parameter combinations
    key_combos = {
        "rr_ratio": [best_params["rr_ratio"] * 0.8, best_params["rr_ratio"], best_params["rr_ratio"] * 1.2],
        "sl_buffer_pct": [0.0005, 0.001, 0.002],
        "max_sl_points": [best_params["max_sl_points"] - 3, best_params["max_sl_points"], best_params["max_sl_points"] + 5],
    }

    combo_keys = list(key_combos.keys())
    combo_values = list(key_combos.values())

    for combo in itertools.product(*combo_values):
        test_params = dict(best_params)
        for k, v in zip(combo_keys, combo):
            test_params[k] = v

        result = _run_single_config(df, test_params)
        results.append({"config_id": config_id, **_flatten_result(result, test_params)})

        if result["score"] > best_score:
            best_score = result["score"]
            best_params = dict(test_params)
            if verbose:
                _print_short_result(config_id, result)
                print(f"    *** NEW BEST: score={best_score:.4f}")

        config_id += 1

    # Phase 4: Try disabling individual sweep types from best
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 4: Sweep type toggling")
        print("=" * 60)

    sweep_toggles = [
        "trade_asian_sweep_in_london",
        "trade_asian_sweep_in_ny",
        "trade_london_sweep_in_ny",
        "trade_pdh_sweep",
        "trade_pdl_sweep",
    ]
    for toggle in sweep_toggles:
        test_params = dict(best_params)
        test_params[toggle] = not best_params[toggle]

        result = _run_single_config(df, test_params)
        results.append({"config_id": config_id, **_flatten_result(result, test_params)})

        if verbose:
            _print_short_result(config_id, result, toggle, test_params[toggle])

        if result["score"] > best_score:
            best_score = result["score"]
            best_params = dict(test_params)
            if verbose:
                print(f"    *** NEW BEST: score={best_score:.4f}")

        config_id += 1

    # Build results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)

    return results_df


def _flatten_result(result: dict, params: dict) -> dict:
    """Flatten a result dict for DataFrame storage."""
    flat = {}
    for k, v in params.items():
        flat[f"p_{k}"] = v
    for k, v in result["metrics"].items():
        flat[f"m_{k}"] = v
    flat["score"] = result["score"]
    return flat


def _print_short_result(
    config_id: int,
    result: dict,
    param_name: str = "",
    param_val: Any = "",
) -> None:
    """Print a one-line result summary."""
    m = result["metrics"]
    param_str = f" [{param_name}={param_val}]" if param_name else ""
    print(
        f"  #{config_id:3d}{param_str}: "
        f"EV={m['expectancy_r']:+.3f}R "
        f"WR={m['win_rate']:.1f}% "
        f"PF={m['profit_factor']:.2f} "
        f"Trades={m['total_trades']} "
        f"DD={m['max_drawdown_r']:.1f}R "
        f"TotalR={m['total_r']:.1f} "
        f"Score={result['score']:.4f}"
    )


def print_report(metrics: dict, params: dict, trades: list) -> None:
    """Pretty-print the backtest results."""
    sep = "=" * 55

    print(f"\n{sep}")
    print("  XAUUSD SESSION-LIQUIDITY STRATEGY BACKTEST REPORT")
    print(sep)

    print("\nSTRATEGY PARAMETERS:")
    print(f"  R:R Ratio .............. {params['rr_ratio']}")
    print(f"  SL Buffer .............. {params['sl_buffer_pct']*100:.2f}%")
    print(f"  Max SL Points .......... {params['max_sl_points']}")
    print(f"  Min SL Points .......... {params['min_sl_points']}")
    print(f"  Min Sweep Wick ......... {params['min_sweep_wick_pct']*100:.3f}%")
    print(f"  Use Session SL ......... {params['use_session_sl']}")
    print(f"  Only First Sweep ....... {params['only_first_sweep']}")
    print(f"  Require Confluence ..... {params['require_pdl_pdh_confluence']}")
    print(f"  Friday Cutoff Hour ..... {params['no_trade_friday_after']}")
    print(f"  Asian→London ........... {params['trade_asian_sweep_in_london']}")
    print(f"  Asian→NY ............... {params['trade_asian_sweep_in_ny']}")
    print(f"  London→NY .............. {params['trade_london_sweep_in_ny']}")
    print(f"  PDH Sweep .............. {params['trade_pdh_sweep']}")
    print(f"  PDL Sweep .............. {params['trade_pdl_sweep']}")

    print(f"\nPERFORMANCE METRICS:")
    print(f"  Total Trades ........... {metrics['total_trades']}")
    print(f"  Win Rate ............... {metrics['win_rate']:.2f}%")
    print(f"  Expectancy (EV) ........ {metrics['expectancy_r']:.4f}R")
    print(f"  Profit Factor .......... {metrics['profit_factor']:.4f}")
    print(f"  Total R ................ {metrics['total_r']:.2f}R")
    print(f"  Max Drawdown ........... {metrics['max_drawdown_r']:.2f}R")
    print(f"  Calmar Ratio ........... {metrics['calmar_ratio']:.2f}")
    print(f"  Sharpe Ratio ........... {metrics['sharpe_ratio']:.4f}")
    print(f"  Avg Trades/Month ....... {metrics['avg_trades_per_month']:.2f}")
    print(f"  Max Consec. Wins ....... {metrics['max_consecutive_wins']}")
    print(f"  Max Consec. Losses ..... {metrics['max_consecutive_losses']}")
    print(f"  Avg Winner ............. {metrics['avg_winner_r']:.4f}R")
    print(f"  Avg Loser .............. {metrics['avg_loser_r']:.4f}R")

    # Trade breakdown by signal type
    if trades:
        print(f"\nTRADE BREAKDOWN BY SIGNAL TYPE:")
        breakdown: dict = {}
        for t in trades:
            sig_type = t.metadata.get("signal_type", "unknown")
            detail = ""
            if sig_type == "session_sweep":
                swept = t.metadata.get("swept_session", "?")
                current = t.metadata.get("current_session", "?")
                detail = f"{swept}→{current}"
            elif sig_type == "pdl_pdh_sweep":
                detail = t.metadata.get("level_name", "?")

            key = f"{sig_type} ({detail})" if detail else sig_type
            if key not in breakdown:
                breakdown[key] = {"count": 0, "wins": 0, "total_r": 0.0}
            breakdown[key]["count"] += 1
            if t.result == "win":
                breakdown[key]["wins"] += 1
            breakdown[key]["total_r"] += t.pnl_r

        for key, stats in sorted(breakdown.items(), key=lambda x: -x[1]["count"]):
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
            pct = stats["count"] / len(trades) * 100
            print(
                f"  {key:35s} {stats['count']:4d} trades ({pct:5.1f}%) "
                f"WR: {wr:5.1f}%  TotalR: {stats['total_r']:+.1f}"
            )

    print(f"\n{sep}\n")


def run_full_optimization(csv_path=None, last_n_years: int = 5) -> dict:
    """Main entry point: load data, optimize, report."""
    print("Loading XAUUSD data...")
    df = load_data(csv_path)
    if last_n_years:
        cutoff = df.index.max() - pd.DateOffset(years=last_n_years)
        df = df[df.index >= cutoff]
    print(f"  {len(df)} bars, {df.index.min()} to {df.index.max()}")

    print("\nStarting optimization...\n")
    t0 = time.time()
    results_df = smart_search(df, verbose=True)
    elapsed = time.time() - t0
    print(f"\nOptimization completed in {elapsed:.0f}s ({len(results_df)} configurations)")

    # Get best params
    best_row = results_df.iloc[0]
    best_params = {
        col[2:]: best_row[col]
        for col in results_df.columns
        if col.startswith("p_")
    }

    # Convert numpy types to Python types
    for k, v in best_params.items():
        if isinstance(v, (np.bool_, np.integer, np.floating)):
            best_params[k] = v.item()

    # Re-run with best params for full trade list
    print("\nRe-running with best parameters...")
    strategy = SessionLiquidityStrategy(df, best_params)
    engine = BacktestEngine(strategy.df)
    trades = strategy.run(engine)
    metrics = BacktestEngine.compute_metrics(trades)

    print_report(metrics, best_params, trades)

    # Save results CSV
    results_path = str(load_data.__module__).replace(".", "/").rsplit("/", 1)[0]
    results_df.to_csv("backtest/data/optimization_results.csv", index=False)
    print("Optimization results saved to backtest/data/optimization_results.csv")

    return {
        "best_params": best_params,
        "metrics": metrics,
        "trades": trades,
        "all_results": results_df,
    }


if __name__ == "__main__":
    run_full_optimization()
