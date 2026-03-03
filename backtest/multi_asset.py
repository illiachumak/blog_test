"""
Multi-asset data loader and backtest runner.

Downloads and processes forex/commodity data from ejtraderLabs,
adapts the session-liquidity strategy to each instrument's price scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

# Instrument configuration
# divisor: raw price → real price conversion
# spread: typical spread in real price units
# pip: 1 pip in real price units
# sl_scale: multiplier to convert XAUUSD SL points to this instrument's scale
INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    "XAUUSD": {
        "divisor": 100,
        "spread": 0.30,
        "pip": 0.01,
        "decimals": 2,
    },
    "EURUSD": {
        "divisor": 100_000,
        "spread": 0.00010,
        "pip": 0.00010,
        "decimals": 5,
    },
    "GBPUSD": {
        "divisor": 100_000,
        "spread": 0.00015,
        "pip": 0.00010,
        "decimals": 5,
    },
    "AUDUSD": {
        "divisor": 100_000,
        "spread": 0.00012,
        "pip": 0.00010,
        "decimals": 5,
    },
    "USDCAD": {
        "divisor": 100_000,
        "spread": 0.00015,
        "pip": 0.00010,
        "decimals": 5,
    },
    "USDCHF": {
        "divisor": 100_000,
        "spread": 0.00015,
        "pip": 0.00010,
        "decimals": 5,
    },
    "EURGBP": {
        "divisor": 100_000,
        "spread": 0.00012,
        "pip": 0.00010,
        "decimals": 5,
    },
    "EURCHF": {
        "divisor": 100_000,
        "spread": 0.00015,
        "pip": 0.00010,
        "decimals": 5,
    },
    "USDJPY": {
        "divisor": 1_000,
        "spread": 0.010,
        "pip": 0.010,
        "decimals": 3,
    },
    "EURJPY": {
        "divisor": 1_000,
        "spread": 0.015,
        "pip": 0.010,
        "decimals": 3,
    },
    "GBPJPY": {
        "divisor": 1_000,
        "spread": 0.020,
        "pip": 0.010,
        "decimals": 3,
    },
    "AUDJPY": {
        "divisor": 1_000,
        "spread": 0.015,
        "pip": 0.010,
        "decimals": 3,
    },
}


def load_instrument(symbol: str) -> pd.DataFrame:
    """Load and process 1H data for any supported instrument."""
    symbol = symbol.upper()
    if symbol not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument: {symbol}. Available: {list(INSTRUMENTS.keys())}")

    config = INSTRUMENTS[symbol]

    # Check for processed file first
    processed_path = DATA_DIR / f"{symbol.lower()}_1h.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path, index_col="timestamp", parse_dates=["timestamp"])
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    # Load from raw
    raw_path = DATA_DIR / f"{symbol.lower()}_1h_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df.rename(columns={"Date": "timestamp", "tick_volume": "volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Convert prices
    divisor = config["divisor"]
    decimals = config["decimals"]
    for col in ["open", "high", "low", "close"]:
        df[col] = np.round(df[col] / divisor, decimals)

    df["volume"] = df["volume"].astype(int)

    # Remove weekends
    df = df[df.index.dayofweek != 5]
    df = df[~df.index.duplicated(keep="first")]

    # Save processed
    df.to_csv(processed_path)

    return df


def get_adaptive_params(symbol: str, base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Scale strategy parameters to match instrument's price scale.

    The base strategy was designed for XAUUSD (~$1800).
    We scale absolute price parameters (SL points, etc.) to match
    other instruments using the ratio of typical prices.
    """
    if base_params is None:
        base_params = {}

    config = INSTRUMENTS.get(symbol.upper(), INSTRUMENTS["XAUUSD"])

    # Reference: XAUUSD typical price ~1800, min_sl=3 → 0.167%
    # We use percentage-based scaling
    xauusd_ref_price = 1800.0
    xauusd_min_sl = base_params.get("min_sl_points", 3.0)
    xauusd_max_sl = base_params.get("max_sl_points", 15.0)

    # Estimate typical price for each instrument
    typical_prices = {
        "XAUUSD": 1800.0,
        "EURUSD": 1.15,
        "GBPUSD": 1.35,
        "AUDUSD": 0.75,
        "USDCAD": 1.30,
        "USDCHF": 0.95,
        "EURGBP": 0.87,
        "EURCHF": 1.08,
        "USDJPY": 110.0,
        "EURJPY": 130.0,
        "GBPJPY": 150.0,
        "AUDJPY": 82.0,
    }

    typical_price = typical_prices.get(symbol.upper(), 1.0)
    scale = typical_price / xauusd_ref_price

    params = dict(base_params)
    params["min_sl_points"] = round(xauusd_min_sl * scale, 6)
    params["max_sl_points"] = round(xauusd_max_sl * scale, 6)

    return params, config["spread"]


def run_multi_asset_backtest(
    symbols: list,
    strategy_class,
    base_params: Dict[str, Any],
    years: list,
) -> pd.DataFrame:
    """Run backtest across multiple assets and years."""
    from .engine import BacktestEngine

    results = []

    for symbol in symbols:
        try:
            df = load_instrument(symbol)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {symbol}: SKIP ({e})")
            continue

        adapted_params, spread = get_adaptive_params(symbol, base_params)

        for year in years:
            df_year = df[f"{year}-01-01":f"{year}-12-31"]
            if len(df_year) < 100:
                continue

            try:
                strategy = strategy_class(df_year, adapted_params)
                engine = BacktestEngine(df_year, spread=spread)
                trades = engine.run_strategy(strategy.get_strategy_func())
                metrics = BacktestEngine.compute_metrics(trades)

                results.append({
                    "symbol": symbol,
                    "year": year,
                    **metrics,
                    "spread": spread,
                    "min_sl": adapted_params["min_sl_points"],
                    "max_sl": adapted_params["max_sl_points"],
                })
            except Exception as e:
                print(f"  {symbol} {year}: ERROR ({e})")

    return pd.DataFrame(results)
