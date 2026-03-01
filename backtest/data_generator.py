"""
XAUUSD 1-hour OHLCV data loader.

Downloads real historical XAUUSD data from the ejtraderLabs GitHub repository
(mirror of broker data, 2012-2022), processes it into a clean pandas DataFrame.

Original Kaggle dataset: novandraanugrah/xauusd-gold-price-historical-data-2004-2024
GitHub mirror: https://github.com/ejtraderLabs/historical-data
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "xauusd_1h_raw.csv"
CSV_PATH = DATA_DIR / "xauusd_1h.csv"

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/ejtraderLabs/historical-data"
    "/main/XAUUSD/XAUUSDh1.csv"
)


def download_raw_data() -> Path:
    """Download the raw XAUUSD 1H CSV from GitHub."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading XAUUSD 1H data from {DOWNLOAD_URL} ...")
    subprocess.run(
        ["curl", "-sL", DOWNLOAD_URL, "-o", str(RAW_CSV)],
        check=True,
    )
    print(f"  Saved raw CSV to {RAW_CSV}")
    return RAW_CSV


def process_raw_data(raw_path: Path | None = None) -> pd.DataFrame:
    """Read the raw ejtraderLabs CSV and convert to a clean DataFrame.

    The raw file has columns: Date, open, high, low, close, tick_volume
    Prices are in the format 154759.0 which means $1547.59 (multiply by 0.01).
    """
    src = raw_path or RAW_CSV

    df = pd.read_csv(src)
    df.rename(columns={"Date": "timestamp", "tick_volume": "volume"}, inplace=True)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Convert prices: raw format is price * 100 (e.g., 154759.0 = $1547.59)
    for col in ["open", "high", "low", "close"]:
        df[col] = np.round(df[col] / 100.0, 2)

    df["volume"] = df["volume"].astype(int)

    # Remove any weekend candles (data quality)
    dow = df.index.dayofweek
    df = df[dow != 5]  # Remove Saturday

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    return df


def save_data(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write the OHLCV DataFrame to CSV."""
    out = Path(path) if path else CSV_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=True)
    return out


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load XAUUSD 1H data. Downloads and processes if not available.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume
        with a UTC DatetimeIndex named 'timestamp'.
    """
    src = Path(path) if path else CSV_PATH

    if not src.exists():
        # Try to process from raw file first
        if RAW_CSV.exists():
            print("Processing raw data...")
            df = process_raw_data()
            save_data(df, src)
            return df

        # Download raw data
        download_raw_data()
        df = process_raw_data()
        save_data(df, src)
        return df

    df = pd.read_csv(
        src,
        index_col="timestamp",
        parse_dates=["timestamp"],
    )
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
    return df


if __name__ == "__main__":
    # Download if raw not present
    if not RAW_CSV.exists():
        download_raw_data()

    print("Processing raw XAUUSD 1H data...")
    df = process_raw_data()
    out_path = save_data(df)

    print(f"  Rows:        {len(df):,}")
    print(f"  Date range:  {df.index.min()} -> {df.index.max()}")
    print(f"  Open range:  ${df['open'].min():.2f} - ${df['open'].max():.2f}")
    print(f"  Close range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  High range:  ${df['high'].max():.2f}")
    print(f"  Low range:   ${df['low'].min():.2f}")
    print(f"  Saved to:    {out_path}")

    print("\nSession volatility comparison (avg bar range $):")
    for session_name, (start_h, end_h) in [
        ("Asian (00-08)", (0, 8)),
        ("London (08-13)", (8, 13)),
        ("Overlap (13-16)", (13, 16)),
        ("NY solo (16-22)", (16, 22)),
    ]:
        mask = (df.index.hour >= start_h) & (df.index.hour < end_h)
        avg_range = (df.loc[mask, "high"] - df.loc[mask, "low"]).mean()
        print(f"  {session_name}: ${avg_range:.2f}")

    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    print("\nDone.")
