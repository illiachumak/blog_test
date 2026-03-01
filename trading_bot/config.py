"""
Configuration loader for the Bybit trading bot.

All settings are read from environment variables with sensible defaults.
The critical risk parameter RISK_PER_TRADE_USD (1R in dollars) MUST be set
manually in the .env file — there is no default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BotConfig:
    """Immutable bot configuration loaded from environment variables."""

    # --- Bybit API credentials (REQUIRED) ---
    api_key: str
    api_secret: str

    # --- Risk management (REQUIRED) ---
    risk_per_trade_usd: float  # 1R in USD — must be set manually

    # --- Strategy parameters ---
    rr_ratio: float = 2.0
    sl_buffer_pct: float = 0.001
    max_sl_points: float = 15.0
    min_sl_points: float = 2.0
    min_sweep_wick_pct: float = 0.0003

    # Session sweep toggles
    trade_asian_sweep_in_london: bool = True
    trade_asian_sweep_in_ny: bool = True
    trade_london_sweep_in_ny: bool = True
    trade_pdh_sweep: bool = True
    trade_pdl_sweep: bool = True

    # Filters
    use_session_sl: bool = True
    only_first_sweep: bool = True
    require_pdl_pdh_confluence: bool = False
    confluence_distance_pct: float = 0.003

    # Time filters
    no_trade_friday_after: int = 18

    # --- Bybit connection ---
    symbol: str = "XAUUSD"
    category: str = "linear"
    timeframe: str = "60"  # 1H candles
    testnet: bool = False

    # --- Bot behaviour ---
    poll_interval_sec: int = 60  # how often to check for new candles
    lookback_bars: int = 200  # bars to fetch for session analysis
    max_open_positions: int = 1

    # --- Logging ---
    log_level: str = "INFO"


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def load_config() -> BotConfig:
    """Load configuration from environment variables.

    Raises
    ------
    ValueError
        If required variables (API_KEY, API_SECRET, RISK_PER_TRADE_USD)
        are missing.
    """

    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    risk_str = os.environ.get("RISK_PER_TRADE_USD", "")

    errors: list[str] = []
    if not api_key:
        errors.append("BYBIT_API_KEY is not set")
    if not api_secret:
        errors.append("BYBIT_API_SECRET is not set")
    if not risk_str:
        errors.append("RISK_PER_TRADE_USD is not set (1R dollar amount)")

    if errors:
        raise ValueError(
            "Missing required environment variables:\n  - " + "\n  - ".join(errors)
        )

    risk_per_trade_usd = float(risk_str)
    if risk_per_trade_usd <= 0:
        raise ValueError(f"RISK_PER_TRADE_USD must be positive, got {risk_per_trade_usd}")

    return BotConfig(
        api_key=api_key,
        api_secret=api_secret,
        risk_per_trade_usd=risk_per_trade_usd,
        rr_ratio=float(os.environ.get("RR_RATIO", "2.0")),
        sl_buffer_pct=float(os.environ.get("SL_BUFFER_PCT", "0.001")),
        max_sl_points=float(os.environ.get("MAX_SL_POINTS", "15.0")),
        min_sl_points=float(os.environ.get("MIN_SL_POINTS", "2.0")),
        min_sweep_wick_pct=float(os.environ.get("MIN_SWEEP_WICK_PCT", "0.0003")),
        trade_asian_sweep_in_london=_parse_bool(
            os.environ.get("TRADE_ASIAN_SWEEP_IN_LONDON", "true")
        ),
        trade_asian_sweep_in_ny=_parse_bool(
            os.environ.get("TRADE_ASIAN_SWEEP_IN_NY", "true")
        ),
        trade_london_sweep_in_ny=_parse_bool(
            os.environ.get("TRADE_LONDON_SWEEP_IN_NY", "true")
        ),
        trade_pdh_sweep=_parse_bool(os.environ.get("TRADE_PDH_SWEEP", "true")),
        trade_pdl_sweep=_parse_bool(os.environ.get("TRADE_PDL_SWEEP", "true")),
        use_session_sl=_parse_bool(os.environ.get("USE_SESSION_SL", "true")),
        only_first_sweep=_parse_bool(os.environ.get("ONLY_FIRST_SWEEP", "true")),
        require_pdl_pdh_confluence=_parse_bool(
            os.environ.get("REQUIRE_PDL_PDH_CONFLUENCE", "false")
        ),
        confluence_distance_pct=float(
            os.environ.get("CONFLUENCE_DISTANCE_PCT", "0.003")
        ),
        no_trade_friday_after=int(os.environ.get("NO_TRADE_FRIDAY_AFTER", "18")),
        symbol=os.environ.get("SYMBOL", "XAUUSD"),
        category=os.environ.get("CATEGORY", "linear"),
        timeframe=os.environ.get("TIMEFRAME", "60"),
        testnet=_parse_bool(os.environ.get("BYBIT_TESTNET", "false")),
        poll_interval_sec=int(os.environ.get("POLL_INTERVAL_SEC", "60")),
        lookback_bars=int(os.environ.get("LOOKBACK_BARS", "200")),
        max_open_positions=int(os.environ.get("MAX_OPEN_POSITIONS", "1")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
