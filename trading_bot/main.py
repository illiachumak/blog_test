"""
Entry point for the Bybit trading bot.

Usage:
    python -m trading_bot.main
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from .config import load_config
from .bot import TradingBot


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pybit").setLevel(logging.WARNING)


def main() -> None:
    # Load .env file
    load_dotenv()

    try:
        config = load_config()
    except ValueError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config.log_level)

    bot = TradingBot(config)
    bot.run()


if __name__ == "__main__":
    main()
