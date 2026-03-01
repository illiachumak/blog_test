"""
Bybit API client wrapper for the trading bot.

Handles kline fetching, order placement, position management,
and account balance queries via the pybit HTTP API.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from pybit.unified_trading import HTTP

from .config import BotConfig

logger = logging.getLogger(__name__)


class BybitClient:
    """Thin wrapper around pybit for the operations we need."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.session = HTTP(
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet,
        )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_klines(self, limit: int | None = None) -> pd.DataFrame:
        """Fetch recent klines (candlesticks) as a DataFrame.

        Returns a DataFrame with columns: open, high, low, close, volume
        and a UTC DatetimeIndex named 'timestamp'.
        """
        limit = limit or self.config.lookback_bars

        resp = self.session.get_kline(
            category=self.config.category,
            symbol=self.config.symbol,
            interval=self.config.timeframe,
            limit=limit,
        )

        rows = resp["result"]["list"]

        # Bybit returns: [startTime, open, high, low, close, volume, turnover]
        # Most recent candle first — we reverse to chronological order.
        records = []
        for r in reversed(rows):
            records.append(
                {
                    "timestamp": pd.Timestamp(int(r[0]), unit="ms", tz="UTC"),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
            )

        df = pd.DataFrame(records)
        df = df.set_index("timestamp")
        return df

    def get_current_price(self) -> float:
        """Get the last traded price for the symbol."""
        resp = self.session.get_tickers(
            category=self.config.category,
            symbol=self.config.symbol,
        )
        return float(resp["result"]["list"][0]["lastPrice"])

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        side: str,
        qty: str,
        stop_loss: float,
        take_profit: float,
        order_type: str = "Market",
    ) -> dict:
        """Place a market order with SL and TP.

        Parameters
        ----------
        side : str
            "Buy" or "Sell".
        qty : str
            Position size as a string (Bybit requirement).
        stop_loss : float
            Stop loss price.
        take_profit : float
            Take profit price.
        order_type : str
            Order type — default "Market".

        Returns
        -------
        dict
            Bybit API response.
        """
        logger.info(
            "Placing %s %s order: qty=%s SL=%.2f TP=%.2f",
            order_type,
            side,
            qty,
            stop_loss,
            take_profit,
        )

        resp = self.session.place_order(
            category=self.config.category,
            symbol=self.config.symbol,
            side=side,
            orderType=order_type,
            qty=qty,
            stopLoss=str(round(stop_loss, 2)),
            takeProfit=str(round(take_profit, 2)),
            timeInForce="GTC",
        )

        logger.info("Order response: %s", resp)
        return resp

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict]:
        """Return list of open positions for the configured symbol."""
        resp = self.session.get_positions(
            category=self.config.category,
            symbol=self.config.symbol,
        )
        positions = resp["result"]["list"]
        # Filter out positions with zero size
        return [p for p in positions if float(p.get("size", "0")) > 0]

    def close_all_positions(self) -> None:
        """Close all open positions for the symbol."""
        positions = self.get_open_positions()
        for pos in positions:
            side = "Sell" if pos["side"] == "Buy" else "Buy"
            size = pos["size"]
            logger.info("Closing position: side=%s size=%s", side, size)
            self.session.place_order(
                category=self.config.category,
                symbol=self.config.symbol,
                side=side,
                orderType="Market",
                qty=size,
                reduceOnly=True,
                timeInForce="GTC",
            )

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_wallet_balance(self, coin: str = "USDT") -> Optional[float]:
        """Get available wallet balance for a coin."""
        resp = self.session.get_wallet_balance(
            accountType="UNIFIED",
        )
        for account in resp["result"]["list"]:
            for c in account.get("coin", []):
                if c["coin"] == coin:
                    return float(c["walletBalance"])
        return None

    # ------------------------------------------------------------------
    # Instrument info
    # ------------------------------------------------------------------

    def get_instrument_info(self) -> dict:
        """Get instrument specs (min qty, qty step, tick size, etc.)."""
        resp = self.session.get_instruments_info(
            category=self.config.category,
            symbol=self.config.symbol,
        )
        return resp["result"]["list"][0]
