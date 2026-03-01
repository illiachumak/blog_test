"""
Main trading bot loop.

Polls Bybit for new candles, evaluates the strategy, and places
orders with proper 1R risk sizing.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional

from .bybit_client import BybitClient
from .config import BotConfig
from .live_strategy import LiveStrategy
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class TradingBot:
    """Live Bybit XAUUSD trading bot."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.client = BybitClient(config)
        self.strategy = LiveStrategy(config)
        self.risk_manager = RiskManager(config)
        self._running = False
        self._last_candle_ts: Optional[str] = None

    def _setup_signal_handlers(self) -> None:
        """Graceful shutdown on SIGTERM / SIGINT."""

        def _shutdown(signum, frame):
            signame = signal.Signals(signum).name
            logger.info("Received %s — shutting down gracefully...", signame)
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

    def _init_instrument(self) -> None:
        """Fetch instrument specs and configure the risk manager."""
        try:
            info = self.client.get_instrument_info()
            lot_filter = info.get("lotSizeFilter", {})
            min_qty = float(lot_filter.get("minOrderQty", "0.01"))
            qty_step = float(lot_filter.get("qtyStep", "0.01"))
            self.risk_manager.set_instrument_specs(min_qty, qty_step)
            logger.info(
                "Instrument %s initialized: minQty=%s qtyStep=%s",
                self.config.symbol,
                min_qty,
                qty_step,
            )
        except Exception:
            logger.exception("Failed to fetch instrument info — using defaults")

    def _has_open_position(self) -> bool:
        """Check if we already have an open position."""
        try:
            positions = self.client.get_open_positions()
            return len(positions) >= self.config.max_open_positions
        except Exception:
            logger.exception("Error checking open positions")
            return True  # Be conservative — assume position exists

    def _execute_signal(self, sig: dict) -> None:
        """Place an order based on the strategy signal."""
        direction = sig["direction"]
        stop_loss = sig["stop_loss"]
        take_profit = sig["take_profit"]

        # Get current price as entry estimate
        try:
            entry_price = self.client.get_current_price()
        except Exception:
            logger.exception("Failed to get current price — skipping signal")
            return

        # Calculate position size for 1R risk
        qty = self.risk_manager.calculate_position_size(entry_price, stop_loss)
        if qty <= 0:
            logger.warning("Position size is 0 — skipping trade")
            return

        side = "Buy" if direction == "long" else "Sell"

        try:
            resp = self.client.place_order(
                side=side,
                qty=str(qty),
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            ret_code = resp.get("retCode", -1)
            if ret_code == 0:
                logger.info(
                    "ORDER PLACED: %s %s qty=%.4f SL=%.2f TP=%.2f | 1R=$%.2f",
                    side,
                    self.config.symbol,
                    qty,
                    stop_loss,
                    take_profit,
                    self.config.risk_per_trade_usd,
                )
            else:
                logger.error("Order failed: retCode=%s msg=%s", ret_code, resp.get("retMsg", ""))
        except Exception:
            logger.exception("Exception placing order")

    def run(self) -> None:
        """Main bot loop. Runs until interrupted."""
        self._setup_signal_handlers()
        self._running = True

        logger.info("=" * 60)
        logger.info("Bybit Trading Bot starting")
        logger.info("Symbol: %s", self.config.symbol)
        logger.info("Timeframe: %s", self.config.timeframe)
        logger.info("1R risk: $%.2f", self.config.risk_per_trade_usd)
        logger.info("RR ratio: %.1f", self.config.rr_ratio)
        logger.info("Testnet: %s", self.config.testnet)
        logger.info("Poll interval: %ds", self.config.poll_interval_sec)
        logger.info("=" * 60)

        self._init_instrument()

        consecutive_errors = 0
        max_consecutive_errors = 10

        while self._running:
            try:
                self._tick()
                consecutive_errors = 0
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — stopping")
                break
            except Exception:
                consecutive_errors += 1
                logger.exception(
                    "Error in main loop (consecutive: %d/%d)",
                    consecutive_errors,
                    max_consecutive_errors,
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        "Too many consecutive errors (%d) — shutting down",
                        consecutive_errors,
                    )
                    break

            if self._running:
                time.sleep(self.config.poll_interval_sec)

        logger.info("Bot stopped.")

    def _tick(self) -> None:
        """Single iteration of the bot loop."""
        # Fetch candles
        df = self.client.get_klines()

        if df.empty:
            logger.warning("No kline data received")
            return

        latest_ts = str(df.index[-1])

        # Only evaluate when a new candle appears
        if latest_ts == self._last_candle_ts:
            return

        self._last_candle_ts = latest_ts
        logger.debug("New candle: %s (total bars: %d)", latest_ts, len(df))

        # Check if we already have an open position
        if self._has_open_position():
            logger.debug("Position already open — skipping signal evaluation")
            return

        # Evaluate strategy
        sig = self.strategy.evaluate(df)

        if sig is None:
            logger.debug("No signal on this candle")
            return

        # Execute the trade
        self._execute_signal(sig)
