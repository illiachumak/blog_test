"""
Risk manager for the Bybit trading bot.

Calculates position sizes based on 1R risk (dollar amount from ENV),
the stop-loss distance, and the instrument specifications.
"""

from __future__ import annotations

import logging
import math

from .config import BotConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """Calculates position size so that hitting SL costs exactly 1R."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._min_qty: float = 0.01
        self._qty_step: float = 0.01

    def set_instrument_specs(self, min_qty: float, qty_step: float) -> None:
        """Update instrument min order qty and qty step from exchange."""
        self._min_qty = min_qty
        self._qty_step = qty_step
        logger.info(
            "Instrument specs updated: min_qty=%.4f, qty_step=%.4f",
            min_qty,
            qty_step,
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """Calculate the position size in contracts/lots.

        The position size is computed so that if SL is hit, the loss
        equals exactly RISK_PER_TRADE_USD (1R).

        For XAUUSD linear perpetual on Bybit:
            1 contract = 1 troy ounce
            PnL = qty * (exit_price - entry_price) for longs
            => qty = risk_usd / sl_distance_price

        Parameters
        ----------
        entry_price : float
            Expected entry price.
        stop_loss : float
            Stop-loss price level.

        Returns
        -------
        float
            Position size rounded down to the instrument's qty step.
            Returns 0.0 if the calculated size is below minimum.
        """
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            logger.warning("SL distance is zero — cannot calculate position size")
            return 0.0

        risk_usd = self.config.risk_per_trade_usd
        raw_qty = risk_usd / sl_distance

        # Round down to qty_step
        qty = math.floor(raw_qty / self._qty_step) * self._qty_step

        if qty < self._min_qty:
            logger.warning(
                "Calculated qty %.4f is below min_qty %.4f (risk=$%.2f, SL dist=%.2f). "
                "Skipping trade.",
                qty,
                self._min_qty,
                risk_usd,
                sl_distance,
            )
            return 0.0

        logger.info(
            "Position size: %.4f (1R=$%.2f, SL_dist=%.2f, raw=%.4f)",
            qty,
            risk_usd,
            sl_distance,
            raw_qty,
        )
        return qty
