"""Tests for trading_bot.risk_manager module."""

import pytest

from trading_bot.config import BotConfig
from trading_bot.risk_manager import RiskManager


def _make_config(**overrides) -> BotConfig:
    """Create a BotConfig with test defaults."""
    defaults = {
        "api_key": "test",
        "api_secret": "test",
        "risk_per_trade_usd": 100.0,
    }
    defaults.update(overrides)
    return BotConfig(**defaults)


class TestRiskManager:
    def test_basic_position_size(self):
        """If 1R=$100 and SL distance=10 points, qty should be 10."""
        config = _make_config(risk_per_trade_usd=100.0)
        rm = RiskManager(config)
        rm.set_instrument_specs(min_qty=0.01, qty_step=0.01)

        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=1990.0)
        assert qty == 10.0  # 100 / 10 = 10

    def test_position_size_short(self):
        """Short trade: entry < stop_loss."""
        config = _make_config(risk_per_trade_usd=200.0)
        rm = RiskManager(config)
        rm.set_instrument_specs(min_qty=0.01, qty_step=0.01)

        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=2005.0)
        assert qty == 40.0  # 200 / 5 = 40

    def test_position_size_rounding(self):
        """Position size should be rounded down to qty_step."""
        config = _make_config(risk_per_trade_usd=100.0)
        rm = RiskManager(config)
        rm.set_instrument_specs(min_qty=0.01, qty_step=0.1)

        # 100 / 7 = 14.2857... -> rounded down to 14.2
        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=1993.0)
        assert qty == pytest.approx(14.2, abs=0.01)

    def test_position_size_below_minimum(self):
        """If calculated qty is below min, should return 0."""
        config = _make_config(risk_per_trade_usd=1.0)  # Very small risk
        rm = RiskManager(config)
        rm.set_instrument_specs(min_qty=1.0, qty_step=1.0)

        # 1.0 / 10 = 0.1 which is < min_qty=1.0
        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=1990.0)
        assert qty == 0.0

    def test_zero_sl_distance(self):
        """Zero SL distance should return 0."""
        config = _make_config(risk_per_trade_usd=100.0)
        rm = RiskManager(config)

        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=2000.0)
        assert qty == 0.0

    def test_large_risk(self):
        """Large 1R should produce large position."""
        config = _make_config(risk_per_trade_usd=10000.0)
        rm = RiskManager(config)
        rm.set_instrument_specs(min_qty=0.01, qty_step=0.01)

        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=1995.0)
        assert qty == 2000.0  # 10000 / 5 = 2000

    def test_small_sl_distance(self):
        """Very tight SL should produce large position (capped by real risk)."""
        config = _make_config(risk_per_trade_usd=100.0)
        rm = RiskManager(config)
        rm.set_instrument_specs(min_qty=0.01, qty_step=0.01)

        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=1999.0)
        assert qty == 100.0  # 100 / 1 = 100

    def test_default_instrument_specs(self):
        """Defaults should work (min_qty=0.01, qty_step=0.01)."""
        config = _make_config(risk_per_trade_usd=50.0)
        rm = RiskManager(config)
        # Don't call set_instrument_specs — use defaults

        qty = rm.calculate_position_size(entry_price=2000.0, stop_loss=1990.0)
        assert qty == 5.0  # 50 / 10 = 5
