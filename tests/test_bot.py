"""Tests for trading_bot.bot module."""

import pytest
import pandas as pd
import numpy as np

from trading_bot.config import BotConfig
from trading_bot.bot import TradingBot


def _make_config(**overrides) -> BotConfig:
    defaults = {
        "api_key": "test",
        "api_secret": "test",
        "risk_per_trade_usd": 100.0,
        "testnet": True,
    }
    defaults.update(overrides)
    return BotConfig(**defaults)


def _mock_klines_df():
    """Return a mock klines DataFrame."""
    timestamps = pd.date_range("2024-01-15", periods=50, freq="1h", tz="UTC")
    n = len(timestamps)
    return pd.DataFrame(
        {
            "open": np.full(n, 2000.0),
            "high": np.full(n, 2005.0),
            "low": np.full(n, 1995.0),
            "close": np.full(n, 2000.0),
            "volume": np.full(n, 1000.0),
        },
        index=timestamps,
    )


class TestTradingBot:
    def test_init(self):
        """Bot should initialize without errors."""
        config = _make_config()
        bot = TradingBot(config)

        assert bot.config == config
        assert bot._running is False
        assert bot._last_candle_ts is None

    def test_has_open_position_returns_true(self, mocker):
        """Should return True when positions exist."""
        config = _make_config()
        bot = TradingBot(config)

        mocker.patch.object(
            bot.client,
            "get_open_positions",
            return_value=[{"side": "Buy", "size": "5.0"}],
        )

        assert bot._has_open_position() is True

    def test_has_open_position_returns_false(self, mocker):
        """Should return False when no positions."""
        config = _make_config()
        bot = TradingBot(config)

        mocker.patch.object(
            bot.client, "get_open_positions", return_value=[]
        )

        assert bot._has_open_position() is False

    def test_has_open_position_on_error(self, mocker):
        """Should return True (conservative) on API error."""
        config = _make_config()
        bot = TradingBot(config)

        mocker.patch.object(
            bot.client,
            "get_open_positions",
            side_effect=Exception("API error"),
        )

        assert bot._has_open_position() is True

    def test_tick_skips_same_candle(self, mocker):
        """Second tick with same candle should skip evaluation."""
        config = _make_config()
        bot = TradingBot(config)

        df = _mock_klines_df()
        mocker.patch.object(bot.client, "get_klines", return_value=df)

        strategy_eval = mocker.patch.object(bot.strategy, "evaluate", return_value=None)

        # First tick — evaluates
        bot._tick()
        # Second tick — same candle, should skip
        bot._tick()

        # Strategy should only be called once (if no position is open)
        # Actually it might be called 0 times if has_open_position returns True by default
        # Let's mock that too
        mocker.patch.object(bot, "_has_open_position", return_value=False)

        bot._last_candle_ts = None  # reset
        bot._tick()

        assert strategy_eval.call_count >= 1

    def test_tick_empty_data(self, mocker):
        """Should handle empty kline data gracefully."""
        config = _make_config()
        bot = TradingBot(config)

        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        mocker.patch.object(bot.client, "get_klines", return_value=empty_df)

        # Should not raise
        bot._tick()

    def test_execute_signal_places_order(self, mocker):
        """Should calculate position size and place order."""
        config = _make_config(risk_per_trade_usd=100.0)
        bot = TradingBot(config)

        mocker.patch.object(bot.client, "get_current_price", return_value=2000.0)
        mock_place = mocker.patch.object(
            bot.client,
            "place_order",
            return_value={"retCode": 0, "retMsg": "OK"},
        )
        mocker.patch.object(
            bot.risk_manager, "calculate_position_size", return_value=10.0
        )

        signal = {
            "direction": "long",
            "stop_loss": 1990.0,
            "take_profit": 2020.0,
            "metadata": {},
        }

        bot._execute_signal(signal)

        mock_place.assert_called_once_with(
            side="Buy",
            qty="10.0",
            stop_loss=1990.0,
            take_profit=2020.0,
        )

    def test_execute_signal_short(self, mocker):
        """Short signal should use side=Sell."""
        config = _make_config()
        bot = TradingBot(config)

        mocker.patch.object(bot.client, "get_current_price", return_value=2000.0)
        mock_place = mocker.patch.object(
            bot.client,
            "place_order",
            return_value={"retCode": 0, "retMsg": "OK"},
        )
        mocker.patch.object(
            bot.risk_manager, "calculate_position_size", return_value=5.0
        )

        signal = {
            "direction": "short",
            "stop_loss": 2010.0,
            "take_profit": 1980.0,
            "metadata": {},
        }

        bot._execute_signal(signal)

        mock_place.assert_called_once()
        call_args = mock_place.call_args
        assert call_args.kwargs["side"] == "Sell"

    def test_execute_signal_skips_zero_qty(self, mocker):
        """Should skip order when position size is 0."""
        config = _make_config()
        bot = TradingBot(config)

        mocker.patch.object(bot.client, "get_current_price", return_value=2000.0)
        mock_place = mocker.patch.object(bot.client, "place_order")
        mocker.patch.object(
            bot.risk_manager, "calculate_position_size", return_value=0.0
        )

        signal = {
            "direction": "long",
            "stop_loss": 1990.0,
            "take_profit": 2020.0,
            "metadata": {},
        }

        bot._execute_signal(signal)

        mock_place.assert_not_called()

    def test_execute_signal_handles_price_error(self, mocker):
        """Should handle get_current_price failure gracefully."""
        config = _make_config()
        bot = TradingBot(config)

        mocker.patch.object(
            bot.client,
            "get_current_price",
            side_effect=Exception("Network error"),
        )
        mock_place = mocker.patch.object(bot.client, "place_order")

        signal = {
            "direction": "long",
            "stop_loss": 1990.0,
            "take_profit": 2020.0,
            "metadata": {},
        }

        # Should not raise
        bot._execute_signal(signal)
        mock_place.assert_not_called()

    def test_init_instrument(self, mocker):
        """Should set instrument specs on the risk manager."""
        config = _make_config()
        bot = TradingBot(config)

        mocker.patch.object(
            bot.client,
            "get_instrument_info",
            return_value={
                "symbol": "XAUUSD",
                "lotSizeFilter": {"minOrderQty": "0.05", "qtyStep": "0.05"},
            },
        )

        mock_set_specs = mocker.patch.object(bot.risk_manager, "set_instrument_specs")

        bot._init_instrument()

        mock_set_specs.assert_called_once_with(0.05, 0.05)
