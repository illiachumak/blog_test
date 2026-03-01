"""Tests for trading_bot.bybit_client module (mocked API calls)."""

import pytest

from trading_bot.config import BotConfig
from trading_bot.bybit_client import BybitClient


def _make_config(**overrides) -> BotConfig:
    defaults = {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "risk_per_trade_usd": 100.0,
        "testnet": True,
    }
    defaults.update(overrides)
    return BotConfig(**defaults)


class TestBybitClientKlines:
    def test_get_klines_parses_response(self, mocker):
        """get_klines should parse Bybit kline response into a DataFrame."""
        config = _make_config()
        client = BybitClient(config)

        mock_response = {
            "retCode": 0,
            "result": {
                "list": [
                    # Most recent first (Bybit returns descending)
                    ["1705312800000", "2050.00", "2055.00", "2045.00", "2052.00", "1500.0", "0"],
                    ["1705309200000", "2040.00", "2051.00", "2038.00", "2050.00", "1200.0", "0"],
                    ["1705305600000", "2035.00", "2042.00", "2033.00", "2040.00", "1000.0", "0"],
                ]
            },
        }

        mocker.patch.object(client.session, "get_kline", return_value=mock_response)

        df = client.get_klines(limit=3)

        assert len(df) == 3
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        # Should be in chronological order (reversed from API response)
        assert df.iloc[0]["open"] == 2035.00
        assert df.iloc[-1]["open"] == 2050.00
        assert df.index.name == "timestamp"

    def test_get_current_price(self, mocker):
        """get_current_price should extract lastPrice from ticker response."""
        config = _make_config()
        client = BybitClient(config)

        mock_response = {
            "retCode": 0,
            "result": {
                "list": [{"lastPrice": "2055.50"}]
            },
        }

        mocker.patch.object(client.session, "get_tickers", return_value=mock_response)

        price = client.get_current_price()
        assert price == 2055.50


class TestBybitClientOrders:
    def test_place_order_calls_api(self, mocker):
        """place_order should call the Bybit API with correct params."""
        config = _make_config()
        client = BybitClient(config)

        mock_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"orderId": "abc123"},
        }

        mock_place = mocker.patch.object(
            client.session, "place_order", return_value=mock_response
        )

        resp = client.place_order(
            side="Buy",
            qty="10.0",
            stop_loss=1990.0,
            take_profit=2020.0,
        )

        assert resp["retCode"] == 0
        mock_place.assert_called_once_with(
            category="linear",
            symbol="XAUUSD",
            side="Buy",
            orderType="Market",
            qty="10.0",
            stopLoss="1990.0",
            takeProfit="2020.0",
            timeInForce="GTC",
        )


class TestBybitClientPositions:
    def test_get_open_positions_filters_zero(self, mocker):
        """get_open_positions should filter out zero-size positions."""
        config = _make_config()
        client = BybitClient(config)

        mock_response = {
            "retCode": 0,
            "result": {
                "list": [
                    {"side": "Buy", "size": "5.0", "symbol": "XAUUSD"},
                    {"side": "Sell", "size": "0", "symbol": "XAUUSD"},
                ]
            },
        }

        mocker.patch.object(client.session, "get_positions", return_value=mock_response)

        positions = client.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["size"] == "5.0"

    def test_get_open_positions_empty(self, mocker):
        """Should return empty list when no positions."""
        config = _make_config()
        client = BybitClient(config)

        mock_response = {
            "retCode": 0,
            "result": {"list": []},
        }

        mocker.patch.object(client.session, "get_positions", return_value=mock_response)

        positions = client.get_open_positions()
        assert positions == []


class TestBybitClientInstrument:
    def test_get_instrument_info(self, mocker):
        """get_instrument_info should return the first instrument."""
        config = _make_config()
        client = BybitClient(config)

        mock_response = {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "XAUUSD",
                        "lotSizeFilter": {
                            "minOrderQty": "0.01",
                            "qtyStep": "0.01",
                        },
                    }
                ]
            },
        }

        mocker.patch.object(
            client.session, "get_instruments_info", return_value=mock_response
        )

        info = client.get_instrument_info()
        assert info["symbol"] == "XAUUSD"
        assert info["lotSizeFilter"]["minOrderQty"] == "0.01"

    def test_get_wallet_balance(self, mocker):
        """get_wallet_balance should return balance for the coin."""
        config = _make_config()
        client = BybitClient(config)

        mock_response = {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "coin": [
                            {"coin": "USDT", "walletBalance": "5000.00"},
                            {"coin": "BTC", "walletBalance": "0.5"},
                        ]
                    }
                ]
            },
        }

        mocker.patch.object(
            client.session, "get_wallet_balance", return_value=mock_response
        )

        balance = client.get_wallet_balance("USDT")
        assert balance == 5000.00

        balance_btc = client.get_wallet_balance("BTC")
        assert balance_btc == 0.5
