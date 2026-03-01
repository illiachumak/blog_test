"""Tests for trading_bot.config module."""

import os
import pytest

from trading_bot.config import load_config, BotConfig, _parse_bool


class TestParseBool:
    def test_true_values(self):
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("  true  ") is True

    def test_false_values(self):
        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("") is False
        assert _parse_bool("random") is False


class TestLoadConfig:
    """Tests for load_config()."""

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("BYBIT_API_KEY", raising=False)
        monkeypatch.setenv("BYBIT_API_SECRET", "secret")
        monkeypatch.setenv("RISK_PER_TRADE_USD", "100")

        with pytest.raises(ValueError, match="BYBIT_API_KEY"):
            load_config()

    def test_missing_api_secret_raises(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "key")
        monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
        monkeypatch.setenv("RISK_PER_TRADE_USD", "100")

        with pytest.raises(ValueError, match="BYBIT_API_SECRET"):
            load_config()

    def test_missing_risk_raises(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "key")
        monkeypatch.setenv("BYBIT_API_SECRET", "secret")
        monkeypatch.delenv("RISK_PER_TRADE_USD", raising=False)

        with pytest.raises(ValueError, match="RISK_PER_TRADE_USD"):
            load_config()

    def test_negative_risk_raises(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "key")
        monkeypatch.setenv("BYBIT_API_SECRET", "secret")
        monkeypatch.setenv("RISK_PER_TRADE_USD", "-50")

        with pytest.raises(ValueError, match="positive"):
            load_config()

    def test_zero_risk_raises(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "key")
        monkeypatch.setenv("BYBIT_API_SECRET", "secret")
        monkeypatch.setenv("RISK_PER_TRADE_USD", "0")

        with pytest.raises(ValueError, match="positive"):
            load_config()

    def test_valid_config_with_defaults(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "test_key")
        monkeypatch.setenv("BYBIT_API_SECRET", "test_secret")
        monkeypatch.setenv("RISK_PER_TRADE_USD", "250.50")

        # Clear optional envs so defaults are used
        for var in [
            "RR_RATIO", "SL_BUFFER_PCT", "MAX_SL_POINTS", "MIN_SL_POINTS",
            "SYMBOL", "CATEGORY", "TIMEFRAME", "BYBIT_TESTNET",
            "POLL_INTERVAL_SEC", "LOOKBACK_BARS", "LOG_LEVEL",
        ]:
            monkeypatch.delenv(var, raising=False)

        cfg = load_config()

        assert cfg.api_key == "test_key"
        assert cfg.api_secret == "test_secret"
        assert cfg.risk_per_trade_usd == 250.50
        assert cfg.rr_ratio == 2.0
        assert cfg.symbol == "XAUUSD"
        assert cfg.category == "linear"
        assert cfg.timeframe == "60"
        assert cfg.testnet is False
        assert cfg.poll_interval_sec == 60
        assert cfg.log_level == "INFO"

    def test_custom_env_values(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "k")
        monkeypatch.setenv("BYBIT_API_SECRET", "s")
        monkeypatch.setenv("RISK_PER_TRADE_USD", "500")
        monkeypatch.setenv("RR_RATIO", "3.0")
        monkeypatch.setenv("BYBIT_TESTNET", "true")
        monkeypatch.setenv("SYMBOL", "BTCUSDT")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("TRADE_PDH_SWEEP", "false")

        cfg = load_config()

        assert cfg.risk_per_trade_usd == 500.0
        assert cfg.rr_ratio == 3.0
        assert cfg.testnet is True
        assert cfg.symbol == "BTCUSDT"
        assert cfg.log_level == "DEBUG"
        assert cfg.trade_pdh_sweep is False

    def test_config_is_frozen(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "k")
        monkeypatch.setenv("BYBIT_API_SECRET", "s")
        monkeypatch.setenv("RISK_PER_TRADE_USD", "100")

        cfg = load_config()

        with pytest.raises(AttributeError):
            cfg.risk_per_trade_usd = 999

    def test_multiple_missing_vars_listed(self, monkeypatch):
        monkeypatch.delenv("BYBIT_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
        monkeypatch.delenv("RISK_PER_TRADE_USD", raising=False)

        with pytest.raises(ValueError) as exc_info:
            load_config()

        msg = str(exc_info.value)
        assert "BYBIT_API_KEY" in msg
        assert "BYBIT_API_SECRET" in msg
        assert "RISK_PER_TRADE_USD" in msg
