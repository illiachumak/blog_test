FROM python:3.11-slim

LABEL maintainer="trading-bot"
LABEL description="Bybit XAUUSD session-liquidity trading bot"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backtest/ ./backtest/
COPY trading_bot/ ./trading_bot/
COPY tests/ ./tests/

# Run tests at build time to catch issues early
RUN python -m pytest tests/ -v --tb=short

# Healthcheck: verify the bot module can be imported
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from trading_bot.config import load_config; print('ok')" || exit 1

# Default command: run the trading bot
CMD ["python", "-m", "trading_bot"]
