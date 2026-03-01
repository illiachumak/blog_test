#!/usr/bin/env bash
# Start the trading bot in detached mode.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "=== Starting trading bot ==="
docker compose up -d
echo "=== Bot started. Use './scripts/logs.sh' to monitor ==="
