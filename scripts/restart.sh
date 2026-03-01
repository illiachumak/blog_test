#!/usr/bin/env bash
# Restart the trading bot (stop + start).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Restarting trading bot ==="
docker compose down
docker compose up -d
echo "=== Bot restarted ==="
