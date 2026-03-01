#!/usr/bin/env bash
# Stop the trading bot gracefully.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Stopping trading bot ==="
docker compose down
echo "=== Bot stopped ==="
