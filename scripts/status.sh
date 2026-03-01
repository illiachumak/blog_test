#!/usr/bin/env bash
# Show the status of the trading bot container.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Container status ==="
docker compose ps

echo ""
echo "=== Last 10 log lines ==="
docker compose logs --tail=10 2>/dev/null || echo "(no logs available)"
