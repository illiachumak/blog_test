#!/usr/bin/env bash
# Follow the trading bot logs.
set -euo pipefail

cd "$(dirname "$0")/.."

LINES="${1:-100}"
docker compose logs -f --tail="$LINES"
