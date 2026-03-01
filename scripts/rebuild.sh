#!/usr/bin/env bash
# Full rebuild: stop, build fresh image (with tests), start.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "=== Stopping current instance ==="
docker compose down 2>/dev/null || true

echo "=== Rebuilding image (no cache) ==="
docker compose build --no-cache

echo "=== Starting bot ==="
docker compose up -d

echo "=== Rebuild complete. Showing logs ==="
docker compose logs -f --tail=20
