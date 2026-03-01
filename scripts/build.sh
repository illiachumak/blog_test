#!/usr/bin/env bash
# Build the Docker image (runs tests during build).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Building trading bot Docker image ==="
docker compose build --no-cache
echo "=== Build complete ==="
