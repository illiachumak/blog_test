#!/usr/bin/env bash
# Run tests locally (outside Docker).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Running tests ==="
python -m pytest tests/ -v "$@"
