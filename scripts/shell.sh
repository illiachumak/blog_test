#!/usr/bin/env bash
# Open a shell inside the running bot container (for debugging).
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose exec trading-bot /bin/bash
