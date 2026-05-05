#!/usr/bin/env bash
# Crypto paper-trade auto-logger — runs every 4 hours from cron.
# Reads the off-switch + safeguards from config.json; dormant by default.
#
# Install via crontab (every 4 hours, on the hour):
#   0 */4 * * *  /Users/ollie/Desktop/options/scripts/auto_log_crypto.sh \
#     >> /Users/ollie/Desktop/options/logs/auto_log_crypto.log 2>&1

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

if [[ ! -x venv/bin/python ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ERROR: venv/bin/python missing — bootstrap the venv first" >&2
  exit 1
fi

exec venv/bin/python -m src.crypto.auto_logger
