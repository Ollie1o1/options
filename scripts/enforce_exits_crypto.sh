#!/usr/bin/env bash
# Hourly exit-rule enforcer for paper_trades_crypto.db.
# Closes any open crypto paper trade that has hit its take-profit, stop-loss,
# or time-exit threshold per config.json's exit_rules. Prices via Deribit.
# Safe to re-run.
#
# Install via crontab (every hour, all hours, every day — crypto is 24/7):
#   0 * * * * /Users/ollie/Desktop/options/scripts/enforce_exits_crypto.sh \
#     >> /Users/ollie/Desktop/options/logs/enforce_exits_crypto.log 2>&1

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }

if [[ ! -x venv/bin/python ]]; then
  echo "[$(ts)] ERROR: venv/bin/python missing — bootstrap the venv first" >&2
  exit 1
fi

venv/bin/python -m src.crypto.exit_enforcer
