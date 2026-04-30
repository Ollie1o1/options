#!/usr/bin/env bash
# Daily exit-rule enforcer for paper_trades.db.
# Closes any open paper trade that has hit its take-profit, stop-loss, or
# time-exit threshold per config.json's exit_rules. Safe to re-run.
#
# Install via crontab (runs at 14:07 ET, after market settles a bit):
#   7 14 * * 1-5 /Users/ollie/Desktop/options/scripts/enforce_exits.sh >> /Users/ollie/Desktop/options/logs/enforce_exits.log 2>&1

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }
echo "[$(ts)] enforce_exits.sh starting"

if [[ ! -x venv/bin/python ]]; then
  echo "[$(ts)] ERROR: venv/bin/python missing — bootstrap the venv first" >&2
  exit 1
fi

venv/bin/python -m src.options_screener --enforce-exits
echo "[$(ts)] enforce_exits.sh done"
