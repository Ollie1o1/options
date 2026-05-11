#!/usr/bin/env bash
# Hourly exit-rule enforcer for paper_trades_crypto.db.
# Closes any open crypto paper trade that has hit its take-profit, stop-loss,
# or time-exit threshold per config.json's exit_rules. Prices via Deribit.
# Safe to re-run.
#
# Schedule lives in:
#   ~/Library/LaunchAgents/com.ollie.options.crypto-enforce-exits.plist
# launchd catches up missed runs at wake; cron does not.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }

VENV="${HOME}/.venvs/options/bin/python"
if [[ ! -x "$VENV" ]]; then
  echo "[$(ts)] ERROR: $VENV missing — bootstrap the venv first" >&2
  exit 1
fi

exec /usr/bin/caffeinate -i "$VENV" -m src.crypto.exit_enforcer
