#!/usr/bin/env bash
# Crypto paper-trade auto-logger — runs every 4 hours via launchd.
# Reads the off-switch + safeguards from config.json; dormant by default.
#
# Schedule lives in:
#   ~/Library/LaunchAgents/com.ollie.options.crypto-auto-log.plist
# launchd is used (not cron) so that runs missed while the Mac was asleep
# get caught up at wake. Cron silently drops them.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

VENV="${HOME}/.venvs/options/bin/python"
if [[ ! -x "$VENV" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ERROR: $VENV missing — bootstrap the venv first" >&2
  exit 1
fi

# caffeinate -i prevents idle sleep for the duration of this process so
# a slow Deribit fetch can't be killed by the system going to sleep.
exec /usr/bin/caffeinate -i "$VENV" -m src.crypto log
