#!/usr/bin/env bash
# Phase 1 weekly checkpoint — computes forward-cohort IC and emits a gate decision.
# Reads phase1_start_date from config.json; writes reports to reports/.
# Never modifies paper_trades.db or config.json.
#
# Install via crontab (Sunday evenings, weekly):
#   0 18 * * 0  /Users/ollie/Desktop/options/scripts/phase1_checkpoint.sh >> /Users/ollie/Desktop/options/logs/phase1_checkpoint.log 2>&1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs reports

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }

echo "[$(ts)] phase1_checkpoint starting"

VENV="${HOME}/.venvs/options/bin/python"
if [[ ! -x "$VENV" ]]; then
  echo "[$(ts)] ERROR: $VENV missing — bootstrap the venv first" >&2
  exit 1
fi

"$VENV" -m src.phase1_checkpoint \
  --db paper_trades.db \
  --config config.json \
  --output reports

RC=$?
echo "[$(ts)] phase1_checkpoint done (rc=$RC)"
exit $RC
