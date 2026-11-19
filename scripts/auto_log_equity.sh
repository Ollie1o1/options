#!/usr/bin/env bash
# Equity side jobs — launchd `com.ollie.options.auto-log-equity`, M-F 10:30,
# 12:30, 14:15 local. (The label is historical; reloading a LaunchAgent can
# re-trigger the macOS background-item approval, so it was kept.)
#
# WHAT IT NO LONGER DOES (2026-11-18): run its own `run.py` scan. The entry
# windows (-ds, -sps, -ics) belong to `src.maintenance`, which launchd runs ten
# minutes earlier and which logs each window once per day. This script's scans
# duplicated them, and overlapping scans can log the same contract twice at a
# slightly different price (see maintenance.run_catchup). Its 14:15 slot still
# ran `-ss`, the Short Put window #136 dropped. On 2026-11-06 that `-ss` scan
# hit a full disk and hung for 12 days, which also stopped the two jobs below.
#
# WHAT IT DOES: push trade alerts for any scan since the last run, and keep the
# earnings calendar fresh for the earnings gates. Each runs under an
# awake-time timeout and the Mac is held awake until the script exits.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }

echo "[$(ts)] auto_log_equity starting"

VENV="${HOME}/.venvs/options/bin/python"
if [[ ! -x "$VENV" ]]; then
  echo "[$(ts)] ERROR: $VENV missing — bootstrap the venv first" >&2
  exit 1
fi

# 1. Off-switch
ENABLED=$("$VENV" - <<'PY'
import json, sys
try:
    with open("config.json") as f:
        c = json.load(f)
except Exception as e:
    print(f"ERROR {type(e).__name__}: {e}")
    sys.exit(0)
print("true" if (c.get("equity") or {}).get("auto_log_enabled") else "false")
PY
)
if [[ "$ENABLED" == ERROR* ]]; then
  echo "[auto-log-eq] ERROR reading config: $ENABLED" >&2
  exit 1
fi
if [[ "$ENABLED" != "true" ]]; then
  echo "[auto-log-eq] skipped: off-switch (auto_log_enabled=false)"
  echo "[$(ts)] auto_log_equity done"
  exit 0
fi

# 2. Weekday gate (1=Mon..7=Sun)
WD=$(date +%u)
if (( WD > 5 )); then
  echo "[auto-log-eq] skipped: weekend (weekday=$WD)"
  echo "[$(ts)] auto_log_equity done"
  exit 0
fi

# 3. RTH gate (09:45–15:30 local)
HM=$(date +%H%M)
HMN=$((10#$HM))   # force base-10 (avoid octal on leading zero)
if (( HMN < 945 || HMN > 1530 )); then
  echo "[auto-log-eq] skipped: outside RTH (clock=$HM local)"
  echo "[$(ts)] auto_log_equity done"
  exit 0
fi

# 4. Hold the Mac awake while this script runs; `-w $$` ends with it.
if [[ -x /usr/bin/caffeinate ]]; then
  /usr/bin/caffeinate -i -w $$ >/dev/null 2>&1 &
fi

# Run "$@" with a ceiling on AWAKE time (Python's monotonic clock stops while
# macOS sleeps). Exit 124 on timeout, like coreutils `timeout`.
with_timeout() {
  local secs="$1"; shift
  "$VENV" -c 'import subprocess, sys
try:
    sys.exit(subprocess.call(sys.argv[2:], timeout=float(sys.argv[1])))
except subprocess.TimeoutExpired:
    print("[auto-log-eq] TIMEOUT after %ss: %s" % (sys.argv[1], " ".join(sys.argv[2:])), flush=True)
    sys.exit(124)' "$secs" "$@"
}

set +e
RC=0
# 5. Push trade alerts (Discord/Telegram) for new candidates from ANY scan since
#    the last run. Never fails the job; no-op until a channel is set in .env.
with_timeout 600 "$VENV" -m src.trade_alerts || echo "[auto-log-eq] trade alerts failed (ignored)"

# 6. Keep the earnings calendar fresh for the earnings gates. Only symbols
#    cached >= 7 days ago are re-queried, so most runs fetch nothing; the
#    cache had silently gone stale from 2026-08-20 to 11-09 without this.
with_timeout 1800 "$VENV" -m src.dolt_earnings --refresh --max-age-days 7 > logs/earnings_refresh.log 2>&1 \
  || echo "[auto-log-eq] earnings refresh failed (ignored)"
set -e

echo "[$(ts)] auto_log_equity done (rc=$RC)"
exit $RC
