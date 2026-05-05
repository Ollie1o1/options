#!/usr/bin/env bash
# Equity paper-trade auto-logger — runs M-F at 10:30, 12:30, 14:15 ET.
# Reads off-switch + stress threshold from config.json; dormant by default.
#
# Install via crontab (three weekday lines, same wrapper each time):
#   30 10 * * 1-5  /Users/ollie/Desktop/options/scripts/auto_log_equity.sh \
#     >> /Users/ollie/Desktop/options/logs/auto_log_equity.log 2>&1
#   30 12 * * 1-5  /Users/ollie/Desktop/options/scripts/auto_log_equity.sh \
#     >> /Users/ollie/Desktop/options/logs/auto_log_equity.log 2>&1
#   15 14 * * 1-5  /Users/ollie/Desktop/options/scripts/auto_log_equity.sh \
#     >> /Users/ollie/Desktop/options/logs/auto_log_equity.log 2>&1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }

echo "[$(ts)] auto_log_equity starting"

if [[ ! -x venv/bin/python ]]; then
  echo "[$(ts)] ERROR: venv/bin/python missing — bootstrap the venv first" >&2
  exit 1
fi

# 1. Off-switch
ENABLED=$(venv/bin/python - <<'PY'
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

# 4. Stress gate
STRESS=$(venv/bin/python scripts/equity_stress_check.py 2>&1 || true)
if [[ "$STRESS" != SAFE* ]]; then
  echo "[auto-log-eq] skipped: stress gate $STRESS"
  echo "[$(ts)] auto_log_equity done"
  exit 0
fi

# 5. Mode by clock window
MODE=""
if   (( HMN >= 1015 && HMN <= 1130 )); then MODE="-ds"
elif (( HMN >= 1215 && HMN <= 1330 )); then MODE="-sps"
elif (( HMN >= 1400 && HMN <= 1500 )); then MODE="-ics"
fi
if [[ -z "$MODE" ]]; then
  echo "[auto-log-eq] skipped: no mode window for clock $HM"
  echo "[$(ts)] auto_log_equity done"
  exit 0
fi

# 6. Invoke screener
echo "[auto-log-eq] mode=$MODE stress=$STRESS"
echo "[auto-log-eq] invoking: run.py $MODE --1 --no-ai"
venv/bin/python run.py "$MODE" --1 --no-ai
RC=$?

echo "[$(ts)] auto_log_equity done (rc=$RC)"
exit $RC
