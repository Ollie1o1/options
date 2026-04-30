#!/usr/bin/env bash
# Weekly calibration snapshot: runs --calibrate, appends the component-IC table
# to logs/calibration_history.tsv, and a human-readable copy to
# logs/calibration_<DATE>.txt. Read-only — never writes to config.json.
#
# Install via crontab (runs Sunday 18:13 ET — well after the week's last close):
#   13 18 * * 0 /Users/ollie/Desktop/options/scripts/calibrate_snapshot.sh >> /Users/ollie/Desktop/options/logs/calibrate.log 2>&1

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }
DATE_TAG=$(date +%Y-%m-%d)
echo "[$(ts)] calibrate_snapshot.sh starting"

if [[ ! -x venv/bin/python ]]; then
  echo "[$(ts)] ERROR: venv/bin/python missing — bootstrap the venv first" >&2
  exit 1
fi

REPORT="logs/calibration_${DATE_TAG}.txt"

# Pre-flight data-integrity check — fail loud if any closed row violates the bounds.
# Caught a real bug once: QQQ Bear Call closed with pnl_pct=+3.58 and exit_price=-1.22,
# which silently inverted the IC sign on skew_align. Repeat that and IC drift becomes
# noise. Now sanitize_close_values clamps these at write time, so this check is a
# safety net for any bypass paths (manual SQL, future code).
ANOMALIES=$(venv/bin/python - <<'PY'
import sqlite3, sys
conn = sqlite3.connect("paper_trades.db")
problems = []
# Credit structures: |pnl_pct| > 1.0 is impossible
rows = conn.execute("""
  SELECT entry_id, ticker, strategy_name, entry_price, exit_price, pnl_pct
    FROM trades
   WHERE status='CLOSED'
     AND strategy_name IN ('Bull Put','Bear Call','Iron Condor')
     AND (pnl_pct > 1.0 OR pnl_pct < -1.0)
""").fetchall()
for r in rows:
    problems.append(f"credit-spread out of bounds: id={r[0]} {r[1]} {r[2]} pnl_pct={r[5]}")
# Negative exit prices
rows = conn.execute("""
  SELECT entry_id, ticker, strategy_name, exit_price
    FROM trades WHERE status='CLOSED' AND exit_price < 0
""").fetchall()
for r in rows:
    problems.append(f"negative exit_price: id={r[0]} {r[1]} {r[2]} exit={r[3]}")
# Closed but pnl_usd missing
n_missing = conn.execute(
    "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_usd IS NULL"
).fetchone()[0]
if n_missing:
    problems.append(f"{n_missing} closed trades missing pnl_usd")
print("\n".join(problems))
PY
)
if [[ -n "$ANOMALIES" ]]; then
  echo "[$(ts)] DATA-INTEGRITY WARNINGS:" >&2
  echo "$ANOMALIES" >&2
  echo "$ANOMALIES" > "logs/calibration_${DATE_TAG}.warnings"
fi

venv/bin/python -m src.backtester --calibrate > "$REPORT" 2>&1

# Extract n_trades and per-component IC into a tab-separated history row so the
# IC drift can be plotted over time without re-parsing free-text reports.
HIST="logs/calibration_history.tsv"
if [[ ! -f "$HIST" ]]; then
  echo -e "date\tn_trades\tcomponent\tic" > "$HIST"
fi

N_TRADES=$(grep -E "^  Closed paper trades:" "$REPORT" | awk '{print $4}' | head -n1)
N_TRADES=${N_TRADES:-0}

# Lines like:    "    vrp              IC = +0.267  ↑"  --> component, ic
awk -v d="$DATE_TAG" -v n="$N_TRADES" '
  /^    [a-z_]+ +IC = / {
    component=$1
    for (i=1; i<=NF; i++) if ($i == "=") { ic=$(i+1); break }
    print d "\t" n "\t" component "\t" ic
  }
' "$REPORT" >> "$HIST"

echo "[$(ts)] calibrate_snapshot.sh done — report: $REPORT  history: $HIST"
