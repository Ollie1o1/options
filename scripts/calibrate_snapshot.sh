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
#
# Note (2026-05-01): credit-spread pnl_pct CAN now exceed -1.0 in magnitude on a true
# max-loss close (e.g. $0.50 credit on $5-wide spread → -9.0). The check below now
# flags only positive overages (>1.0, impossible) and negative violations of the
# structural floor `-(width/credit - 1)` — the previous flat |pnl| > 1.0 rule
# was truncating real tail losses.
ANOMALIES=$(venv/bin/python - <<'PY'
import sqlite3, sys
conn = sqlite3.connect("paper_trades.db")
problems = []
# Credit structures: pnl_pct > 1.0 is impossible (can't make more than the credit).
# Negative bound is `-(width/credit - 1)`; flag rows that exceed it by >5%.
rows = conn.execute("""
  SELECT entry_id, ticker, strategy_name, entry_price, exit_price, pnl_pct,
         spread_width, net_credit, short_put_strike, long_put_strike,
         short_call_strike, long_call_strike, strike, long_strike
    FROM trades
   WHERE status='CLOSED'
     AND strategy_name IN ('Bull Put','Bear Call','Iron Condor')
""").fetchall()
for r in rows:
    eid, ticker, strat, entry_p, exit_p, pct = r[0], r[1], r[2], r[3], r[4], r[5]
    if pct is None:
        continue
    if pct > 1.0:
        problems.append(f"credit-spread pct>1.0 (impossible gain): id={eid} {ticker} {strat} pnl_pct={pct}")
        continue
    # derive structural floor
    width = r[6]
    credit = r[7] or entry_p
    if not width:
        # iron condor: max wing width
        try:
            sp, lp_s, sc, lc = (float(x) if x is not None else None for x in (r[8], r[9], r[10], r[11]))
            if None not in (sp, lp_s, sc, lc):
                width = max(abs(sp - lp_s), abs(lc - sc))
        except (TypeError, ValueError):
            width = None
        if not width:
            # spread: from strike vs long_strike
            try:
                k = float(r[12]); lk = float(r[13]) if r[13] not in (None, 0) else None
                if lk is not None:
                    width = abs(k - lk)
            except (TypeError, ValueError):
                width = None
    if width and credit and credit > 0 and width > credit:
        floor = -((width / credit) - 1.0) - 0.05  # 5% slack for friction
        if pct < floor:
            problems.append(
                f"credit-spread below floor: id={eid} {ticker} {strat} pnl_pct={pct:.3f} floor={floor:.3f} (width={width}, credit={credit})"
            )
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
