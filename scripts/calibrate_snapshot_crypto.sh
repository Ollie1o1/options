#!/usr/bin/env bash
# Weekly calibration snapshot for paper_trades_crypto.db.
# Computes per-component IC across closed crypto trades, appends a tab-
# separated history row to logs/calibration_history_crypto.tsv, and
# writes a human-readable report at logs/calibration_crypto_<DATE>.txt.
# Read-only — never writes to config.json.
#
# Install via crontab (Sundays 18:30 ET — after equity's 18:13 calibrate):
#   30 18 * * 0  /Users/ollie/Desktop/options/scripts/calibrate_snapshot_crypto.sh \
#     >> /Users/ollie/Desktop/options/logs/calibrate_crypto.log 2>&1

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }
DATE_TAG=$(date +%Y-%m-%d)
echo "[$(ts)] calibrate_snapshot_crypto.sh starting"

if [[ ! -x venv/bin/python ]]; then
  echo "[$(ts)] ERROR: venv/bin/python missing — bootstrap the venv first" >&2
  exit 1
fi

if [[ ! -f paper_trades_crypto.db ]]; then
  echo "[$(ts)] No crypto ledger yet (paper_trades_crypto.db missing). Skipping."
  exit 0
fi

REPORT="logs/calibration_crypto_${DATE_TAG}.txt"

# Pre-flight integrity check tuned to the crypto schema.
# Crypto-specific quirks:
#   - Calendars repurpose `spread_width` to hold days-between-expirations,
#     so the equity-side floor check `pnl_pct < -(width/credit-1)` is
#     nonsensical for them. Skip rows where strategy_name LIKE 'Calendar%'.
#   - Iron condors use the same convention as equity.
ANOMALIES=$(venv/bin/python - <<'PY'
import sqlite3, sys
conn = sqlite3.connect("paper_trades_crypto.db")
problems = []

# Credit structures (skip calendars — repurposed spread_width)
rows = conn.execute("""
  SELECT entry_id, ticker, strategy_name, entry_price, exit_price, pnl_pct,
         spread_width, net_credit, short_put_strike, long_put_strike,
         short_call_strike, long_call_strike, strike, long_strike
    FROM trades
   WHERE status='CLOSED'
     AND strategy_name IS NOT NULL
     AND strategy_name NOT LIKE 'Calendar%'
     AND strategy_name LIKE '%Bear Call%' OR strategy_name LIKE '%Bull Put%' OR strategy_name LIKE '%Iron Condor%'
""").fetchall()
for r in rows:
    eid, ticker, strat, entry_p, exit_p, pct = r[0], r[1], r[2], r[3], r[4], r[5]
    if pct is None:
        continue
    if pct > 1.0:
        problems.append(f"crypto credit-spread pct>1.0 (impossible): id={eid} {ticker} {strat} pnl_pct={pct}")
        continue
    width = r[6]
    credit = r[7] or entry_p
    if not width:
        try:
            sp, lp_s, sc, lc = (float(x) if x is not None else None for x in (r[8], r[9], r[10], r[11]))
            if None not in (sp, lp_s, sc, lc):
                width = max(abs(sp - lp_s), abs(lc - sc))
        except (TypeError, ValueError):
            width = None
        if not width:
            try:
                k = float(r[12]); lk = float(r[13]) if r[13] not in (None, 0) else None
                if lk is not None:
                    width = abs(k - lk)
            except (TypeError, ValueError):
                width = None
    if width and credit and credit > 0 and width > credit:
        floor = -((width / credit) - 1.0) - 0.05
        if pct < floor:
            problems.append(
                f"crypto credit-spread below floor: id={eid} {ticker} {strat} "
                f"pnl_pct={pct:.3f} floor={floor:.3f} (width={width}, credit={credit})"
            )

# Long premium / calendars: pnl_pct should be in [-1.0, +∞) for calendars
# (capped loss = full debit), [-1.0, +∞) for long single legs.
rows = conn.execute("""
  SELECT entry_id, ticker, strategy_name, pnl_pct
    FROM trades
   WHERE status='CLOSED'
     AND strategy_name IS NOT NULL
     AND (strategy_name LIKE 'Long%' OR strategy_name LIKE 'Calendar%')
     AND pnl_pct < -1.05
""").fetchall()
for r in rows:
    problems.append(f"crypto long/calendar pct<-1 (over-loss): id={r[0]} {r[1]} {r[2]} pnl_pct={r[3]}")

# Negative exit prices
rows = conn.execute("""
  SELECT entry_id, ticker, strategy_name, exit_price
    FROM trades WHERE status='CLOSED' AND exit_price < 0
""").fetchall()
for r in rows:
    problems.append(f"crypto negative exit_price: id={r[0]} {r[1]} {r[2]} exit={r[3]}")

# Closed but pnl_usd missing
n_missing = conn.execute(
    "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_usd IS NULL"
).fetchone()[0]
if n_missing:
    problems.append(f"{n_missing} crypto closed trades missing pnl_usd")

print("\n".join(problems))
PY
)
if [[ -n "$ANOMALIES" ]]; then
  echo "[$(ts)] CRYPTO DATA-INTEGRITY WARNINGS:" >&2
  echo "$ANOMALIES" >&2
  echo "$ANOMALIES" > "logs/calibration_crypto_${DATE_TAG}.warnings"
fi

# Run the IC analysis via PaperManager.compute_ic() — same path as equity.
venv/bin/python - <<PY > "$REPORT" 2>&1
from src.paper_manager import PaperManager
import json

pm = PaperManager(db_path="paper_trades_crypto.db")
ic = pm.compute_ic()

print("=" * 70)
print(f"CRYPTO CALIBRATION SNAPSHOT  —  ${DATE_TAG:-}")
print("=" * 70)

if not isinstance(ic, dict):
    print("compute_ic() returned non-dict:", ic)
    raise SystemExit(0)

n = int(ic.get("n_closed", 0))
print(f"\nClosed trades analysed: {n}")
print(f"Apply gate: needs >=100 closed for IC analysis to be meaningful")
print()

if n == 0:
    print("No closed trades yet. Logged-but-open positions don't contribute.")
    raise SystemExit(0)

# Top-level overall IC if present
for k in ("ic", "ic_p", "spearman", "spearman_p"):
    if k in ic:
        print(f"  {k:<14}  {ic[k]}")

components = ic.get("components") or {}
if components:
    print("\nPer-component IC (sorted by |IC|):")
    sorted_c = sorted(components.items(), key=lambda x: -abs(float(x[1] or 0)))
    for comp, val in sorted_c:
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        marker = "*" if abs(v) >= 0.10 else " "
        print(f"  {marker} {comp:<22}  IC = {v:+.3f}")

# Per-strategy breakdown
strategies = pm.get_strategy_breakdown() if hasattr(pm, "get_strategy_breakdown") else None
if strategies:
    print("\nPer-strategy breakdown:")
    for s in strategies:
        n_s = s.get("n", 0)
        win = s.get("win_rate", 0) * 100 if isinstance(s.get("win_rate"), (int, float)) else 0
        pf = s.get("pf", 0) or s.get("profit_factor", 0)
        print(f"  {str(s.get('strategy_name','?')):<16}  n={n_s:3}  win={win:.0f}%  PF={pf:.2f}x")
PY

# Append a TSV row per component for plotting drift over time.
HIST="logs/calibration_history_crypto.tsv"
if [[ ! -f "$HIST" ]]; then
  echo -e "date\tn_trades\tcomponent\tic" > "$HIST"
fi

N_TRADES=$(grep -E "^Closed trades analysed:" "$REPORT" | awk '{print $4}' | head -n1)
N_TRADES=${N_TRADES:-0}

# Lines like:    "  * vrp_score              IC = +0.267"  --> component, ic
awk -v d="$DATE_TAG" -v n="$N_TRADES" '
  /^[ ]+[\*]?[ ]+[a-z_]+_score[ ]+IC = / {
    for (i=1; i<=NF; i++) if ($i ~ /_score$/) { component=$i; break }
    for (i=1; i<=NF; i++) if ($i == "=") { ic=$(i+1); break }
    if (component != "" && ic != "") print d "\t" n "\t" component "\t" ic
  }
' "$REPORT" >> "$HIST"

echo "[$(ts)] calibrate_snapshot_crypto.sh done — report: $REPORT  history: $HIST"
