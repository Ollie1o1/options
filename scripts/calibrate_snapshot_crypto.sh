#!/usr/bin/env bash
# Weekly calibration snapshot for paper_trades_crypto.db.
# Computes per-component IC across closed crypto trades, appends a tab-
# separated history row to logs/calibration_history_crypto.tsv, and
# writes a human-readable report at logs/calibration_crypto_<DATE>.txt.
# Read-only — never writes to config.json (no --apply passed).
#
# Install via crontab (Sundays 18:30 ET — after equity's 18:13 calibrate):
#   30 18 * * 0  /Users/ollie/Desktop/options/scripts/calibrate_snapshot_crypto.sh \
#     >> /Users/ollie/Desktop/options/logs/calibrate_crypto.log 2>&1

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }
DATE_TAG=$(date +%Y-%m-%d)
echo "[$(ts)] calibrate_snapshot_crypto.sh starting"

VENV="${HOME}/.venvs/options/bin/python"
if [[ ! -x "$VENV" ]]; then
  echo "[$(ts)] ERROR: $VENV missing — bootstrap the venv first" >&2
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
ANOMALIES=$("$VENV" - <<'PY'
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
     AND (strategy_name LIKE '%Bear Call%' OR strategy_name LIKE '%Bull Put%' OR strategy_name LIKE '%Iron Condor%')
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

# Long premium / calendars: capped loss at -1.0 (full debit)
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

# Reuse the same calibration entrypoint equity uses, but pointed at the
# crypto DB. This is what generates the per-component IC table the awk
# parser below expects. NEVER pass --apply: crypto IC must not write the
# shared config.json weights.
"$VENV" -m src.backtester --calibrate --db paper_trades_crypto.db > "$REPORT" 2>&1

# Append per-component IC rows to the long-form history TSV.
HIST="logs/calibration_history_crypto.tsv"
if [[ ! -f "$HIST" ]]; then
  echo -e "date\tn_trades\tcomponent\tic" > "$HIST"
fi

N_TRADES=$(grep -E "^  Closed paper trades:" "$REPORT" | awk '{print $4}' | head -n1)
# Strip any "/MIN" suffix on the under-threshold form ("13/100").
N_TRADES="${N_TRADES%%/*}"
N_TRADES=${N_TRADES:-0}

# Lines like:  "    vrp              IC = +0.267  ↑"  --> component, ic
awk -v d="$DATE_TAG" -v n="$N_TRADES" '
  /^    [a-z_]+ +IC = / {
    component=$1
    for (i=1; i<=NF; i++) if ($i == "=") { ic=$(i+1); break }
    print d "\t" n "\t" component "\t" ic
  }
' "$REPORT" >> "$HIST"

# If no component rows were appended (sample below per-component minimum),
# write a stub row so the TSV records that calibration ran. Matches the
# semantics of "we tried, no signal yet."
ROWS_ADDED=$(awk -v d="$DATE_TAG" 'BEGIN{c=0} $1==d{c++} END{print c}' "$HIST")
if [[ "$ROWS_ADDED" == "0" ]]; then
  printf "%s\t%s\t%s\t%s\n" "$DATE_TAG" "$N_TRADES" "_no_component_ic_yet" "0.000" >> "$HIST"
fi

echo "[$(ts)] calibrate_snapshot_crypto.sh done — report: $REPORT  history: $HIST  rows=$ROWS_ADDED"
