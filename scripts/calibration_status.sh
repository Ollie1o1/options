#!/usr/bin/env bash
# One-page calibration dashboard. Run anytime: scripts/calibration_status.sh
# No side effects — pure read-only summary of paper_trades.db, config flags, and
# the latest snapshot in logs/calibration_history.tsv.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DB="paper_trades.db"
HIST="logs/calibration_history.tsv"

if [[ ! -f "$DB" ]]; then
  echo "ERROR: $DB not found"; exit 1
fi

bar() { printf '%.0s─' {1..70}; echo; }

bar
echo "  CALIBRATION STATUS                    $(date '+%Y-%m-%d %H:%M %Z')"
bar
echo

# Summary counts
N_CLOSED=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE status='CLOSED'")
N_OPEN=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE status='OPEN'")
TOTAL_PNL=$(sqlite3 "$DB" "SELECT printf('%+.2f', COALESCE(SUM(pnl_usd),0)) FROM trades WHERE status='CLOSED'")
echo "  Closed trades:  $N_CLOSED        Open positions: $N_OPEN        Realized P&L: \$$TOTAL_PNL"
echo

# Per-strategy progress against threshold
echo "  Per-strategy progress (threshold = 30 closed for actionable IC):"
sqlite3 "$DB" <<SQL
.mode column
.headers off
SELECT
  printf('    %-14s %3d / 30  %s',
         strategy_name,
         COUNT(*),
         CASE
           WHEN COUNT(*) >= 30 THEN '[ready]'
           ELSE printf('(need %d more)', 30 - COUNT(*))
         END)
FROM trades
WHERE status='CLOSED'
GROUP BY strategy_name
ORDER BY COUNT(*) DESC;
SQL
echo

# Apply-gate decision tree
N_SPREAD=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND strategy_name IN ('Bull Put','Bear Call')")
N_IC=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND strategy_name='Iron Condor'")
echo "  Apply-gate checklist (must all pass before --calibrate --apply):"
[[ $N_CLOSED -ge 200 ]] && echo "    [✓] total closed >= 200" || echo "    [ ] total closed >= 200    (have $N_CLOSED, need $((200-N_CLOSED)))"
[[ $N_SPREAD -ge 30 ]] && echo "    [✓] spreads closed >= 30" || echo "    [ ] spreads closed >= 30    (have $N_SPREAD, need $((30-N_SPREAD)))"
[[ $N_IC -ge 30 ]] && echo "    [✓] ICs closed >= 30" || echo "    [ ] ICs closed >= 30        (have $N_IC, need $((30-N_IC)) — gated by 49 DTE)"
echo

# Latest IC snapshot from history file. Multiple snapshots per day are common
# (each calibrate_snapshot.sh run appends a fresh block), so we keep only the
# last-seen value per component before sorting.
if [[ -f "$HIST" ]]; then
  LAST_DATE=$(awk -F'\t' 'NR>1 {d=$1} END {print d}' "$HIST")
  if [[ -n "$LAST_DATE" ]]; then
    echo "  Top-5 signal-bearing components (latest snapshot $LAST_DATE):"
    awk -F'\t' -v d="$LAST_DATE" 'NR>1 && $1==d {last[$3]=$4}
        END {for (c in last) printf "%s\t%s\n", c, last[c]}' "$HIST" |
      sort -t$'\t' -k2 -gr | head -5 |
      awk -F'\t' '{ic=$2+0; printf "    %-22s IC = %+.3f\n", $1, ic}'
  fi
fi
echo

# Data integrity check
ANOMALIES=$(ls logs/calibration_*.warnings 2>/dev/null | head -1 || true)
N_OOB=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND strategy_name IN ('Bull Put','Bear Call','Iron Condor') AND (pnl_pct > 1.0 OR pnl_pct < -1.0)")
N_NEG=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND exit_price < 0")
N_NULL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_usd IS NULL")
echo "  Data integrity:"
echo "    Out-of-bounds spread PnL: $N_OOB"
echo "    Negative exit prices:     $N_NEG"
echo "    NULL pnl_usd:             $N_NULL"
[[ -n "$ANOMALIES" ]] && echo "    Recent warnings file:     $ANOMALIES" || echo "    Recent warnings file:     none"
echo

# Auto-log flag visibility
SKIP_LP=$(grep -E '"auto_log_skip_long_puts"' config.json 2>/dev/null | grep -oE 'true|false')
echo "  Active filters: auto_log_skip_long_puts = ${SKIP_LP:-(unset)}"
bar
echo "  Next steps:        see LOGGING_PLAN.md"
echo "  Preview weights:   python3 -m src.backtester --calibrate"
echo "  Force re-snapshot: scripts/calibrate_snapshot.sh"
bar
