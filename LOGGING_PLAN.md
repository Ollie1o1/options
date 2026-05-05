# Trade Logging & Calibration Plan
**Updated:** 2026-05-05
**Goal:** Accumulate the closed-trade ledger needed to calibrate `composite_weights`, `credit_spread_weights`, and `iron_condor_weights` so the screener actually picks better contracts. Path to "good data" = enough volume + enough variance + zero anomalies.

---

## Quick status check

```bash
scripts/calibration_status.sh
```

One-page dashboard: closed-trade counts, per-strategy progress, apply-gate checklist, top-5 IC components from the latest snapshot, and data-integrity check. No side effects — safe to run anytime.

---

## Current State (snapshot — 2026-04-30 EOD)

```
Closed trades: 127     Open positions: 36     Realized P&L: +$9,521
Calibration shrinkage: 0.68  (system trusts component IC at ~68% strength)
```

**Per-strategy closed counts** (the gating constraint for per-structure weights):

| Strategy | Closed | Per-structure threshold | Status |
|---|---|---|---|
| Long Call | 57 | 30 | ready |
| Long Put | 38 | 30 | ready BUT auto-log disabled (PF 0.69) |
| Bull Put | 16 | 30 | need ~14 more |
| Bear Call | 9 | 30 | need ~21 more |
| Short Put | 9 | 30 | not actively logging |
| Iron Condor | 0 | 30 | gated by 49-DTE time decay (first closes mid-May) |

**Confirmed signal-bearing components** (cross-validated by paper IC + synthetic walk-forward backtest, both directions agree):

| Component | Paper IC | Synthetic IC | Verdict |
|---|---|---|---|
| `vrp` | +0.27 | +0.14 | strongest, both agree |
| `term_structure` | +0.26 | +0.15 | strong, both agree |
| `vega_risk` | +0.23 | +0.14 | strong, both agree |
| `spread` | +0.19 | ~0 | strong in paper |
| `iv_edge` | +0.12 | +0.15 | medium, both agree |

Components with zero variance (always store as constant — already weighted at 0): `gamma_pin`, `max_pain`, `pcr`, `sentiment_score_norm`, `option_rvol`. `oi_change` was previously stuck but should self-heal now that the fd-limit fix lets `.oi_snapshot.json` save.

---

## Automation In Place (2026-04-30)

| Mechanism | What it does | Where |
|---|---|---|
| `auto_log_skip_long_puts: true` | Filters Long Put picks out of `--auto-log` (PF 0.69, dragging IC) | `config.json` |
| Long-Put filter announce line | Auto-log summary now prints `filtered N Long Put(s)` for visibility | `src/options_screener.py` ~ line 4170 |
| `_sanitize_close_values()` | Clamps `pnl_pct` to physically-possible bounds, forces `exit_price ≥ 0`, computes `pnl_usd` | `src/paper_manager.py` ~ line 330 |
| Calibrate-snapshot integrity check | Pre-flight scan for out-of-bounds spread PnL, negative exits, NULL `pnl_usd`. Writes `logs/calibration_<date>.warnings` if any found | `scripts/calibrate_snapshot.sh` |
| `RLIMIT_NOFILE` bump to 8192 | Prevents `[Errno 24] Too many open files` mid-scan on macOS | `src/options_screener.py main()` |
| Daily exit enforcer | Auto-closes anything past TP/stop/time-exit | `scripts/enforce_exits.sh` |
| Weekly calibration snapshot | Runs `--calibrate`, appends per-component IC to `logs/calibration_history.tsv` so drift is plottable | `scripts/calibrate_snapshot.sh` |
| `equity.auto_log_enabled` | Master off-switch for the auto-log driver (default `false`). Flip to `true` in `config.json` to start the M-F 10:30/12:30/14:15 cron loop. | `config.json` |
| `equity.stress_gate_pct_book` | Skip threshold for the stress gate (% of book the –20%/+10pp loss is allowed to be). Default `100.0`. | `config.json` |
| Equity auto-log wrapper | Cron-driven: off-switch → weekday → RTH → stress → clock→mode → `run.py [mode] --1 --no-ai` | `scripts/auto_log_equity.sh` |
| Equity stress check helper | Prints `SAFE` / `UNSAFE` based on portfolio –20%/+10pp scenario vs the configured threshold | `scripts/equity_stress_check.py` |

**Cron lines to install** (one-time, via `crontab -e`):

```cron
# Daily exit-rule enforcer (weekdays 14:07 ET)
7 14 * * 1-5 /Users/ollie/Desktop/options/scripts/enforce_exits.sh >> /Users/ollie/Desktop/options/logs/enforce_exits.log 2>&1

# Weekly calibration snapshot (Sundays 18:13 ET)
13 18 * * 0 /Users/ollie/Desktop/options/scripts/calibrate_snapshot.sh >> /Users/ollie/Desktop/options/logs/calibrate.log 2>&1

# Equity auto-log driver — M-F three runs (off-switch in config.json, default dormant)
30 10 * * 1-5 /Users/ollie/Desktop/options/scripts/auto_log_equity.sh >> /Users/ollie/Desktop/options/logs/auto_log_equity.log 2>&1
30 12 * * 1-5 /Users/ollie/Desktop/options/scripts/auto_log_equity.sh >> /Users/ollie/Desktop/options/logs/auto_log_equity.log 2>&1
15 14 * * 1-5 /Users/ollie/Desktop/options/scripts/auto_log_equity.sh >> /Users/ollie/Desktop/options/logs/auto_log_equity.log 2>&1
```

---

## Per-Session Routine (revised after 4 days of running)

**As of 2026-05-05, this rotation is automated** by
`scripts/auto_log_equity.sh` running on cron (M-F at 10:30 / 12:30 /
14:15 ET). The off-switch is `equity.auto_log_enabled` in
`config.json` (default `false` — flip to `true` to start). The stress
gate (`equity.stress_gate_pct_book`, default `100.0`) blocks runs
while the book's –20%/+10pp scenario loss exceeds 100% of book — so
when the book is over-leveraged, the cron self-suppresses until you
close some positions.

The manual schedule below is preserved as documentation of what the
cron does each weekday. You can still run any of these by hand if the
off-switch is off or you want a one-off scan.

Run scans **during market hours** (10:00–15:30 ET — avoid open/close volatility AND post-close stale-quote fills). Outside RTH every contract gets a synthetic ±5% bid/ask and the liquidity filter kills the scan.

| Day | Command | Why |
|---|---|---|
| Mon | `python3 run.py -ds --10` | Long Calls (largest sample, well-calibrated). Long Puts auto-filtered. |
| Tue | `python3 run.py -sps --10` | Bull Puts + Bear Calls — these are the gating samples for spread calibration. |
| Wed | `python3 run.py -ics --5` | ICs (small batch — concentration concern with 11+ already open). |
| Thu | `python3 run.py -ds --10` | More Long Calls, rotate ticker base. |
| Fri | rest, OR `-sps --5` if stress test allows | **Skip `-ss` until concentration drops** — naked shorts compound the -$24k stress-test loss. |

Why no `--sell-scoring` right now: portfolio max-loss in the -20% / -10% IV scenario is currently -$24k, ~158% of book. Adding Short Puts on top is reckless until the existing IC stack matures and bleeds into profit.

Auto-log filters at work (don't be surprised when fewer trades log):
- Long Puts skipped silently (config flag)
- Per-symbol dedup: max one position per ticker per scan
- Long Put filter is announced in the summary line (`filtered N Long Put(s)`)

---

## "Don't Apply Weights Yet" — Decision Tree

```
Closed trades < 200 ?       → keep logging, don't apply
Per-structure samples
  spread < 30 ?              → keep logging spreads, don't apply
  iron_condor < 30 ?         → wait for time decay
Shrinkage < 0.80 ?           → IC values are noisy, deltas would overshoot
ANY anomaly warning in logs/ → fix that first; recompute calibration
ALL OF: n≥200, per-structure ≥30, shrinkage ≥0.80, no warnings
                            → safe to run `--calibrate --apply`
```

To preview without applying any time:
```bash
python3 -m src.backtester --calibrate                # singles
python3 -c "from src.backtester import recommend_weights_for_structure as r; print(r('paper_trades.db','config.json','spread'))"
python3 -c "from src.backtester import recommend_weights_for_structure as r; print(r('paper_trades.db','config.json','iron_condor'))"
```

---

## Updated 4-Week Schedule

### Week 1 (Apr 27 – May 1) — Cold start ✅ in progress
- **Targets**: ~70 singles / ~25 spreads / ~15 ICs **opened**
- **Actual end-of-Apr-30**: 95 singles / 16 spreads (closed) + 9 spreads (open) / 18 ICs (open, 0 closed)
- ✅ on track for opens; closes lag because the strategy mix shifted to longer-DTE structures

### Week 2 (May 4 – May 8) — Build closure mass
- Run `enforce_exits.sh` daily (cron handles it)
- First May 8 expirations close → **~10 spreads + ~5 long calls expected to close**
- **Target cumulative closed**: 145 singles / 30 spreads / still 0 ICs
- **First spread per-structure IC check possible** end of week if spread closed count ≥ 30

### Week 3 (May 11 – May 15) — Spread calibration window opens
- May 15 is the heaviest expiry — many May 8 / May 15 spreads close
- **Target cumulative closed**: 160 singles / 50 spreads / 1–3 ICs
- Run `recommend_weights_for_structure("spread")` and inspect — **don't apply yet**
- Compare to prior week's snapshot in `logs/calibration_history.tsv` to see IC drift

### Week 4 (May 18 – May 22) — First apply window
- Most May 22 expirations close
- **Target cumulative closed**: 200+ singles / 70+ spreads / 5–10 ICs
- If decision tree above passes, apply weights to `composite_weights` and `credit_spread_weights`
- ICs still gated — wait for May 29 / Jun 5 / Jun 18 expiries

### Week 5+ (May 25 onwards) — IC calibration
- Jun 18 expiries dominate the open IC book — they'll mature 6–8 weeks from open
- **Target**: 30+ closed ICs by mid-June; apply `iron_condor_weights` then

---

## What "Good Data" Looks Like (sanity checks)

Run this any time:
```bash
sqlite3 paper_trades.db "
SELECT
  strategy_name,
  COUNT(*) AS n,
  ROUND(AVG(pnl_pct),3) AS avg_pnl,
  ROUND(MIN(pnl_pct),2) AS min_p,
  ROUND(MAX(pnl_pct),2) AS max_p,
  COUNT(DISTINCT ROUND(quality_score,2)) AS score_diversity
FROM trades
WHERE status='CLOSED'
GROUP BY strategy_name
ORDER BY n DESC"
```

**Pass criteria** for each strategy with n ≥ 30:
- `avg_pnl` between −0.10 and +0.20 (not all wins, not all losses)
- `min_p` ≥ −1.0 (no out-of-bounds — sanitizer enforces this)
- `max_p` ≤ +5.0 for long premium, ≤ +1.0 for credit spreads (no anomalies)
- `score_diversity` ≥ 10 (≥10 distinct quality_score buckets — without variance, IC is flat)

If any check fails, look in `logs/calibration_<date>.warnings` and the raw report.

---

## Files Updated 2026-04-30

| Path | Change |
|---|---|
| `src/paper_manager.py` | Added `_sanitize_close_values()`, both close paths now write clean `pnl_usd` |
| `src/options_screener.py` | `RLIMIT_NOFILE` bump in `main()`; Long Put filter in single-leg auto-log |
| `config.json` | Added `auto_log_skip_long_puts: true` |
| `scripts/enforce_exits.sh` | New — daily exit enforcer wrapper |
| `scripts/calibrate_snapshot.sh` | New — weekly calibration snapshot wrapper with integrity check |
| `logs/calibration_history.tsv` | New — per-component IC time series |
| `paper_trades.db` | Cleaned: 1 anomaly fixed (entry 162), 115 NULL `pnl_usd` backfilled |

Backups: `paper_trades.db.bak.20260429-135202`, `paper_trades.db.bak.20260430-131404`, `paper_trades.db.bak.20260430-161255`.

---

## Key Files (unchanged — for reference)

- `src/paper_manager.py` — multi-leg log + exit (schema v10), now with sanitizer
- `src/backtester.py` — `run_paper_trade_ic_for_structure`, `recommend_weights_for_structure`, `--calibrate` CLI
- `src/check_pnl.py` — portfolio view + per-trade IC inputs
- `src/options_screener.py` — scoring + auto-log + Long Put filter
- `src/spread_scoring.py` — credit spread + iron condor enrichment
- `tests/_phase5_stress.py`, `_phase6_stress.py`, `_phase7_stress.py` — smoke tests
