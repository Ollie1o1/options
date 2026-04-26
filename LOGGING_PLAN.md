# Trade Logging Plan — Per-Structure Weight Optimization
**Updated:** 2026-04-26
**Goal:** Accumulate ~30+ closed trades **per structure** (single, spread, iron_condor) so the per-structure optimizer (`recommend_weights_for_structure`) can produce statistically meaningful weight maps for each.

---

## What Changed

The paper manager now treats credit spreads and iron condors as first-class structures (schema v10). Every logged trade — single-leg, spread, or IC — is marked-to-market with real per-leg yfinance reprices, exit-enforced through the same pipeline, and stored with its full per-component score vector at entry. This means:

- One unified DB (`paper_trades.db`) holds **all three structures**, distinguishable by the strike columns:
  - `single`     → `long_strike IS NULL AND short_put_strike IS NULL`
  - `spread`     → `long_strike IS NOT NULL AND short_put_strike IS NULL`
  - `iron_condor`→ `short_put_strike IS NOT NULL AND short_call_strike IS NOT NULL`
- IC analysis is **per-structure**: `run_paper_trade_ic_for_structure(db, structure, feature_cols)` filters the trade subset and computes Pearson IC of each component score vs `pnl_pct`.
- Weight recommendation is **per-structure**: the optimizer reads the correct config block (`composite_weights`, `credit_spread_weights`, `iron_condor_weights`), applies shrinkage `n / (n + CALIBRATION_PRIOR_N)`, caps each weight at `CALIBRATION_WEIGHT_CAP=0.30`, and renormalizes back to the original budget.

You no longer have to keep singles and spreads "separate" by hand — the structure column does the segregation.

---

## The Problem With Only Logging Top Trades
If you only log your best picks, every trade will have a high quality_score and the optimizer can't tell which factors actually matter. You need score variation — some mediocre trades in the mix.

This applies independently to **each structure**: 30 high-scoring spreads tells the spread optimizer nothing.

---

## Per-Session Routine (2× per day)

Run all three scanners back-to-back each session:

| Step | Mode | Logs |
|---|---|---|
| 1 | `DISCOVER` (singles) | top 5 + 2–3 calibration picks (rank 6–15, q≈0.40–0.65) |
| 2 | `SPREADS`            | top 3 + 1–2 calibration picks |
| 3 | `IRON`               | top 2 + 1 calibration pick |

That's ~12–14 trades per session × 2 sessions × 5 days = **~120–140 trades/week** spread across the three structures.

Auto-log shortcut for singles:
```bash
python3 run.py --default-scoring         # DISCOVER, baseline weights, top 5, dedup'd
python3 run.py -ds --10                  # same, top 10
```

For spreads/IC the screener will prompt to log; tag with `--weights NAME` to keep profiles separable.

---

## 4-Week Schedule (2026-04-27 → 2026-05-22)

### Week 1 (Apr 27 – May 1) — Cold-start across all structures
- Run all three scanners morning + evening
- Just collect — don't tune weights yet
- Target: ~70 singles, ~25 spreads, ~15 ICs

### Week 2 (May 4 – May 8) — Build IC mass
- Same routine; aim for ~30 closed trades **in each structure** (closes drive the optimizer, not opens)
- Run `--enforce-exits` daily so closes finalize quickly
- Target cumulative: ~140 singles, ~50 spreads, ~30 ICs

### Week 3 (May 11 – May 15) — First per-structure IC check
At end of week, inspect IC per structure:
```python
from src.backtester import (
    run_paper_trade_ic_for_structure,
    _SPREAD_FEATURE_COLS, _IRON_FEATURE_COLS,
)
ic_single = run_paper_trade_ic_for_structure("paper_trades.db", "single",      None)
ic_spread = run_paper_trade_ic_for_structure("paper_trades.db", "spread",      _SPREAD_FEATURE_COLS)
ic_iron   = run_paper_trade_ic_for_structure("paper_trades.db", "iron_condor", _IRON_FEATURE_COLS)
```
Per-structure thresholds are the same as before:
- IC < 0.05  → noise, keep logging
- IC 0.05–0.15 → weak signal, note which components
- IC > 0.15  → actionable

### Week 4 (May 18 – May 22) — Optimize per structure
For each structure with meaningful IC, run the recommender:
```python
from src.backtester import recommend_weights_for_structure
rec = recommend_weights_for_structure("paper_trades.db", "config.json", "spread")
print(rec["weights_key"], rec["recommended"], rec["deltas"])
```
Apply the recommended weights to the corresponding `config.json` block:
- `single`      → `composite_weights`
- `spread`      → `credit_spread_weights`
- `iron_condor` → `iron_condor_weights`

Keep logging through Week 5 to validate the new weights aren't overfit.

---

## What to Watch Each Session

| Factor | Why It Matters | Applies To |
|---|---|---|
| IV Rank at entry | Premium-selling edge | all |
| DTE at entry | 17–30 DTE sweet spot; log outliers too | all |
| Delta at entry | 0.30–0.45 standard for shorts | single, spread |
| Net delta at entry | ICs should be near 0; log non-zero too | iron_condor |
| Credit-to-width | Higher = better spread/IC edge | spread, iron_condor |
| VIX regime | Affects PoP scaling and weight multipliers | all |
| Exit reason | TP / SL / Time / Strike Breach — patterns differ by structure | all |

---

## Calibration Trades (How to Pick Them)

After logging your top picks, pick a few more that:
- Have a quality_score of **0.4–0.65** (mid-tier, not trash)
- You wouldn't normally trade — log them purely for data
- Spread across different tickers/sectors

This gives each per-structure optimizer something to compare its high-scorers against. A pure top-N log produces flat IC and useless weight deltas.

---

## What a Good Log Looks Like (sanity checks)

```sql
-- Count by structure
SELECT
  CASE
    WHEN long_strike IS NULL AND short_put_strike IS NULL THEN 'single'
    WHEN long_strike IS NOT NULL AND short_put_strike IS NULL THEN 'spread'
    WHEN short_put_strike IS NOT NULL AND short_call_strike IS NOT NULL THEN 'iron_condor'
  END AS structure,
  status,
  COUNT(*) AS n,
  ROUND(AVG(pnl_pct), 3) AS avg_pnl
FROM trades
GROUP BY structure, status;
```

Healthy state after Week 2:
- ≥30 CLOSED rows per structure
- `pnl_pct` spread across positive and negative (not all winners → IC will be flat)
- `quality_score` spread across ~0.40–0.85 (not all 0.75+ → IC will be flat)
- `exit_reason` populated for every CLOSED row (Take Profit / Stop Loss / Time Exit / Strike Breach)

---

## Progress Tracker

| Week | Sessions | Singles | Spreads | ICs | Cum Closed | IC (single / spread / IC) | Notes |
|---|---|---|---|---|---|---|---|
| 1 (Apr 27–May 1) | | | | | | — / — / — | |
| 2 (May 4–8)      | | | | | | — / — / — | |
| 3 (May 11–15)    | | | | | | — / — / — | first IC check |
| 4 (May 18–22)    | | | | | | — / — / — | apply recommended weights |

---

## Config Weights Location

`config.json` — three independent blocks now:
- `composite_weights`     — singles (27 components)
- `credit_spread_weights` — spreads (9 components incl. credit_to_width)
- `iron_condor_weights`   — ICs (7 components incl. delta_neutral)

A/B test by copying any block into `configs/weights/<name>.json` and running with `--weights <name>` (per-profile dedup keeps profiles separable in the DB).

## Key Files
- `src/paper_manager.py` — multi-leg log + exit (schema v10), real per-leg mark-to-market
- `src/backtester.py` — `run_paper_trade_ic_for_structure`, `recommend_weights_for_structure`
- `src/check_pnl.py` — portfolio view + per-trade IC inputs
- `src/options_screener.py` — scoring + auto-log entry points
- `tests/_phase5_stress.py` — exit-rule enforcement smoke test
- `tests/_phase6_stress.py` — scan-report parity smoke test
- `tests/_phase7_stress.py` — per-structure optimizer smoke test
