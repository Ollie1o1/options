# Trade Logging Plan — Weight Optimization
**Started:** 2026-04-14
**Goal:** Accumulate ~150-200 trades with enough score variance to meaningfully optimize quality_score weights

---

## The Problem With Only Logging Top Trades
If you only log your best picks, every trade will have a high quality_score and the optimizer can't tell which factors actually matter. You need score variation — some mediocre trades in the mix.

---

## Per-Session Routine (2x per day)

Run the default screener (option 3 — DISCOVER) each session.

**Log these trades every session:**
- ✅ **Top 5** from the screener results (your real candidates)
- ✅ **2-3 "calibration" trades** from rank 6–15 (don't need to take these positions — log them for score variance)

That's ~7-8 trades per session × 2 sessions × 5 days = **~70-80 trades/week**

---

## 4-Week Schedule

### Week 1 (Apr 14–18) — Baseline
- Run DISCOVER scan morning + evening
- Log top 5 + 2 calibration trades each session
- Don't touch config weights yet — just collect
- Focus: CALLS and PUTS (no spreads this week — keep strategy type clean)
- Target: ~70 trades

### Week 2 (Apr 21–25) — Add Spreads
- Same routine as Week 1
- Now also log from the SPREADS scanner (option 5)
- Keep spreads and naked options separate — tag them clearly
- Target: ~140 trades cumulative

### Week 3 (Apr 28 – May 2) — First IC Check
- Same logging routine
- At end of week, check Information Coefficient in portfolio view
  - IC < 0.05 → noise, keep logging
  - IC 0.05–0.15 → weak signal, note which factors
  - IC > 0.15 → actionable, ready to adjust weights
- Target: ~210 trades cumulative

### Week 4 (May 5–9) — Optimize
- If IC is meaningful, adjust weights in `config.json` based on which sub-scores correlated best
- Keep logging to validate the new weights aren't overfitted
- Target: ~280 trades cumulative

---

## What to Watch Each Session

| Factor | Why It Matters |
|--------|---------------|
| IV Rank at entry | Most predictive for premium selling — high IVR = better edge |
| DTE at entry | 17–30 DTE is the sweet spot; log outliers too |
| Delta at entry | ~0.30–0.45 is standard; log some 0.20 and 0.50 to compare |
| VIX regime | Note if VIX is low/normal/high — affects expected win rate |
| Exit reason | Take profit vs stop loss vs time exit — patterns emerge over time |

---

## Calibration Trades (How to Pick Them)

From DISCOVER results, after logging your top 5, pick 2-3 more that:
- Have a quality_score of **0.4–0.65** (mid-tier, not trash)
- You wouldn't normally trade — just logging for data
- Spread across different tickers/sectors if possible

This gives the optimizer something to compare against the high-scorers.

---

## Progress Tracker

| Week | Sessions | Trades Logged | Cumulative | IC | Notes |
|------|----------|---------------|------------|-----|-------|
| 1 (Apr 14–18) | | | | — | |
| 2 (Apr 21–25) | | | | — | |
| 3 (Apr 28–May 2) | | | | — | |
| 4 (May 5–9) | | | | — | |

---

## Config Weights Location
`C:\Users\Oliver\desktop\options\config.json` — quality_score weights to adjust in Week 4

## Key Files
- `src/paper_manager.py` — trade logging logic
- `src/check_pnl.py` — portfolio view + IC calculation
- `src/options_screener.py` — scoring logic
