# Preregistration — Test O1 execution: is the real limit-fill concession `k` ≤ 0.35?

**Registered:** 2026-11-25, with **0** fills in `data/fills.db` (nothing observed).
**Implements:** `src/fill_experiment.py::o1_fill_verdict` (tests: `tests/test_o1_fill_verdict.py`).
**Shown in:** TUI `[5] LAB`, web sidebar, `opt lab fills`.

## Why

`etf_wide_spread` (Test O1: SPY 60–120 DTE, 5%-wide bull put, TP50 / 21 DTE) is **LEAD**, not
VALIDATED, for one reason: its delta-adjusted excess clears zero only if both entry and exit
fill at a concession `k = 0.35` of the half-spread or better
([RESULT_ODX_WIDE_60_120_20261124.md](RESULT_ODX_WIDE_60_120_20261124.md): crossed `k = 1`
fails by 27 bps; `k = 0.35` passes, `x_entry +1.59% [+0.26%, +2.67%]`). The backtest
**assumed** `k`; this test **measures** it.

## Definitions (frozen)

- **Sample:** rows of `data/fills.db::real_fills` whose `structure` starts with `O1`
  (case-insensitive). The TUI fill logger tags rows `O1 …` automatically when the highlighted
  row is a live `etf_wide_spread` spread. Entry and exit orders both count — the backtest
  applied `k` to both.
- **Quote:** the spread's own bid/ask at posting time: bid = natural credit
  (short bid − long ask), ask = short ask − long bid. A combo-level `k` on that quote equals
  the backtest's per-leg `k` exactly (`credit = combo mid − k × combo half-spread`).
- **Realized `k`** of a filled order = `compute_concession_pct(side, bid, ask, fill_price)`
  (0 = filled at mid, 1 = at the touch, negative = better than mid). Unfilled attempts have
  no `k`; they enter the fill rate.
- **Statistic:** the mean realized `k` over filled O1 orders (cost is linear in `k`, so the
  mean, not the median, is what the backtest's assumption is about), with one-sided 95%
  Student-t bounds `mean ± t(0.95, n−1)·sd/√n`.

## Decision rule (frozen)

| Filled O1 orders | Rule | Verdict |
|---|---|---|
| < 20 | — | **COLLECTING** (running mean shown as provisional, no verdict) |
| ≥ 20 | upper bound ≤ 0.35 | **PASS** — real execution is at least as good as the backtest assumed |
| ≥ 20 | lower bound > 0.35 | **FAIL** — execution is worse than assumed; re-run the O1 estimator at the measured mean `k` before any further O1 entry |
| ≥ 20 | otherwise | **INCONCLUSIVE** — keep collecting; at 40 fills, if still inconclusive, re-run the O1 estimator at the measured mean `k` and decide on that |

A **fill-rate flag** is raised (it does not change the verdict) when fewer than half of O1
attempts filled: an unfilled order is a trade the backtest assumed was taken.

## What PASS does and does not mean

PASS removes the one stated reason O1 is LEAD rather than VALIDATED. It does not by itself
promote O1: the promotion decision also needs the SPY-only caveat (the backtest covers SPY
2010–2023; other ETFs on the live scanner are untested and flagged `TESTED_ON_SPY_ONLY`).

## Known limitations (stated before data)

- **Adverse selection:** a passive limit order tends to fill when the market moves through it.
  The concession is measured against the quote at posting time, so this test does not see
  that cost. A later test can compare the mark a few minutes after each fill.
- **Self-reported quotes:** bid/ask are typed or pre-filled at posting time; a stale quote
  biases `k`. The TUI pre-fills them from the live chain the scan just read.
