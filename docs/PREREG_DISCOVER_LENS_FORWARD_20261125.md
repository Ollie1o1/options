# Preregistration — Discover lenses, forward record and label review

**Registered:** 2026-11-25, before the first recorded snapshot. No outcome has been computed.
**Recorder:** `src/topdown/forward_log.py` → `data/lens_forward.db` (daily, after the close, inside
startup maintenance; also `opt discover --record`).
**Report:** `opt discover --forward` (counts only until the read date — the code refuses to
compute outcomes earlier).

## Collection correction before the first readout

Scheduled snapshots wait for settled prices after 18:00 ET. A stale or degraded
board stays retryable. Outcome entry uses the first session after the later of
the row's data date and the date it was first recorded. The data date still
defines the cohort week and deduplication key. A week with a missing price path
for a listed ticker is incomplete. These changes address data availability;
the registered horizons, benchmarks, costs and statistical thresholds still
apply.

## Why

Every Discover lens carries an evidence label (PROVEN / PROMISING / TESTING / WATCHLIST / CONTEXT
/ AVOID). Without a forward record those labels can never change: nothing measures what the
lists would have done. This registers the record and the rule for reading it, before any of it
exists.

## What is recorded (frozen)

One row per (lens, ticker, data date of that row) — `INSERT OR IGNORE`, so a cached input that is
re-recorded on later days (a stale Radar snapshot) counts once, on its own date. Empty lens runs
are recorded too (an empty list is data). Newly added lenses can be recorded without entering this evaluation. Lenses evaluated:

| Lens | Label at registration | Horizon H (sessions) | Why this horizon |
|---|---|---:|---|
| core_rotation | PROVEN | 21 | monthly rebalance |
| pullbacks | PROMISING | 10 | the playbook's own max hold |
| leaders | CONTEXT | 21 | 12-1 momentum, monthly |
| revisions | TESTING | 21 | same horizon as its own test (Arm B) |
| insiders | WATCHLIST | 21 | Form 4 literature horizon |
| landmines | AVOID | 21 | the measured 20-40d window |

`etf_spreads` (options; evaluated by the O1 and B1-F tests) and `holdings` (the user's own plan)
are recorded but not evaluated here.

## Outcome (frozen)

- **Entry:** close of the first session strictly after both the row's data date and the first recorded date.
- **Exit:** H sessions after entry. Prices: `data/equity_ohlcv.db` closes.
- **Excess:** log return of the ticker minus log return of SPY over the same window, minus 10 bps
  round-trip cost.
- **One observation = one lens-week:** the equal-weight mean excess of the distinct tickers the
  lens listed with data dates in that ISO week (first appearance in the week). Rows are not
  observations — consecutive days list mostly the same names.
- **Secondary (reported, not decided on):** excess over the ticker's own sector ETF.

## Read rule (frozen)

- **No read before 2027-05-25 and 26 lens-weeks with complete outcomes.** Before that the report
  prints counts only.
- **Statistic:** mean lens-week excess; CI by moving-block bootstrap over weeks, block length
  `ceil(H/5) + 1`, 10,000 draws, seed 20261125.
- **Multiplicity:** six lenses are evaluated, so decisions use the Bonferroni 99.17% CI
  (1 − 0.05/6); the 95% CI is shown for context.

## What a read can change (proposed to a human, never automatic)

| Lens label now | Evidence at read | Proposed label |
|---|---|---|
| CONTEXT / WATCHLIST / TESTING | 99.17% CI lower bound > 0 | PROMISING |
| any | 99.17% CI upper bound < 0 | AVOID (for AVOID: confirmed) |
| PROMISING | CI spans 0 and point ≤ 0 | flagged for review |
| PROVEN | — | monitored only: its validation is the 2006-26 backtest; a forward CI upper bound < 0 flags drift |

PROVEN needs a full out-of-sample backtest; a forward read alone never grants it.

## Known limitations (stated before data)

- Recording runs when maintenance runs; a skipped day is missing, assumed unrelated to outcomes.
- Lens contents depend on upstream caches (Radar snapshot, estimates archive) whose freshness
  varies; the row's own data date is what is recorded and evaluated.
- Six months of weekly observations is a small sample. An unreadable CI at the read date means
  keep recording, not a verdict.
