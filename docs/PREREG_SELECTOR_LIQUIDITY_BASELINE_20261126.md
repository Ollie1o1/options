# Preregistration — generic bull-put selector vs a matched liquidity-first baseline (paper)

**Registered:** 2026-11-26, before any study outcome existed. One archived session (2026-11-25)
had been captured, and all of its entries are open.
**Code:** `src/selector_study/` → `data/selector_study.db` (separate from `paper_trades.db`,
`candidates.db` and every promotion gate).
**Report:** `opt lab study readout`. Before the read date it prints counts only. The code
withholds returns until then.
**Scope:** paper research. Nothing here places, sizes, alerts on or promotes a trade, or changes
a scoring weight.

## Question

`opt lab scan-replay` showed the generic selector is repeatable on identical quotes. This study asks
whether it chooses a **better-paying contract** than a simple, transparent alternative that
looks at liquidity alone, on the same ticker, session and expiration, after costs.

## Workflow (one documented path)

```
opt lab study run            # capture every archived session -> build both arms -> mark -> readout
opt lab study capture        # step 1 alone  (--session YYYY-MM-DD, repeatable)
opt lab study mark           # step 2 alone
opt lab study readout        # step 3 alone  (--json for the machine-readable form)
opt lab study demo --dir D   # the same workflow on synthetic data in D (never data/)
```

Maintenance runs the same capture and marks automatically after the day's chain archive is
complete (`config.json → selector_study.enabled`). With the study enabled, the scheduled
archive also covers the 40 study symbols.

## 1. Capture (frozen)

- **Source:** `data/chain_archive.db`, the scheduled daily archive (weekday afternoons).
- **Symbols:** the 40 in `src/selector_study/db.py → AGREED_SYMBOLS` (the local archive refresh
  universe). They are frozen into the study database the first time it is created.
- **Time:** observations must be on the session date and between **14:00 and 20:00 ET**.
  The readout reports the capture-time range and how many observations were taken after 16:00.
  Post-close quotes are wider. Both arms share each snapshot, so capture time cannot favour either arm.
- **Quote time:** Yahoo gives no bid/ask timestamp. `observed_at` is the request time and
  `quote_time` is stored as NULL. Last trade time is kept per leg but is not a quote time.
- **One observation per (session, symbol)**, empty ones included:

  | status | meaning | retried? |
  |---|---|---|
  | captured | usable two-sided puts in the 14–58 DTE window; selector ran (may pick nothing) | no, frozen |
  | missing_chain | no archived rows; the archive recorded no provider error | yes |
  | provider_failure | no archived rows; the archive recorded an error or dead feed | yes |
  | unusable_quotes | rows exist, but no expiration in the window has two-sided puts | no |
  | missing_spot | rows without an underlying price | no |
  | stale | rows observed on a different day than the session | no |
  | off_window | observed outside 14:00–20:00 ET, or with no observation time | no |

  Each capture run is logged (`capture_runs`: running / partial / complete). Retries only touch
  the two retryable statuses, and every entry key is deterministic, so re-runs never duplicate.

## 2. The two arms (frozen)

- **Structure:** bull put credit spread, one expiration. It is the vertical the archive can
  mark leg by leg. Calendars stay out, and so does the 60–125 DTE region, which belongs to Test O1.
- **Selector arm:** `src.tui.options_engine._build_candidates_for_slice(strategy_filter="bull_put")`
  on the captured slice nearest 35 DTE in 14–58 DTE, `max_friction_pct = 0.28`. This is the same
  call `scan-replay` checks.
- **Baseline arm (liquidity-first):** in the **same slice** (ticker, session, expiration), every
  bull-put vertical with
  - short put |delta| within **±0.05** of the selector's short |delta|,
  - width within **±25%** of the selector's width,
  - short bid ≥ 0.10, long bid ≥ 0.02, ask ≥ bid, open interest ≥ 1 on both legs,
  - mid credit > 0.10, crossed credit > 0, mid credit < width, round-trip spread / mid credit ≤ 0.28.

  **Rule:** lowest round-trip bid-ask spread divided by mid credit. **Tie-breaks:** smaller
  short-delta gap, then smaller width gap, then higher short strike, then higher long strike.
- The selector's own legs are eligible, so the baseline can pick the **identical** spread. Those pairs
  count with a difference of zero, and the readout shows how many there are.
- **No match** (no eligible vertical, legs absent from the captured quotes, no short delta) stays
  visible as a `no_match` pair with its reason. The readout reports how many comparisons were possible.
- Each entry freezes its legs (bid, ask, mid, delta, IV, OI, volume, snapshot and last-trade
  times), spot, observation time, source, earnings days (only when captured on the session day
  itself, never retroactively) and the settings hash.

## 3. Marks and outcomes (frozen)

- **Marks:** each trading session after entry, up to expiration, from the same archive. Every
  session gets a row: `marked`, `unpriceable`, `missing_leg` or `missing_session`. Gaps are retried
  on later runs. At expiration the underlying's close from `data/equity_ohlcv.db` settles at
  intrinsic value. If no close is found within five days, the outcome is recorded as missing.
- **Identical rules for both arms**, read from `config.json → exit_rules`: take profit at 50% of
  the entry credit, stop at −100%, time exit at ≤ 21 DTE, no exit before 3 calendar days.
- **Costs:** `paper_trading.commission_per_contract` per leg per side (US$0 at registration).
  - **Primary — crossed:** enter at short bid − long ask, exit at short ask − long bid, plus fees.
  - **Sensitivity — midpoint:** every leg at mid, plus fees. The fill is assumed, not shown.
    A quote snapshot cannot establish that a passive order filled.
  Each case walks the marks with its own prices, so an exit can fall on a different day in each case.
- **Statuses:** closed, open, **not_filled** (no positive credit after costs), **missing** (more
  than 3 consecutive sessions without a usable mark, or expiration passed with no settlement price).
- **Return:** P&L / capital at risk, both per share. Capital at risk = width − entry credit.

## 4. Readout (frozen)

- **Pair difference:** selector − baseline return, for pairs where **both** arms closed under that cost case.
  Open, missing and not-filled pairs are counted and excluded, never imputed.
- **Clustering:** one observation = one **ISO week** of entry sessions (the equal-weight mean of
  that week's pair differences). Candidates from one week share one market path.
- **Interval:** 95% moving-block bootstrap over consecutive weeks, block 3 weeks, 10,000 draws,
  seed 20261126. There is one primary hypothesis, so no multiplicity adjustment.
- **Read date:** no returns before **2027-03-01**.
- **Minimum evidence:** ≥ **12 weeks** and ≥ **40 resolved pairs** in the crossed case. Below that,
  the verdict is **Insufficient data**.
- **Verdict (crossed case):** CI lower bound > 0 → *Selector ahead of the liquidity baseline*;
  upper bound < 0 → *Liquidity baseline ahead of the selector*; otherwise *No difference
  established*. When the midpoint and crossed means disagree in sign, the readout says that
  costs reverse the midpoint result.
- A verdict proposes, to a human, whether the generic selector should prefer liquidity. It
  changes nothing by itself and is not a promotion under `DECISION_RULE_20261015.md`.

## Known limitations (stated before data)

- Capture depends on the archive job running. A day it does not run is missing, and that is assumed
  unrelated to outcomes.
- The archive keeps strikes within ±15% of spot and OI ≥ 1. A leg that leaves the band shows as
  `missing_leg`.
- Settlement closes come from the local OHLCV archive, which is split-adjusted. A split during a
  holding period would misprice that settlement. This is rare for these symbols, but it is not handled.
- Crossed prices assume a marketable order fills at the displayed quote, in any size. No fill is measured.
- Most baseline picks may equal the selector's pick. That is a result in its own right: it means the
  selector already chooses liquidity in practice.

## Verified at registration

- `tests/selector_study/` (16 tests): duplicates, stale and off-window inputs, partial provider failure
  and retry, no eligible match, missing marks, a cost reversal of a midpoint advantage, the
  read-date lock, and an end-to-end synthetic multi-session run.
- `opt lab study demo` (synthetic, 85 sessions, 6 symbols) runs the full workflow to a verdict.
  Synthetic results say nothing about markets.
