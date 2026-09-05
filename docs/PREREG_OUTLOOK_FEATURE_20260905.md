# Pre-registration — does the outlook composite transfer to single names?

**Frozen 2026-09-05. Immutable.**

Written before this feature has ever been computed against a real trade's
outcome, and before `outlook_composite` has been fed into `feature_ic` or
`residual_ic` on any real data. Only a data-coverage check was done first
(`data/dolt_options.db::stocks_close` has history for 109 of the ~120-symbol
universe, 2016-2026) — a coverage count, never an outcome, matching every
other prereg here.

## 1. Why this

`src/outlook/` is validated: on its 16-instrument sector/asset-ETF universe,
bullish calls are right 66% of the time at 2mo and 72% at 3mo, and the real,
survives-scrutiny edge is a relative cross-sectional rank IC of +0.05 to
+0.08 (`docs/OUTLOOK_FINDINGS.md`). It has never been connected to a single
options trade — it ships as a display-only narrative box
(`src/outlook/display.py`) and nothing in `src/paper_manager.py` or
`src/alloc/` reads it.

Your book is short-premium (Bull Put ~88% of allocation), which is exactly
the shape this signal fits best: a bull put doesn't need a rally, it needs
the absence of a bad fall, and `outlook`'s validated skill is precisely on
distinguishing "won't underperform" from "might" — not the harder,
structurally-unfixable "will fall" call.

## 2. Hypothesis

**H-OUTLOOK.** `outlook_composite` — computed by reusing `src/outlook/
factors.py`'s formulas (`mom_12_1`, `trend_score`, `reversal_1m`,
`relative_strength` vs SPY) and `src/outlook/engine.py::rank_universe`'s
cross-sectional z-score/composite logic, UNCHANGED, applied point-in-time to
the ~109-symbol single-name universe instead of the 16 sector/asset ETFs it
was validated on — has a residualized IC (`residual_ic`, controlling for
`credit_pct_width`/`atm_iv`, identical to every other feature in
`ATTRIBUTION_FEATURES`) against realized `bull_put` return on capital, on
the 2020-21 holdout, that clears Harvey's `|t| ≥ 3.0` hurdle
(`src/alloc/report.py::MIN_TSTAT`).

**This is a transfer test, stated as one.** The factor formulas and weights
are frozen exactly as `src/outlook/engine.py` already has them — nothing is
re-tuned for the new population. A null result here says "the construction
that works on 16 broad, comparatively low-idiosyncratic-vol instruments does
not carry its skill onto noisier single names," which is itself a real,
useful answer, not a failure to find something that was always going to be
there.

**Sign convention, fixed now:** `outlook_composite` is built so that a
higher score means more bullish. For a bull put — profits from the
underlying NOT falling — the expected sign is **positive**: a more bullish
underlying should mean a better realized return on that structure. A
significant NEGATIVE ic would mean the composite is inversely related to
short-put safety and is reported as such, not reframed.

## 3. Population and windows — LOCKED, matching every other feature test

Identical to `docs/PREREG_ENSEMBLE_20260905.md` and `docs/HOLDOUT_20260809.md`,
for direct comparability:

- **In-sample:** `--start 2022-01-07 --end 2024-12-31`.
- **Holdout (the one look):** `--start 2020-01-27 --end 2021-12-31`.
- **Universe:** `--all-names`, the full 123-symbol cache — of which 109 have
  the `stocks_close` history this feature needs; the other 11 (mostly
  delisted/acquired names — `WLTW`, `PBCT`, `CDAY`, `AAWW`, `HA`, and a few
  sub-industry SPDRs that were never individual equities) simply report
  `outlook_composite` as unmeasurable for those symbols, the same as any
  other feature's coverage gap.
- **Structure:** `bull_put` only. This is deliberately narrower than the
  ensemble prereg's bull_put + long_call — the mechanism argument in §1 is
  specific to short premium and does not extend cleanly to a long call
  (a bullish signal helps a long call too, but that strategy is already
  independently known to be structurally cost-challenged regardless of
  timing quality, so a long_call test would confound "does the signal work"
  with "can the strategy ever be profitable at all").

## 4. The feature — LOCKED

- **Computed by** `src/outlook/cross_sectional.py::composite_lookup`, built
  in this PR. Reuses `src/outlook/factors.py` and `rank_universe` verbatim.
- **Point-in-time, by construction:** for a target date, each symbol's
  factor row uses only that symbol's own close history up to and including
  that date (`_index_asof`), and the benchmark (SPY) is looked up
  independently by date rather than by a shared positional index — two
  series with different gap patterns must never be compared at the same
  array position, or `relative_strength`'s lookback silently reads the wrong
  calendar date for one side.
- **Cross-sectional universe for z-scoring:** whatever symbols in the
  123-name universe have enough history as of that date — not the original
  16 ETFs. A date where fewer than 2 symbols have enough history is skipped
  entirely (nothing to rank against).
- **Wired into `replay()`** via a new optional `outlook_lookup` parameter
  (`src/alloc/engine.py`) — additive only. Trades from a `replay()` call
  without it are byte-identical to before this PR; the 3 existing tests
  covering `outlook_lookup`'s absence confirm this.

## 5. Decision rule — LOCKED

Identical shape to `PREREG_ENSEMBLE_20260905.md` and
`PREREG_GATE_RD_20260902.md`:

- **REAL**: holdout `|t_clustered| ≥ 3.0` AND the sign matches §2's
  prediction (positive).
- **NULL**: `|t_clustered| < 3.0`.
- **INVERTED**: `|t_clustered| ≥ 3.0` but negative — the construction is
  significantly related to outcome in the WRONG direction, reported as such.
- One look. No EXTEND state.

## 6. What's built now vs. what happens next

**Built and merged in this PR, before any real number exists:**
`src/outlook/cross_sectional.py` (`composite_lookup`), the `outlook_lookup`
wiring into `src/alloc/engine.py::replay()`, `"outlook_composite"` added to
`ATTRIBUTION_FEATURES` (reports "not measurable" until a lookup is actually
supplied — same convention as `entry_depth` on the Dolt cache), 12 new unit
tests on synthetic close series (date-alignment correctness is the one this
design lives or dies on — see the dedicated test for a symbol with a missing
trading day against a benchmark that has none) plus 3 more confirming the
`replay()` wiring is purely additive.

**Explicitly NOT done in this PR:** no code that fetches real closes for the
109-symbol universe from `data/dolt_options.db::stocks_close` and builds a
real `outlook_lookup` dict, and therefore no run of `--attribute` that
actually populates `outlook_composite` on a real trade, and no holdout
number. That is the next step, as its own separate PR/commit — matching how
`PREREG_GATE_RD_20260902.md` / `GATE_RD_RESULT_20260902.md` and
`PREREG_ENSEMBLE_20260905.md` were both sequenced.

## 7. Explicitly out of scope this round

- `long_call` (see §3 for why).
- Re-litigating `trend`/`ret_4w`, the existing single-factor momentum
  features already measured flat in `docs/INTEL_BACKTEST_FINDINGS.md` — this
  is a different, more sophisticated, already-independently-validated (on a
  different universe) construction, not a re-run of those.
- Re-tuning `rank_universe`'s weights or thresholds for the wider universe.
  If this transfer test fails, the right next question is "does it fail
  because single names are just noisier" (informative, stop here) or "would
  it work with weights re-fit for single names" (a DIFFERENT, new hypothesis
  requiring its own registration, not a patch to this one).
