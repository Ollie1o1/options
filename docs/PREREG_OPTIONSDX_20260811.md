# Pre-registration — optionsDX SPY 2010-2023

**Written 2026-08-11, after loading the data and BEFORE running any hypothesis
test.** Nothing in §3-§6 has been evaluated at the time of writing. The
coverage numbers in §2 are the only things measured so far, and they are
properties of the chain calendar, not of any outcome.

The point of this file is that it can be checked against later. If a result
appears that this document did not license, that result is a search finding and
must be labelled one.

---

## 1. Why this exists

`docs/HOLDOUT_20260809.md`:

    term_slope    in-sample +0.0431    holdout -0.0568    FLIPS

That doc's own diagnosis was that the Dolt cache is DTE 10-67, so the
measurable slope was 10d against 60d rather than the 1M/3M the term-structure
literature means — "a data wall, not a fixable specification."

So term structure was never tested. It was wearing the label of a failed test.
This data settles which it is.

## 2. What is loaded (measured, 2026-08-11)

Source: optionsDX free EOD chain exports, SPY, 2010-2023, 20 archives → 168
monthly files → `data/optionsdx.db`, table `odx_chain`.

| | Dolt cache | optionsDX |
|---|---|---|
| mean expirations per symbol-day | 2.6 | **14.7** |
| deepest DTE | 67 | **1,094** |
| coverage | 2020-2026, 121 symbols | 2010-2023, SPY |

Both walls that motivated the load are gone.

Tenor availability, measured across 3,058 SPY quote dates:

| method | days with BOTH a 1M and a 3M reading |
|---|---|
| nearest listed expiry within ±5d | 42.6% |
| nearest within ±10d | 72.5% |
| nearest within ±10d / ±15d | 90.2% |
| **bracketed interpolation** | **99.9%** |

**Disclosed deviation:** `term_slope_tenor` was first written to take the
nearest listed expiry within a tolerance, and was changed to bracketed
interpolation *after* seeing the table above. That is a data-dependent design
choice and is recorded here rather than hidden. It was made on feature
AVAILABILITY and no outcome was examined. The reason it is not a free
parameter to tune: at ±10d the 27.5% of days with no reading are not a random
sample — which days fail is fixed by where the date sits in the expiry cycle,
so a nearest-neighbour spec puts a periodic hole in a time-series signal.
Interpolation is the standard VIX construction, linear in total variance, and
it is the specification for every test below. It will not be revisited.

## 3. LOCKED SPLIT — fixed 2026-08-10, before any data existed

> **In-sample 2010-2016. Holdout 2017-2023.**

Chronological. Do not reverse it: a signal must survive FORWARD in time, which
is the direction it would be traded, and this places Volmageddon (Feb 2018), Q4
2018, COVID and the 2022 bear in the HOLDOUT where the stress belongs.

**Moving this boundary after seeing a result invalidates the exercise.**

## 4. Hypotheses

Each states the feature, the outcome, and what would count as failure. All are
two-sided: a strong inverted result is a finding, not a rescue of the original
direction.

**H1 — a real 1M/3M term slope predicts short-premium outcome.**
Feature `term_slope_1m3m` = ATM IV at 30 days minus ATM IV at 90 days, both
interpolated in total variance from the bracketing expiries
(`src/alloc/signals.py::term_slope_tenor`). Positive is backwardation. Outcome
is realized return on capital at risk for bull-put spreads. Prediction, from
the literature: backwardation at entry precedes worse outcomes.
`term_slope` (the old unpinned nearest-vs-farthest version) is measured
alongside it, unchanged, because the open question is whether the tenor or the
signal was at fault, and that comparison is the entire point.

**H2 — quoted depth at entry predicts realized friction.**
`C_SIZE`/`P_SIZE`, loaded as `bid_size`/`ask_size`. Outcome is realized
round-trip friction as a fraction of credit. This column has never existed in
any source this repo has held.

**H3 — structures beyond the old 10-67 DTE window behave differently.**
Now testable because the chain reaches 1,094 days. Descriptive first: does the
existing cost and outcome model hold at tenors it was never fitted on?

**H4 — time of day changes realized friction.** *NOT TESTABLE with this data.*
Requires the 30-minute intraday files, which were deliberately not bought, and
those cannot be loaded until the `odx_chain` primary key carries `quote_time`
(see §7). H4 stays open and unstarted; it is listed so that its absence is not
later mistaken for a negative result.

## 5. Multiple testing

Every hypothesis is run through `src.alloc --attribute`, which reports Spearman
IC, a t clustered by entry day, the quintile shape (`mono`), and the tail AUC.

**Benjamini-Hochberg is applied across the whole feature family in a run**
(`src/alloc/attribution.py::benjamini_hochberg`, added 2026-08-11 and reported
as the `q(BH)` column). Before this, `n_trials` was carried on every row and
never divided by, so every multi-feature sweep in this repo was read at raw p.
The family is every feature examined in the run — currently 19 — not the
subset that looked interesting afterwards.

A raw p that survives while its q does not is a search artifact. Report both.

## 6. Promotion rule — fixed in advance

A feature ships only if **all four** hold:

1. sign holds out of sample,
2. clustered t significant in BOTH windows,
3. quintile shape monotone in BOTH windows,
4. a stated mechanism that is not an accounting identity.

Everything else is **DELETED, not down-weighted.** That rule is why `vol_of_vol`
went — the shape screen exposed it as arithmetic rather than a discovery.

Real money stays OFF regardless of outcome. No gate here authorises it.

## 7. Known trap, not yet armed

`odx_chain`'s primary key is `(symbol, date, expiration, strike, type)`.
Intraday files carry many snapshots per `QUOTE_DATE` separated by
`QUOTE_READTIME`, so `INSERT OR REPLACE` would silently keep only the last
snapshot of each day. EOD is unaffected and has been verified unaffected — the
loaded files carry `QUOTE_TIME_HOURS = 16.0` and one snapshot per date.

**Add `quote_time` to the key before any intraday file is ever loaded.**

## 8. Honest prior

H1 most likely fails again. Six years, 12.1M rows and dozens of features have
produced essentially nothing that survived a holdout. The value here is that
"untested" and "tested and failed" are different states, and term structure has
been in the first while labelled as the second.

The data cost nothing. There is no sunk cost to justify and no reason to find
something.
