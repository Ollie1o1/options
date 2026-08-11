# optionsDX results — H1, and a defect found on the way

Run 2026-08-11 against `docs/PREREG_OPTIONSDX_20260811.md`. Data: SPY EOD
2010-01-04..2023-12-29, 18,913,800 rows, 3,500 trading days, DTE 0-1096, 26.3
expirations per day.

Command:

```bash
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.alloc --odx --attribute --max-concurrent 10
```

`--max-concurrent 10` rather than the default 3: the default is an
account-level constraint that starves the per-trade sample, and H1 is a
per-trade question. One value was used and no other was tried.

---

## H1 — VERDICT: FAILS the promotion rule. Term structure is now tested.

bull_put, n=1,598, pre-registered boundary 2017-01-01 (707 in-sample / 891
holdout).

| feature | window | n | IC | t_clust | p | mono |
|---|---|---|---|---|---|---|
| **term_slope_1m3m** | full | 1597 | +0.3654 | 14.58 | 0.0000 | 0.30 |
| | in-sample | 706 | +0.3393 | 9.09 | 0.0000 | **0.70** |
| | holdout | 891 | +0.3870 | 11.70 | 0.0000 | **−0.10** |
| term_slope (unpinned) | full | 1598 | +0.0791 | 2.85 | 0.0015 | 0.10 |
| | in-sample | 707 | +0.2830 | 7.22 | 0.0000 | 0.60 |
| | holdout | 891 | **−0.0092** | −0.24 | 0.7835 | 0.10 |

Against §6, which was fixed before the run:

1. **sign holds out of sample** — PASS (+0.339 → +0.387)
2. **clustered t significant in both windows** — PASS (9.09, 11.70)
3. **quantile shape monotone in both windows** — **FAIL.** +0.70 in-sample,
   −0.10 holdout. The effect is ordered in the first half and has no shape at
   all in the second.
4. **a mechanism that is not an identity** — **FAIL**, and this is the
   substantive one:

```
term_slope_1m3m vs atm_iv              rho = +0.5743
term_slope_1m3m vs credit_pct_width    rho = +0.4985
term_slope_1m3m vs rv                  rho = +0.3334

IC of term_slope_1m3m                          +0.3654
IC after rank-residualising on atm_iv          +0.1072
```

For a held-to-expiry credit spread the return on capital is close to a function
of the credit received, and the credit is close to a function of implied vol.
A feature correlated +0.57 with the vol level inherits that mechanically.
**71% of the IC is the vol level restated.** The residual is real (p<0.0001 on
n=1,597) but small, and it does not clear criterion 3.

**Per the rule as written: DELETED, not down-weighted.** No weight changes.

### What was actually learned

The unpinned `term_slope` **flipped again** — +0.2830 in-sample to −0.0092
holdout, p=0.78. `docs/HOLDOUT_20260809.md` recorded +0.0431 → −0.0568 on the
Dolt cache and blamed the DTE 10-67 wall. On completely different data, a
different instrument set, a different span and a proper tenor range, it flips
in the same direction. **That failure was real, not a data artifact.**

Pinning the tenor does change the behaviour: `term_slope_1m3m` does NOT flip,
holding sign and significance across the boundary. So the tenor was part of the
story. But what pinning stabilises turns out to be mostly the vol level, which
the book already had.

Term structure has moved from "untested, wearing the label of a failed test" to
**tested and failed on its own terms.** That was the stated purpose of the
purchase, and the honest prior in §8 of the pre-registration called it.

### Other structures (same run, for the record)

| structure | n | win | RoC | DSR | term_slope_1m3m IC | q(BH) |
|---|---|---|---|---|---|---|
| bull_put | 1591 | 86.4% | +2.30% | 0.699 reject | +0.3654 | 0.0000 |
| bear_call | 1578 | 66.3% | −7.47% | 0.000 reject | +0.0741 | 0.0055 |
| iron_condor | 1578 | 70.8% | −2.96% | 0.000 reject | +0.1461 | 0.0000 |
| long_call | 1583 | 43.3% | +21.93% | 1.000 | +0.0319 | 0.4101 |

The slope's `mono` is **+0.30 / −0.70 / −0.60 / 0.00** across those four. A
feature whose shape reverses sign between structures is not describing one
mechanism.

long_call at +21.93% RoC over 2010-2023 is fourteen years of SPY appreciation
arriving through a long-delta instrument. It is beta, not a finding, and it is
not evidence for anything.

---

## Defect found while validating the load — this one reaches backwards

`signals.atm_iv(chain, spot)` takes the strike nearest spot **with no view on
expiry.** SPY on 2017-01-13 reported an ATM IV of **1.7%**, read off a same-day
expiry whose IV has collapsed; the 26-day expiry on that same chain read 8.7%.

Everything downstream of the level inherits it: `iv_rank`, `iv_velocity`,
`vol_of_vol`, `iv_minus_rv`.

**It is not confined to the new source.** The Dolt cache was believed to be DTE
10-67. It holds **758,273 rows under 10 DTE, some with negative DTE** — an
expiration before its own quote date. On 400 sampled symbol-days carrying such
a contract, **281 change**, median −0.0387, worst **−2.3512**: a reading of
235% "at-the-money implied vol" that was feeding the features. The sign of the
error differs by source — low on optionsDX 0-DTE, high on Dolt near-expiry — so
it is not correctable by hand.

Fixed: `ATM_IV_MIN_DTE = 10` and `_past_the_floor()` in `src/alloc/signals.py`,
applied once in `snapshot()`. It degrades rather than refuses, so a chain
entirely inside the floor is still used.

**Consequence: level-feature numbers measured before 2026-08-11 are not
comparable with anything measured after it.** Results in
`docs/HOLDOUT_20260809.md` and `docs/ATTRIBUTION_20260808.md` that turn on
`iv_rank`, `vol_of_vol` or `iv_velocity` need re-running before they are cited
again. That has NOT been done here.

**The fix broke zero of 3,843 tests** despite moving 281 of 400 sampled Dolt
symbol-days. The level features were never pinned by any test against real
chain data, only against synthetic fixtures with well-behaved expiries. That
gap is still open.

---

## H2 — VERDICT: FAILS the promotion rule, and fails the same way H1 did

Depth is now carried as `entry_depth` in `engine._entry_features`: the quoted
size on the side actually traded against (a sell hits the bid, a buy lifts the
ask), taken as the minimum across legs, because a spread is only as fillable as
its worst leg. Present on **100%** of trades from this source; None on the Dolt
cache, which has no size columns.

### Scope correction, stated plainly

The pre-registration says the outcome is "realized friction". **That is not
answerable here and was not answered.** This backtest has no fills:
`friction_pct_credit` is `crossing_cost`, half the quoted bid-ask summed over
the legs, taken from the *same quote* the depth comes from. Held to expiry
there is no exit crossing, so entry friction is the whole round trip — but it
is a modelled cost, not a realized one. Two answerable questions were
substituted, and the substitution is recorded here rather than in a footnote.

### H2a — is depth redundant with the spread? NO, and that is worth knowing

bull_put, n=1,598:

```
depth vs friction_pct_credit    rho = -0.0618   p=0.014
depth vs credit_pct_width       rho = -0.2776
depth vs atm_iv                 rho = -0.3055
depth vs dte                    rho = -0.0032
```

Depth median 149, IQR 72-392, max 5,662. A correlation of −0.06 with the
spread means **quoted depth and quoted spread are close to independent
dimensions of liquidity.** The cost model uses only the spread, so depth is
genuinely new information. That is the one durable result in this section.

### H2b — does depth predict the outcome? No, once credit richness is removed

| window | n | IC | t_clust | p | mono |
|---|---|---|---|---|---|
| full | 1598 | −0.1907 | −7.83 | 0.0000 | 0.10 |
| in-sample | 707 | −0.0863 | −2.25 | 0.0218 | 0.20 |
| holdout | 891 | −0.2208 | −6.86 | 0.0000 | −0.10 |

Sign holds and both windows are significant — criteria 1 and 2 pass. Then:

```
IC of entry_depth on RoC
  raw                            -0.1907   p=1.5e-14
  less friction_pct_credit       -0.1947   p=4.1e-15
  less atm_iv                    -0.0435   p=0.082
  less credit_pct_width          -0.0155   p=0.535
  less all three                 -0.0099   p=0.693
```

Removing the spread changes nothing — consistent with H2a. Removing the **vol
level** kills it. Removing **credit richness** kills it outright. The quintile
table has no shape at all (+0.0126, +0.0438, +0.0120, +0.0302, +0.0198): bucket
2 is the best and bucket 1 second-worst.

Criterion 3 FAILS (mono +0.20 → −0.10, flat buckets). Criterion 4 FAILS
(IC −0.0099, p=0.69 once credit richness is controlled).

iron_condor is worse: in-sample IC −0.0090 at **p=0.81**, so criterion 2 fails
before the rest, and `mono` reverses +0.50 → −0.60 across the boundary.

**DELETED, not down-weighted.** No filter added.

### The meta-finding, which outlasts both hypotheses

Run the same treatment on H1's feature:

```
term_slope_1m3m   raw               +0.3654   p=1.2e-51
                  less atm_iv       +0.1072   p=1.8e-05
                  less all three    +0.0422   p=0.092
```

H1's surviving residual does not survive either. Both hypotheses produced a
large, highly significant IC, and in both cases **the IC was credit richness
wearing a different name.**

That is mechanical, not coincidental. For a held-to-expiry credit spread,
return on capital is close to a function of the credit received, and the credit
is close to a function of implied vol. **Any entry feature correlated with
implied vol will post a large IC on this book, and that IC means nothing.**
`atm_iv` itself scores +0.5455; `credit_pct_width` scores +0.6991.

This plausibly explains a long run of findings in this repo that looked strong
and then failed a holdout. A raw IC on this book is not evidence.

### The screen, now shipped

`attribution.residual_ic()` reports the IC with `credit_pct_width` and `atm_iv`
regressed out on ranks, carried on every row as `ic_resid` and printed as the
**`IC|ctl`** column beside the raw IC. It is the fifth standing screen, added
for the same reason as `mono` and `tail_auc`: something got through without it.

Applied to the same n=1,598 bull_put sample, it does not only reproduce the two
failures — it reaches an older one:

| feature | raw IC | IC\|ctl | p | controls used |
|---|---|---|---|---|
| credit_pct_width | +0.6991 | +0.4687 | 0.0000 | atm_iv |
| atm_iv | +0.5455 | +0.1146 | 0.0000 | credit_pct_width |
| term_slope_1m3m | +0.3654 | +0.0430 | 0.0855 | both |
| **iv_rank** | **+0.1803** | **−0.0053** | **0.8325** | both |
| entry_depth | −0.1907 | −0.0099 | 0.6917 | both |
| friction_pct_credit | −0.0747 | +0.0135 | 0.5891 | both |

**`iv_rank` is credit richness too.** It was zeroed on 2026-08-09 for failing a
second holdout, and this is the mechanism behind that failure rather than
another symptom of it.

What is left standing is `credit_pct_width` at +0.4687 — and that is the
accounting identity, not a signal: for a held-to-expiry credit spread, return
on capital *is* roughly credit over width. Once it is removed, nothing in the
19-feature family predicts the outcome. That agrees with
`docs/ATTRIBUTION_20260808.md`, which reached the same conclusion with weaker
tools.

Design notes worth keeping:

- A control is never regressed against itself; the residual would be
  identically zero, which reads as "disproven" when it means "not asked". A
  feature whose only control is itself reports `-`.
- Residualisation is on **ranks**, matching the rest of the module. These
  relationships are monotone but not linear, and a least-squares fit on raw
  values leaves most of the control's influence in the residual.
- A feature that is a **renamed control** leaves a residual of pure
  floating-point noise, and ranking that noise scored **+0.97** before a
  collinearity guard was added — the worst available failure, since the screen
  would have endorsed exactly what it exists to catch. Caught by a test.

## H3 — VERDICT: the COST model breaks outside the old window. The outcome model is untestable here.

bull_put, weekly entry (697 dates over 5,107 days), `max_concurrent=200` so the
cap does not bind, six DTE bands. Two methodological choices made before
looking, both of which change the reading:

* **Entry is weekly** so every band gets the same number of entry
  opportunities. Under a binding concurrency cap a long tenor takes fewer
  entries purely because each occupies a slot longer, which confounds tenor
  with sample size.
* **Held-to-expiry positions overlap**, worse the longer the tenor: at 578-day
  holds roughly 70 consecutive weekly entries are live at once and share almost
  all of their outcome. So an **effective n** (calendar span ÷ mean hold) is
  reported beside the nominal one, and `t_eff` deflates by it.

| band | n | effN | hold | win% | meanRoC | medRoC | **fric%cr** | cred%w | t | t_eff | q(BH) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10-25 | 628 | 344 | 15 | 83.8 | −0.18 | 10.86 | **4.8** | 11.3 | −0.14 | −0.10 | 0.9185 |
| 25-60 | 685 | 169 | 30 | 86.6 | 1.98 | 13.64 | **6.4** | 12.8 | 1.51 | 0.75 | 0.5453 |
| 60-120 | 691 | 70 | 73 | 89.1 | 6.25 | 16.55 | **4.9** | 14.8 | 4.91 | 1.57 | 0.1763 |
| 120-250 | 693 | 35 | 145 | 91.5 | 10.37 | 18.48 | **10.9** | 15.9 | 9.13 | 2.06 | 0.1185 |
| 250-500 | 668 | 19 | 274 | 93.4 | 11.92 | 19.62 | **58.9** | 15.7 | 10.96 | 1.83 | 0.1342 |
| 500-1000 | 550 | 9 | 578 | 98.5 | 18.86 | 21.36 | **78.9** | 16.8 | 28.01 | 3.55 | 0.0023 |

RoC is **net of crossing** — `fill_with_reason` takes the bid when selling and
the ask when buying, so entries are filled at the touch, not at mid.

### The finding: friction is not stationary in tenor

The mean alone overstates it, so read the distribution:

| band | fric mean% | fric median% | % over the 25% gate |
|---|---|---|---|
| 10-25 | 4.8 | 3.6 | 0.8 |
| 25-60 | 6.4 | 4.8 | 1.2 |
| 60-120 | 4.9 | 4.2 | 0.3 |
| 120-250 | 10.9 | 7.3 | 3.5 |
| 250-500 | 58.9 | 13.0 | **26.9** |
| 500-1000 | 78.9 | 23.6 | **46.0** |

Mean and median diverge violently past 250 days — 58.9 against 13.0, 78.9
against 23.6 — so the long-tenor cost is a **heavy tail**, not a uniform
shift. Most long-dated trades cost 13-24% of credit to enter; a large minority
cost multiples of it.

Both moments still move hard. The median roughly **quintuples** from 4.8% to
23.6%, and the share breaching the 25% friction gate goes from **~1% inside
the old window to 46% beyond 500 days**. The gate is not broken out there — it
fires, and correctly — but it was calibrated in a region where it essentially
never binds, and beyond 250 DTE it silently becomes the dominant filter,
rejecting between a quarter and a half of all candidates.

This is the actionable H3 result. Any use of long-dated structures needs its
own cost model and its own gate calibration; the current numbers describe a
regime that stops applying somewhere around 120-250 DTE.

### The apparent tenor edge is the bull market, not the tenor

Mean RoC rises monotonically with tenor and win rate reaches 98.5%. Neither is
evidence:

* **Effective n collapses 344 → 9.** Nine non-overlapping 578-day periods in
  fourteen years is not a sample. `t` reads 28.01 and `t_eff` 3.55, and even
  that overstates it.
* **The window is one secular bull market.** SPY runs **+319.6%** from
  2010-01-04 to 2023-12-29, max drawdown −34.3%, and 2008 is not in the data.

Measured directly on the same prices, the share of holding windows that ended
*lower*:

| hold | windows ending lower | band win% |
|---|---|---|
| 30d | 33.8% | 86.6 |
| 145d | 24.2% | 91.5 |
| 274d | 19.4% | 93.4 |
| 578d | **11.9%** | **98.5** |

The win rate tracks the drift base rate. A 25-delta short put wins whenever the
index does not fall far, so a period in which the index quadrupled produces
exactly this ordering with no tenor effect whatsoever. It is the same beta
confound as `long_call`'s +21.93%.

And it is the structure this book has already been burned by:
`docs/ATTRIBUTION_20260808.md` records that **every spread open into the COVID
crash lost 100% of capital at risk**. A 578-day short put is that exposure held
twenty times longer, with nine independent observations, in a window whose two
drawdowns both fully recovered inside the holding period.

**Nothing promoted.** H3's outcome side is not answerable on this data; its
cost side is answered and the answer is a warning.

**H4 (time of day)** remains untestable and unstarted. It needs the 30-minute
intraday files, which were deliberately not bought, and the `odx_chain`
primary key must gain `quote_time` before any intraday file is loaded or
`INSERT OR REPLACE` will silently keep one snapshot per day.

## Methodology shipped with this run

- **Benjamini-Hochberg across the search family**, reported as `q(BH)`.
  `n_trials` had been carried on every ranking row since the module was written
  and never divided by, so every multi-feature sweep in this repo before today
  was read at raw p. The family here is 19.
- **`split_at_date`** — the boundary is a date, because `split_by_time(0.7)`
  cuts at a fraction of trade count and drifts whenever the count does.
- **`term_slope_tenor`** — interpolated in total variance between bracketing
  expiries. Nearest-listed-expiry was tried first and rejected on coverage:
  only 72.5% of days carry an expiry within ±10d of both 30 and 90, and which
  days fail is set by position in the expiry cycle, which puts a periodic hole
  in a time-series signal. Interpolation covers 99.9%. Disclosed in §2 of the
  pre-registration as a data-dependent design choice made on availability, not
  on any outcome.
