# The condor composite is close to inverted — 2026-08-07

`docs/MULTILEG_COMPOSITE_20260807.md` §4 left a lead: iron condors are the only
multi-leg structure with a positive median return net of crossing, yet the
composite ranks them worst (rank IC −0.0965). This is that lead run down.

It resolves into the clearest result of the whole scoring review, and unlike the
rest of it, the mechanism is identified and the sign is stable everywhere it was
checked.

Reproduce: `scripts/measure_multileg_composite.py` and the decomposition in
§2 below.

---

## 1. First, the reconstruction gap was mine

The previous document could not reconstruct condor scores and blamed
`max_net_delta` or weight drift. Neither: **the condor branch uses a different
credit-to-width normalisation from the vertical branch** —
`clip((c2w − 0.10)/0.20)` against the vertical's `clip((c2w − 0.20)/0.30)`,
with the comment "iron condor credit/width is naturally lower than vertical"
(`spread_scoring.py:253`). I had applied the vertical formula.

With the right one, **121 of 139 condors reconstruct exactly** (median absolute
error 0.000000); the 18 that do not are almost all missing `net_delta`. So the
composite can be decomposed after all.

## 2. Sixty percent of the weight is on components that point the wrong way

Component rank IC against return, iron condors:

| component | weight | rank IC | p |
|---|---|---|---|
| **pop** | **0.30** | **−0.3115** | **0.001** |
| credit_to_width | 0.20 | −0.1051 | 0.218 |
| liquidity | 0.10 | −0.1919 | 0.035 |
| iv_rank | 0.12 | −0.1079 | 0.239 |
| delta_neutral | 0.15 | +0.1632 | 0.068 |
| **theta** | **0.08** | **+0.3925** | **<0.001** |
| **spread** | **0.05** | **+0.5331** | **<0.001** |

**The largest weight is the most anti-predictive component, and the two most
predictive components carry the two smallest weights.** 0.60 of the weight sits
on negative-IC components; 0.13 on the two that work. That is a sufficient
explanation for the composite's −0.0965 — it is not noise, it is a weighting
close to reversed.

For contrast, the same decomposition on the verticals, where the composite is
mildly *positive*:

| component | weight | rank IC | p |
|---|---|---|---|
| credit_to_width | 0.20 | +0.2315 | <0.001 |
| return_on_risk | 0.10 | +0.2316 | <0.001 |
| theta | 0.08 | +0.1270 | 0.047 |
| pop | 0.25 | −0.0594 | 0.354 |
| iv_rank | 0.15 | −0.1187 | 0.064 |
| liquidity | 0.10 | −0.1395 | 0.029 |

The verticals put 0.30 on their two best components. Same repo, same weights
file, opposite outcome — which is why one composite ranks and the other does not.

Note `pop` and `liquidity` are negative in *both* structures, and `theta` is
positive in both.

## 3. The mechanism: high PoP means a tiny credit

For a condor, `pop_score = 1 − (|short_put_delta| + |short_call_delta|)`. Pushing
the short strikes further out raises it — and collects less.

`spearman(pop_score, net_credit) = **−0.7197**, p < 0.0001.`

| pop quartile | n | median credit | median width | median c/w | median return | win |
|---|---|---|---|---|---|---|
| Q1 (lowest pop) | 31 | $11.34 | 31.0 | 0.374 | **+15.5%** | **74.2%** |
| Q2 | 30 | $8.56 | 20.0 | 0.365 | −4.0% | 30.0% |
| Q3 | 30 | $6.12 | 17.5 | 0.360 | +9.8% | 56.7% |
| Q4 (highest pop) | 30 | **$1.25** | 4.0 | 0.337 | **−37.8%** | **26.7%** |

The top-PoP quartile collects $1.25 against a $4 width — risking $2.75 to make
$1.25, four legs of crossing charged against that $1.25 — and loses 37.8% at the
median with a 26.7% win rate. The composite gives that quartile its single
largest weight, positively.

This is the PoP-versus-payoff trade priced wrong: a 90%-win-rate structure that
pays 1:2.2 needs a 69% win rate to break even before costs, and these are not
achieving it.

## 4. The signs are stable everywhere they were checked

(This section runs on all 139 condors and stands. A separate stability check on
the 38-row execution-truth subset, added later, turned out to be vacuous — see
§7.)

Expanding walk-forward, condors, rank IC by test window:

| component | ≥05-15 | ≥06-01 | ≥06-15 | ≥07-01 | mean |
|---|---|---|---|---|---|
| pop | −0.324 | −0.304 | −0.303 | −0.359 | **−0.322** |
| spread | +0.543 | +0.526 | +0.527 | +0.617 | **+0.553** |
| theta | +0.428 | +0.453 | +0.466 | +0.439 | **+0.447** |
| liquidity | −0.212 | −0.284 | −0.272 | −0.112 | −0.220 |
| credit_to_width | −0.133 | −0.157 | −0.142 | −0.262 | −0.174 |

**No sign flips in any window.** And on the execution-truth subset, restated at
the crossed credit, `pop` goes from −0.2973 (p 0.07) to **−0.3437 (p 0.03)** —
it gets worse, not better, once you charge what the trade actually costs.

Three components clear Bonferroni at 7 comparisons (α = 0.007): pop, spread and
theta.

## 5. Three caveats, one of them serious

**`spread_score`'s +0.55 is substantially mechanical, and should not be acted
on.** A wide market's mid credit is fictitious: `pnl_pct` is computed against
that inflated entry credit while the close cost is real, so wide-spread condors
record worse returns partly by construction. Restating at the cross does not
rescue the finding — it charges slip explicitly, which is circular for exactly
this component. Read `spread_score` here as "the mid was trustworthy", not as a
tradeable signal.

**`theta_score` was a within-chain rank when these rows were scored, and is not
any more.** It became an absolute mapping today (`b4a376f`). The +0.45 measured
here describes the old quantity. Its forward behaviour has to be re-measured.

**n = 121–139, one structure, about three months.** `pop` at p = 0.001 with a
−0.72 mechanism correlation and four stable windows is the strongest thing in
this review, but it is still one cohort in one regime.

## 6. What follows

The honest reading is that the **condor weighting is the best-evidenced defect
found in this entire scoring review** — better evidenced than the adjustment
stack, because the mechanism is identified, the sign is stable in every window,
it survives the cost restatement, and it clears multiple-comparison correction.

Two things are worth separating:

- **Re-weighting `pop` down for condors** is a calibration decision, and the
  evidence supports it more strongly than anything else measured here. It is
  still the user's call and belongs in `docs/CALIBRATION_JOURNAL.md`, and it
  should not borrow `spread_score`'s inflated +0.55 as justification.
- **Nothing here licenses a `spread_score` re-weight.** See §5.

The cheapest honest change is not a re-weight at all — but not for the reason
first given here. See §7.

---

## 7. Correction: the gate refuses nothing, the *ordering* is what helps

§6 originally claimed the top-PoP condors are "exactly what the
`candidate_verdict` friction gate already refuses on cost grounds". **That was
an assertion, and testing it showed it is false.**

Applying the real gate (`DEFAULT_MAX_FRICTION = 0.25`, plus the credit-vanishes
check) to the 38 condors that carry `entry_price_mid` and `entry_price_cross`:

```
refused: 0 of 38
```

None of them trips it. The claim was wrong.

**What is true is better, and is the actual reason the shipped ordering helps.**
Condor rows carry no `ev_per_contract` — `find_iron_condors` never emits one and
`enrich_iron_condors` copies only `ev_score`. So in `candidate_verdict.rank`,
`_ev` returns `-inf` for every condor, they all tie on EV, and the sort falls
through to its next key: **`-round_trip_pct`, cheapest-to-trade first.**

That key is a far better predictor of a condor's outcome than the composite:

| ranked by | vs net return | vs mid return |
|---|---|---|
| `quality_score` (old ordering) | −0.1752 (p 0.29) | −0.1877 (p 0.26) |
| **`-round_trip`** (shipped ordering) | +0.6503 (p 0.00001) | **+0.5779 (p 0.00015)** |

And `spearman(pop_score, round_trip) = +0.5991, p = 0.0001` — the high-PoP
trades are the high-friction trades, which is the bridge between §3 and this.

Selecting the top K, on the **mid** return so no slip is subtracted:

| K | by `quality_score` | by `-round_trip` |
|---|---|---|
| 5 | +29.4% median, 80% win | +32.3% median, **100%** win |
| 10 | **−7.7%** median, 40% win | +24.5% median, **100%** win |
| 15 | **−6.9%** median, 47% win | +32.0% median, **100%** win |

(Whole cohort: +12.3% median, 63% win.)

### Why this is reported on the mid return

`pnl_net` subtracts the same slip that `round_trip` measures, so ranking one
against the other is partly circular — the identical objection raised against
`spread_score` in §5, and it applies here just as much. The effect survives on
the mid return, which subtracts no slip, so it is not that circularity.

**One confound does remain and cannot be removed with this data:** a wide
market's quoted mid is not achievable, so the recorded entry credit is inflated
and the recorded mid return is depressed. Wide-spread condors look worse partly
by construction on *both* bases. That is arguably the real effect rather than an
artifact — an unfillable mid is a worse trade — but it cannot be separated here.

### Also correcting: the earlier stability check was vacuous

The walk-forward table in §4 of an earlier draft showed `-round_trip` stable
across windows. The 38 execution-truth condors span **2026-06-10 to 2026-07-21,
12 distinct dates** — every "window" contained essentially the same rows
(n = 38, 38, 34). That check demonstrated nothing and should not be cited. The
§4 *component* walk-forward, which runs on all 139 condors, is unaffected.

### The consequence for future work

**Do not add `ev_per_contract` to condor rows without measuring it first.** EV
sorts *before* `-round_trip` in `rank`. Supplying an unvalidated EV would
override a key measuring +0.58 with one that has never been checked for this
structure — and repo-wide, `ev_per_contract` prices fair value on trailing
`hv_252d` and carries 0.68% of the single-leg composite. Adding it here would
look like an improvement and could quietly undo the one that works.
