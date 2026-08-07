# Within-chain ranks, and what the measurement found instead — 2026-08-07

Follow-on to `docs/SCORE_AUDIT_20260807.md`. That audit asked whether any
displayed number meant something other than its label. This one asks a
narrower question — **is `quality_score` comparable between the things it is
used to compare?** — and answers it with a walk-forward measurement.

The tested change passed its gate. The measurement built to test it found
something considerably more important, reported in section 4.

Script: `scripts/measure_absolute_scores.py`. Plan:
`docs/superpowers/plans/2026-08-07-absolute-component-scores.md`.

---

## 1. The score is doing two jobs with different halves of itself

`calculate_scores` runs **once per symbol** (`_score_fetched_data` →
`enrich_and_score` → `calculate_scores`, `options_screener.py:3798`). Every
`rank_norm` call inside it therefore ranks contracts against their own chain
and nothing else. The resulting composite is then used to rank contracts
**across** tickers in the main comparison table.

Splitting the live IC-blended weights by what each component can actually
distinguish:

| | share of composite | components |
|---|---|---|
| **Ticker-level constants** — identical on every row of a chain | **49.9%** | vrp 15.6, iv_velocity 12.2, term_structure 11.2, momentum 8.2, iv_rank 2.3, catalyst 0.5 |
| **Within-chain ranks** — uniform 0→1 inside each chain by construction | **19.4%** | theta 13.3, vega_risk 5.3, ev 0.6, gamma_magnitude 0.3 |
| **Contract-level absolutes** — informative both ways | 30.7% | iv_edge 10.6, spread 6.8, pop 3.1, rr 2.7, em_realism 2.4, skew_align 2.3, … |

The ticker-level half is literal broadcast: `df["iv_trend"] = _yq_iv_trend`
assigns one string to the whole frame (`data_fetching.py:652`), and `vrp_mean`
is a single scalar from `calculate_vrp` over the ticker's history. So roughly
**half the score cannot distinguish two contracts on the same chain, and a
fifth of it cannot distinguish two tickers.**

Ledger corroboration on 908 rows with stored `theta_score`: the standard
deviation of per-ticker means is **0.079** against an overall standard
deviation of **0.179** — about 19% of the variance is between-ticker, and even
that residue exists only because top-N selection filtered which rank got
logged.

The awareness already existed in the code: volume and open interest were
deliberately converted to absolute sigmoids and commented *"cross-ticker
comparable"* (`options_screener.py:1695`). The same treatment was never
extended to theta, vega_risk, ev or gamma_magnitude.

Note this is a comparability defect, not an arithmetic one. Ranking within a
chain is the right thing to do when picking a strike; it is the wrong thing
to feed a cross-ticker table. One number is serving both.

## 2. Scope of what could be tested

- Only **single-leg buyer rows** use the 27-component composite at all. Multi-leg
  rows (`Bear Call` 130, `Iron Condor` 121, `Bull Put` 115) score through
  `spread_scoring` and `credit_spread_weights`; `Short Put` (109) uses the
  `premium_selling_weights` branch. The evaluable cohort is **335 closed
  Long Call / Long Put rows**, 2026-04-18 → 2026-08-05.
- `theta_decay_pressure = |theta| / max(premium, 0.01)` and
  `vega_dollar = |vega| × 100` are exactly reconstructable from stored
  `entry_theta`, `entry_vega`, `entry_price`.
- `ev` and `gamma_magnitude` are **not** reconstructable — no stored
  `ev_per_contract`, no stored `underlying`. Together 0.86% of the composite.
  They stay rank-based and are out of scope.

## 3. The tested change: absolute mappings for theta and vega_risk

Logistic in `log10` of the raw quantity — the raw quantities are right-skewed
and DTE-driven (theta pressure median 0.019 at 30–60 DTE, 0.034 at 14–30,
0.053 at 0–14, pooled p95 0.858 against a median of 0.036), so a linear sigmoid
saturates. Centre and scale are fitted on the **training fold only** and the
delta is propagated through the inverted display scale, where the additive
adjustment stack cancels because it does not depend on the components changed.

Decision rule was fixed before the run: ship at mean OOS rank IC difference
≥ −0.01, stop below.

```
n = 335 closed Long Call/Put rows, 2026-04-18 -> 2026-08-05
live weights: theta 13.26%, vega_risk 5.28%

cut           n_tr  n_te      OOS rank IC old      OOS rank IC new    delta
2026-05-27     138   197       -0.1145 (p0.11)       -0.1112 (p0.12)   0.0033
2026-06-10     167   168       -0.1182 (p0.13)       -0.1249 (p0.11)  -0.0066
2026-06-18     206   129       -0.1224 (p0.17)       -0.1089 (p0.22)   0.0135
2026-07-07     226   109       -0.1778 (p0.06)       -0.1697 (p0.08)   0.0081
2026-07-16     277    58       -0.2622 (p0.05)       -0.2578 (p0.05)   0.0045

mean OOS rank IC:  old -0.1590   new -0.1545   difference +0.0045
rank correlation between the two orderings: 0.9299
```

**Passes**, four folds of five improving, ~7% of the ordering moving. As with
the denominator fix, this is not a returns result — +0.0045 is far inside an IC
standard error of roughly 0.07–0.13 at these fold sizes, and the folds are
nested and overlapping rather than independent. It is justified on semantics:
the score stops depending on what else happened to be fetched alongside it.

## 4. What the measurement actually found

Every fold is negative, and it steepens in later windows. The shipped
`quality_score` ranks Long Call / Long Put outcomes **inversely**, mean OOS
rank IC **−0.159**.

Decomposing the shipped score into the part the weights produce and the part
the ~20 hand-set constants add — inverting `_cross_section_normalize`
(`raw = 0.28 + 0.54 × stored^(1/0.65)`, verified: stored scores span exactly
[0.0, 0.9999] and invert into [0.2800, 0.8199]) and subtracting the recomposed
27-component composite:

| test window | n | shipped | 27-component composite | additive stack |
|---|---|---|---|---|
| ≥ 2026-05-27 | 197 | −0.1145 | −0.0186 | −0.0887 |
| ≥ 2026-06-10 | 168 | −0.1182 | −0.0155 | −0.0808 |
| ≥ 2026-06-18 | 129 | −0.1224 | −0.0300 | −0.0806 |
| ≥ 2026-07-07 | 109 | −0.1778 | +0.0227 | −0.1548 |
| ≥ 2026-07-16 | 58 | −0.2622 | −0.0715 | −0.1940 |
| **full cohort** | **335** | **−0.0947** (p 0.084) | **+0.0044** (p 0.936) | **−0.0964** (p 0.078) |

**The composite is flat. The adjustment stack carries the entire negative
signal, in all five windows.**

It also dominates the variance. The additive residual has sd **0.1004**
against a composite sd of **0.0645** — the hand-set constants move the shipped
score 1.6× more than all 27 weighted components combined. That is why the
composite correlates with the shipped score at only Pearson 0.330 /
Spearman 0.327.

**This is robust to not knowing the historical weights.** The IC blend is
recomputed at runtime from `ic_weights_cache.json` and has drifted across the
sample, so the composite reconstruction uses today's weights rather than the
ones in force at entry. But every component score is bounded in [0, 1] and the
weights sum to 1, so reweighting cannot move the composite far. Rebuilding it
under four very different weightings:

| weighting | composite sd | residual sd |
|---|---|---|
| live IC-blended | 0.0645 | **0.1004** |
| raw config | 0.0679 | **0.1016** |
| uniform | 0.0490 | **0.0995** |
| theta-heavy (the pre-fix 24.3% artifact) | 0.0661 | **0.1024** |

The residual is ~0.10 under all of them. Weight drift cannot account for it.
The per-row attribution is softer — the maximum per-row spread across those
four weightings averages 0.060 — so treat the *aggregate* finding as solid and
individual-row attribution as indicative.

### What this does and does not say

- It **corroborates and extends** the settled LC finding. `docs/TRUST_AUDIT_20260803.md`
  ruled the long-call gate STOP at Spearman −0.132 on n = 92. This is the same
  sign at n = 335, now measured on the full shipped score and localised to the
  adjustment stack. It is not a new contradiction and does not reopen that gate.
- The cohort is the one already known to be the worst thing the screener does
  (long premium bleeds while the credit lines profit), and ≤22 DTE rows are
  force-closed early, which biases the IC. Nothing here generalises to the
  credit strategies, which never touch this composite.
- n = 335 with nested folds, p ≈ 0.08 on the full cohort. This is evidence, not
  proof.
- **It does not license retuning the constants by taste.** The correct
  instrument already shipped: `score_adjustments` (schema 20) records which
  conditions fired, and holds 0 of 947 rows today because it landed this
  morning. What this measurement changes is the priority of that data, and it
  supplies a prior about the sign.

## 5. Status

- Section 3's change: **gate passed, not yet implemented.** Tasks 3–5 of the
  plan are unstarted.
- Known and unchanged: the double-count where all five `risk_flag_count` flags
  also fire as additive penalties; earnings reaching the score from five places;
  the ~50% of the composite that is ticker-level constant.
- `ev` and `gamma_magnitude` remain within-chain ranks.
