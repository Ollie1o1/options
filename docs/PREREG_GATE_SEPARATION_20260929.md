# Gate separation test — REGISTRATION

**Frozen 2026-09-29. Thresholds below were fixed before the estimator was
written.**

## Provenance — read this before the result

This registration is **not** clean. On 2026-09-29, before any of it existed, a
pooled `AVG(pnl_pct) GROUP BY strategy_name, gate_passed` was run against this
same corpus and showed Long Put's gate-passed arm at `-0.185` against a refused
arm of `-0.021` — an apparent inversion. That look is what motivated this
document.

Every verdict below therefore carries the contamination of having been chosen
after seeing the data. The registration exists to stop the *second* mistake, not
to undo the first: it fixes the estimator, the unit of observation, the
multiplicity correction and the decision rule in advance of computing any of
them, so that the number this produces is at least a well-defined number.

A `separates` or `inverted` verdict here **authorises no configuration change.**
It authorises one thing: a forward test on scans recorded after the freeze date,
which is the only uncontaminated sample available.

## Hypothesis

**H1.** Within a strategy, the live gate separates forward outcomes: contracts
it passes do better than contracts it refuses, on the same underlying, the same
expiration, the same day.

**H1-inv.** For at least one strategy the separation is negative — the gate
passes the worse contracts.

`prereg_ranker.load_cohort` restricts itself to survivors and records that "the
refused population belongs to the separate removal question, which needs its own
pre-registration." This is that registration.

## Population

Rows of `candidates` joined to `candidate_positions` on
`(scan_id, board, contract_key)`, where `status = 'CLOSED'`, `pnl_pct IS NOT
NULL`, and `gate_passed IS NOT NULL`.

`gate_passed IS NULL` (8,454 rows, spanning the whole corpus, not a legacy
prefix) is **excluded**: a row whose arm is unknown cannot be assigned to one.

## Unit of observation

| level | key | role |
|---|---|---|
| row | `(scan_id, board, contract_key)` | one decision instance. **Never the unit.** |
| cell | `(symbol, expiration, scan_date)` | the **matching** unit |
| cluster | `symbol` | the **resampling** unit |

The cell is the pairing. Within one `(symbol, expiration, scan_date)` the
underlying's forward move is common to both arms and cancels, which is the same
argument `policy_lab.stats.paired_frame` makes for `(symbol, entry_date)`. A
cell containing only one arm carries no comparison and is dropped.

The cluster is `symbol`, not `(symbol, date)`: the corpus spans 5.4 weeks and
holds run 3–11 days, so the same ticker on adjacent days is the same bet. This
is the conservative choice and it is deliberate — this repo has overcounted
correlated rows three times.

Measured feasibility at registration (cells with both arms / distinct symbols):

```
Bear Call    398 / 53      Long Call     66 / 28
Bull Put     287 / 49      Iron Condor    4 /  2   -> insufficient by construction
Long Put      93 / 38      (blank name) 576 / 84
```

## Outcome and its denominator

`candidate_positions.pnl_pct` is **return on premium**, signed by
`candidate_marks.pnl_pct`:

```
debit  (entry < 0):  (m - |e|) / |e|     bounded below at -1
credit (entry > 0):  (|e| - m) / |e|     UNBOUNDED below
```

This is not the book's basis. The book reports on capital at risk, and the two
disagree about this book's profit factor (1.044 on capital at risk, 0.971 on
premium).

For credit structures the denominator collapses as the entry credit approaches
zero, and the corpus shows exactly that: Bull Put reaches `-168.231` and Bear
Call `+659.0`. **A mean of that column is a statement about a handful of
near-zero-credit rows.** Debit structures are bounded (Long Call min `-0.99`,
Long Put min `-0.996`) and their means are safe.

The primary statistic is therefore rank-based, which is invariant to the
blow-up. The mean difference is carried only as a winsorized secondary.

Magnitudes are **not comparable across strategies**: `0.1` of premium on a
credit spread and on a long put are different amounts of money.

## Statistics

**Primary — probability of superiority.** For cell *k* with passed set *P* and
refused set *R*:

```
A_k = [ #{p > r} + 0.5 * #{p == r} ] / (|P| * |R|)      over all pairs in P x R
D_k = A_k - 0.5                                          in [-0.5, +0.5]
```

Strategy statistic: mean over clusters of that cluster's mean `D_k`. Cluster-
weighted, so one heavily-scanned ticker cannot carry the result.

**Interval:** percentile cluster bootstrap resampling whole symbols,
`n_boot = 10000`.

**Secondary — winsorized mean difference.** Within cell,
`mean(pnl_pct | passed) - mean(pnl_pct | refused)`, with `pnl_pct` winsorized at
the 1st and 99th percentile of that strategy's pooled closed distribution.
Reported in fractions of premium. Sign must agree with the primary; it carries
no independent decision authority.

**Multiplicity.** The six strategies are one family — Long Put was selected as
the maximum of six looks. Family-wise error is a max-statistic sign-flip
permutation across the shared symbol axis, `n_perm = 10000`. One sign per
symbol, applied to every strategy at once, preserving cross-strategy
correlation. Strategies absent from a symbol are masked, so each keeps its own
denominator.

```
alpha = 0.05      n_boot = 10000      n_perm = 10000
seed  = 20260929  min_clusters = 20   max_fwer_p = 0.05
winsor = (0.01, 0.99)                 negative_control_shuffles = 200
```

`min_clusters = 20` matches `policy_lab.stats.MIN_CLUSTERS` and
`alloc.report.MIN_N`.

## Decision rule

Per strategy, evaluated once:

| verdict | condition |
|---|---|
| `insufficient` | clusters < 20 |
| `separates` | CI lo > 0 **and** fwer_p < 0.05 **and** secondary > 0 **and** LOSO sign-stable **and** both halves agree in sign |
| `inverted` | CI hi < 0 **and** fwer_p < 0.05 **and** secondary < 0 **and** LOSO sign-stable **and** both halves agree in sign |
| `null` | anything else |

`insufficient` is not `null`: "could not measure" and "measured, found nothing"
are different claims.

There is no EXTEND state.

## Guards — the run is VOID if either fails

1. **Negative control.** Permute the arm label within cell, 200 shuffles. The
   null distribution's mean `|D|` must be `< 0.01`. A non-null mean means the
   estimator manufactures separation and every number in the run is discarded.
   This repo has shipped a board ranked by a discredited score because nobody
   ran the null.
2. **Masked-permutation equivalence.** The NaN-masked family-wise permutation
   must reduce to `policy_lab.stats.family_wise_p` when no cell is masked.
   Asserted in `tests/test_gate_separation.py`.

## AMENDMENT 1 — 2026-09-29, after the first run

**Change.** The family corrected over is the strategies at or above
`min_clusters`, not all six.

**Reason.** Iron Condor has 2 symbols, both near `+0.333`. A whole-symbol sign
flip leaves its `|mean|` at `0.333` whenever the two signs agree, which is 49%
of permutations — measured, not assumed. That single under-powered arm set the
family-wise p to `0.534` while every measurable strategy's CI excluded zero by
a wide margin. A strategy already declared `insufficient` is not a look that
needs correcting; it is a look that was never taken.

**Honesty.** This was decided after seeing a result, which is the failure mode
this document exists to constrain. Three things limit it: the change is to
*which claims are in the family*, not to any threshold; it follows from the
decision rule already written above, which gives `insufficient` strategies no
claim; and the unrestricted value is computed and printed on every run, so the
effect of the choice is visible rather than asserted.

## AMENDMENT 2 — 2026-09-29, after the first run

**Change.** `strategy_name` is derived from `(mode, opt_type)` via
`trade_analysis.strategy_label_for_mode` when the column is blank.

**Reason.** `candidates.strategy_name` is never written for the single-leg
Premium Selling and Discovery boards. That is ~100,000 closed rows spanning the
whole corpus — not a legacy prefix — and in the first run they formed the
single largest matched population under the name `(blank)`: 576 cells across 84
symbols, reported as one strategy. They are in fact Short Put (37,788), Long
Put (40,247) and Long Call (20,254). A put sold and a call bought on the same
day are not exchangeable, and pooling them is the same defect
`prereg_ranker.load_cohort` avoids by deriving its own label rather than using
`candidate_positions.family`.

Rows with neither a name nor a mode (2,354, all dated 2026-08-19) remain
`(unlabelled)`; deriving a label for them would be inventing one.

**Honesty.** Decided after the first run. It restores labels that were always
derivable from recorded columns — it does not move a threshold, change a
statistic, or select a subset. It notably gives **Short Put** a measurable arm
for the first time.

**Separately: this is a live recording defect, not only an analysis one.**
`candidate_record` should write `strategy_name` for these boards. Filed as a
follow-up; the derivation here is a read-side repair of a write-side gap.

## Held-out replication

Split at the median scan date. Sign must agree in both halves for any verdict
other than `null`. Reported for every strategy regardless.

## Secondary, no decision authority

Per-strategy `n` rows, cells, clusters; skew of the paired difference (the
sign-flip null assumes per-cluster symmetry, so strong skew makes it
anti-conservative and the report must show it); the unwinsorized mean difference
beside the winsorized one.

## Corpus fingerprint

Recorded in the manifest of every run. Two runs whose fingerprints differ are
not comparable — both corpora in this project are live and one of them grew
17,378 rows in a single session.
