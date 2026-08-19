# Pre-registered ranker test — design

Date drafted: 2026-08-19
Status: **DRAFT, awaiting operator review.** No implementation yet.
Scope: sub-project 3 of the ranker-validation sequence. Builds the test and
freezes its terms. It does **not** run the test, and must not until n\* or the
deadline is reached.

Depends on `docs/CANDIDATE_RECORD_SPEC.md` (PR #49),
`docs/CANDIDATE_FORWARD_MARKS_SPEC.md` (PR #51), and PR #50.

---

## 1. Why this is written now

Pre-registration is worthless after the fact, and it is genuinely clean right
now: `candidate_positions` holds **782 rows, all OPEN, 0 closed, 0 with a
`pnl_pct`**. There is nothing to peek at. Every term below is fixed before any
outcome exists, which is the only thing that makes the eventual result mean
anything.

## 2. What randomisation changed

PR #50 made entry selection a random draw among gate survivors. That **removed**
one question and **created** a better one.

Removed: "does `rank_pos` predict outcome" is now null by construction — it
would test whether a shuffle predicts returns.

Created: because entries are an unbiased draw across the whole feature
spectrum, **any** candidate ordering can be tested retrospectively on data no
selection rule shaped. A cohort selected by rule X can never test rule X, which
is why the 1,005-row ledger could never answer this and would still have failed
to under any deterministic replacement.

## 3. The primary hypothesis

> **H1.** Among gate survivors, `ev_net` predicts `pnl_pct`.

Population: `candidate_positions` joined to `candidates` where
`gate_passed = 1` and `status = 'CLOSED'` and `pnl_pct IS NOT NULL`.

Statistic: within each cell, both `ev_net` and `pnl_pct` are rank-transformed
and demeaned; the pooled correlation of those demeaned ranks is the
**rank IC**.

The cell key is `(candidate_positions.entry_date, strategy)`, where `strategy`
is `trade_analysis.strategy_label_for_mode(candidates.mode, candidates.opt_type)`
— i.e. `Long Call`, `Long Put`, and so on, falling back to
`candidates.strategy_name` when the row carries one. Deliberately **not**
`candidate_positions.family`: that collapses Long Call and Long Put into
`long_option`, and a call and a put on the same day are not exchangeable.
`entry_date` rather than a date parsed out of `scan_id`, because the position's
own opening date is a recorded fact and the scan_id is a composite key.

The within-cell demeaning is the pairing, and it is not cosmetic. On the
existing book the carry feature showed a whole-book Spearman of **+0.104
(p=0.002)** that reversed inside strategies — Iron Condor **−0.282**, Long Put
−0.070. That was strategy mix, not signal. Pooling without cells reproduces
exactly that artifact. Demeaning by cell also absorbs market-wide daily moves,
so day-level correlation does not have to be modelled separately.

Interval: **cluster bootstrap resampling `contract_key`**, 10,000 resamples,
percentile 95% CI. The same contract recorded on five scans is one piece of
information, not five.

`alpha = 0.05`, two-sided.

## 4. A bar that cannot be incoherent

The LC gate failed on precisely this. Its rule was `IC >= 0.08 AND p < 0.05`,
and by Fisher-z proving `IC = 0.08` at `p < 0.05` needs n ≈ 601 — so at the
n ≥ 50 trigger the p-clause bound and the 0.08 floor was decorative.

The fix is to make the conditions agree by construction:

- **n\* is powered to detect `IC = 0.08` at 80% power, `alpha = 0.05`.**
- **PASS = the 95% CI lower bound exceeds 0.** One condition.

At an n powered for 0.08, clearing zero implies an estimate in that
neighbourhood anyway. Nothing is hidden and nothing silently binds.

0.08 is not invented here — it is the meaningfulness threshold the v2 gate
already uses.

## 5. n\*, and the risk that it is unreachable

Computed from cluster structure alone, no outcomes touched. Fisher-z at 80%
power, `alpha = 0.05`:

| target IC | effective observations |
|---|---|
| 0.08 | **1,224** |
| 0.10 | 783 |
| 0.15 | 347 |

Nominal n is `effective x design effect`, with `DE = 1 + (m - 1) * ICC`, `m` the
mean positions per `contract_key` and ICC taken from the book's measured
**0.08–0.11**:

| m | DE at ICC 0.11 | nominal n for IC 0.08 |
|---|---|---|
| 2 (today) | 1.11 | 1,359 |
| 5 | 1.44 | 1,763 |
| 10 | 1.99 | 2,436 |
| 20 | 3.09 | 3,783 |
| 30 | 4.19 | 5,129 |

**`m` grows with time.** A contract recorded on every scan for a month has
m = 20+. So the honest statement is that n\* is likely in the **2,000–5,000
nominal closed positions** range, against an accrual of roughly 356 survivor
positions per scan day that then take weeks to close.

**This may not be reachable in one quarter, and that is a legitimate result.**
It is stated here, before the data exists, precisely so that discovering it
later cannot become a reason to move the bar. `scripts/prereg_ranker_power.py`
computes n\* from the *observed* clustering rather than this table, and writes
it into the frozen registration.

## 6. The single look

`n* ` and a calendar deadline `D` are both written into
`docs/PREREG_RANKER_TEST.md` when the power script runs, and are immutable
thereafter.

- Before `n*` and before `D`: the test script **refuses to compute anything**
  and reports progress only.
- At `n*` or `D`, whichever comes first: **one** look, one test.
- The first run writes a result stamp into the registration. A second
  invocation **reports the stored result rather than recomputing**. One look
  means one look, enforced in code rather than in discipline.

Three terminating outcomes, no fourth:

| outcome | meaning | consequence |
|---|---|---|
| **PASS** | CI lower bound > 0 | ranking may be *proposed* again, as its own change |
| **FAIL** | CI includes 0 | refuse-don't-rank stands; entries stay random |
| **INVERTED** | CI entirely **below** 0 | treated as FAIL here; see below |
| **UNDERPOWERED** | `n < n*` at `D` | treated as FAIL |

There is deliberately no EXTEND state. That is the trap that let the LC gate
run forever.

**INVERTED is reported separately rather than folded into FAIL**, because it is
not the same finding. An `ev_net` that reliably predicts outcome *backwards* is
real information — and this book has seen exactly that shape before, in
`quality_score`, whose top quintile was the worst cell in the ledger. It does
not PASS: reversing a sign on the strength of one look is how overfitting
starts. It motivates a **new** pre-registration testing the reversed ordering,
which would need its own data.

**PASS does not authorise real money.** That remains behind its own gate. PASS
authorises a proposal to re-introduce ranking, which would itself need
validating.

## 7. Guards, all pre-committed

- **Negative control.** `pnl_pct` shuffled within cells must return a null
  result. This tests the test: a pipeline bug that manufactures signal is
  otherwise indistinguishable from a finding, and this repo has shipped a
  board ranked by a discredited score because nobody ran the null.
- **Sign consistency** across both halves of the sample, split at the **median
  `entry_date`** — the standard already applied to the condor universe rule and
  the tail study. Reported, not gating: a sign that flips between halves is
  recorded beside the result so a PASS resting on one half is visible.
- **Effective n is reported, never nominal.** The book's own history is that
  batch entries gave ICC 0.08–0.11 and a design effect of 1.23–1.27, so the
  `n >= 50` trigger was systematically early.
- **Secondary features carry no decision authority.** `quality_score`, carry
  (`|theta|/premium`) and `delta` are reported beside the primary. A secondary
  result may motivate a *new* pre-registration; it can never move this one.
  `quality_score` is already measured anti-predictive (rho −0.032, p=0.345,
  n=881) and carry is documented as "an ordering, not a ranking", so neither
  deserves to consume alpha.

## 8. Deliverables

| file | responsibility |
|---|---|
| `src/prereg_ranker.py` | The analysis: cell demeaning, rank IC, cluster bootstrap, design effect, negative control. Pure functions over a frame — no I/O, testable offline against synthetic data with known answers. |
| `scripts/prereg_ranker_power.py` | Computes n\* from observed clustering and writes the frozen registration. Run once. |
| `scripts/prereg_ranker_test.py` | The single look. Refuses to run early; refuses to recompute. |
| `docs/PREREG_RANKER_TEST.md` | The registration itself — hypothesis, n\*, deadline, decision rule, and later the stamped result. Committed, immutable. |

`src/prereg_ranker.py` holds no thresholds. They live in the registration and
are read, the same discipline `candidate_marks` applies to exit rules.

## 9. Testing

Against synthetic frames with known answers, so the statistics are verified
rather than trusted.

1. A frame with a **planted** rank IC of 0.30 recovers ≈0.30 within tolerance.
2. A frame with **no** relationship returns a CI that includes 0.
3. **The Simpson's-paradox case**: two strategies with opposite within-strategy
   signs and a strong pooled correlation. Cell demeaning must recover the
   within-strategy sign, and a deliberately un-demeaned comparison must show
   the artifact — proving the pairing does the work claimed for it.
4. The cluster bootstrap CI is **wider** than a naive i.i.d. CI on the same
   clustered data, and the two coincide when cluster size is 1.
5. Design effect matches `1 + (m - 1) * ICC` on a frame with a known ICC.
6. The negative control returns null on data with a planted effect — the
   shuffle must destroy it.
7. The test script **refuses to compute** below n\* and before D.
8. A second invocation returns the stored result and does **not** recompute,
   asserted by mutating the underlying data between runs and getting the
   original answer back.
9. `UNDERPOWERED` is reported when `n < n*` at `D`, and is treated as FAIL.
10. No test reads the real `candidates.db` or the real registration.

## 10. Out of scope

- **Running the test.** Not until n\* or D.
- **Any change to a gate, ranker or entry path.** PASS authorises a proposal,
  nothing more.
- **The removal question** — do the gates refuse losers? It is now testable,
  since refusals are marked, and it deserves its own pre-registration rather
  than being smuggled in as a secondary here.
- **Real-money authorisation.** Separate gate, unchanged.
