# Gate Redesign Spec — for operator sign-off

Date drafted: 2026-07-31
Status: **DRAFT — nothing in this document is active until the sign-off block at the bottom is filled in.**
Covers ideas: `gate-statistic-wrong-for-skewed-returns`, `fix-gate-power-math`, `repoint-gate-credit` (stage 2), `effective-n-in-gate`, and the feeder ruling from `stop-lc-autolog-bleed`. These are one decision; per DECISIONS.md 2026-06-07 (no silent gate change) they are specified together and applied only on explicit approval.

What is already allowed without this sign-off — and is being shipped alongside this draft — is *reporting*: both statistics rendered side-by-side everywhere the gate reports, and a short-premium cohort reported next to the LC gate. The decision path in `src/phase1_checkpoint.py` remains byte-for-byte what it was until this spec is approved.

---

## 1. The problems, precisely

### 1.1 The decision statistic is wrong for the return distribution

Long-call cohort returns are floored at −100% and effectively truncated above by the take-profit; the mass in between is left-heavy with a handful of +100% TP outliers. Pearson correlation on that shape is dominated by the outliers. The two statistics the checkpoint already computes disagree **in sign** on the current cohort (2026-07-29, n=70):

| statistic | value | p |
|---|---|---|
| Pearson IC (what the gate reads) | **+0.048** | 0.690 |
| Spearman rank IC (computed, ignored) | **−0.020** | 0.873 |

Bootstrap 95% CI on Pearson: [−0.175, +0.248]. And the Pearson history decays monotonically as n grows — 0.22 @ n=4, 0.20 @ n=23, 0.11 @ n=43, 0.048 @ n=70 (`reports/checkpoint_history.tsv`) — the classic signature of an effect that was never there, propped up by early small-sample outliers. Under rank IC the honest description of the current state is **"no forward information detected"**, not "positive but below the bar".

### 1.2 The power math is incoherent and EXTEND is an infinite loop

READY requires `IC ≥ 0.08 AND p < 0.05`. By Fisher-z, proving IC = 0.08 at p < 0.05 needs n ≈ 601; at the n ≥ 50 trigger, p < 0.05 demands an observed IC ≈ 0.286 — 3.6× the stated bar (docs/VALIDATION_POWER.md, PROFITABILITY_FINDINGS §1). The 0.08 floor is decorative; the p-clause binds.

The mirror problem: STOP requires `IC < 0.03` and READY requires `p < 0.05`, so an IC drifting in the 0.03–0.08 band EXTENDs forever. At ~70 trades per 9 weeks, resolving IC ≈ 0.05 at p < 0.05 needs n ≈ 1,500 — years. The explainer says "continue gathering for 2 more weeks" but nothing enforces any deadline. Three states, one of them unbounded.

### 1.3 n is overcounted

Batch auto-logging clusters entries: 57 trades on 18 entry days, ICC +0.108, design effect ≈ 1.23 → effective n ≈ 46 against nominal 57 (PROFITABILITY_FINDINGS §3). The `n ≥ 50` trigger counts nominal trades and is therefore systematically early.

### 1.4 The experiment doesn't match the decision it authorises

The gate validates Long Call — the worst line in the book, scorer IC ≈ 0 — while the strategy real money would actually trade first is short premium (Bull Put 64.2% win / +14.5% avg over 120 closed; Bear Call best scorer IC +0.152; both entirely inside budget). Validating LC to authorise going live, then trading credit structures anyway, is a category error. The robust claim is the *family*, not the name — re-pricing on measured spreads reverses which of Bull Put / Bear Call leads (`multileg-slippage-is-a-flat-floor`).

---

## 2. Proposed design

### 2.1 Decision statistic: Spearman rank IC

The gate's decision statistic becomes **Spearman rank IC**, for both cohorts. Rationale:

- Rank IC is the industry standard for exactly this task (cross-sectional signal vs skewed forward returns).
- It is robust in *both* directions — LC returns are right-outlier-dominated, short-premium returns are left-outlier-dominated (many small wins, occasional large losses). One statistic works unchanged for both gates; there is no per-cohort statistic-shopping.
- Pearson remains computed and reported beside it, permanently. A **sign-agreement guard** applies: if Pearson and Spearman disagree in sign at decision time, the checkpoint flags the disagreement in the report and the decision uses Spearman (documented, visible, not silent).

### 2.2 Sample size: effective n

Every n-threshold in the gate is stated and evaluated in **effective-n** terms:

- Cluster trades by entry date. Estimate ICC from one-way ANOVA over entry-day clusters; design effect `DE = 1 + (mean cluster size − 1) × ICC` (ICC floored at 0); `n_eff = n / DE`.
- `n_eff` replaces nominal n in the gate trigger and in the posterior's standard error (`se = 1/sqrt(n_eff − 3)`).
- The checkpoint prints nominal n, n_eff, ICC, and DE every week.

### 2.3 Decision rule: posterior bands with a bounded EXTEND

The brittle p-value cliff is replaced by the Bayesian posterior the checkpoint already computes and prints (`posterior_ic_above`, flat prior on the Fisher-z scale) — moved from reporting-only to decision-bearing, applied to the **Spearman** IC with `n_eff`:

Let `P₊ = P(true rank IC ≥ 0.08 | data)`.

| state | entry condition | exit condition |
|---|---|---|
| GATHERING | n_eff < 50 | n_eff ≥ 50 → evaluate bands below |
| READY | n_eff ≥ 50 AND `P₊ ≥ 0.85` AND sign-agreement guard passes | terminal (unlocks Phase 3 arming checks; real money still behind `live_execution.enabled`) |
| EXTEND | n_eff ≥ 50 AND `0.15 < P₊ < 0.85` AND rank IC ≥ 0.03 | **granted at most twice, 2 weeks each.** At expiry of the second extension: READY if `P₊ ≥ 0.85`, else STOP |
| STOP | n_eff ≥ 50 AND (rank IC < 0.03 with weeks ≥ 6, OR `P₊ ≤ 0.15`, OR EXTEND allowance exhausted) | terminal (see §2.5 for what STOP means for the feeder) |

Properties worth signing off on explicitly:

- **Every state has an entry AND an exit; no state is unbounded.** The worst case from adoption is 4 more weeks of EXTEND, then a forced READY/STOP resolution.
- The posterior bar scales coherently: READY at n_eff = 50 needs observed rank IC ≈ 0.23 (vs ≈ 0.29 under the old p-clause); at n_eff = 100 ≈ 0.18; at n_eff = 200 ≈ 0.15. A strong edge fires early, a modest one needs proportionally more data, and the claim being tested is the claim stated (IC ≥ 0.08), which the old `p < 0.05` (a test of IC > 0) never was.
- `P₊ ≤ 0.15` gives STOP a fast path when the data actively argues the edge isn't there, instead of waiting out the week counter.
- The EXTEND clock starts at spec adoption, not retroactively (the current streak since 2026-07-27 does not count against the allowance).

### 2.4 The short-premium gate (stage 2 of `repoint-gate-credit`)

A second, first-class cohort: **Bull Put + Bear Call + Short Put, closed, post-window, affordable-only** (`capital_at_risk ≤ auto_log.max_capital_at_risk`). Affordable-only is deliberate and differs from the LC gate's nominal cohort: real money cannot take unaffordable trades, and since 2026-07-29 the feeder refuses them anyway — for this cohort the affordable subset *is* the population of interest.

Return basis: **per-trade return on capital at risk** (`pnl / capital_at_risk`), never raw percentage of premium — the credit-vs-collateral basis problem is exactly what made prior cross-strategy comparisons meaningless.

The short-premium gate has **two arms**, because the repo's own finding (PROFITABILITY_FINDINGS §2) is that P&L is driven by family deployment, not within-family ranking:

- **Arm A — family viability (authorises capital):** bootstrap posterior `P(true median RoR > 0)`, where per-trade RoR is computed **net of measured per-structure costs** (`src/execution_costs.py` medians, re-measured monthly as the archive grows — not the flat constant). READY-equivalent requires n_eff ≥ 50 and `P ≥ 0.85`, same bands and the same bounded-EXTEND machinery as §2.3.
- **Arm B — selection skill (decides whether the scorer ranks entries):** Spearman rank IC of `quality_score` vs RoR, reported with the same posterior treatment. Arm B failing does **not** block going live; it decides whether live trading uses scorer ranking or trades the family on structural rules alone.

Median (not mean) in Arm A so a single large contract cannot carry the line — the Long Put lesson.

### 2.5 The LC feeder ruling (`stop-lc-autolog-bleed`)

Decided on IC, not on the sizing-artifact dollars (the −$17.6k is dead; affordable LC is +$183). Recommendation:

- **Keep the LC feeder running** (affordable-only, as now enforced). Paper accrual is free, and stopping it would make the LC gate's EXTEND/STOP resolution impossible to ever revisit with data.
- **Move real-money authorisation authority to the short-premium gate** (§2.4). The LC gate keeps running under §2.3 rules until it terminates in READY or STOP on its own bounded schedule; its result decides whether LC ever joins the live book, nothing more.
- If the LC gate resolves STOP, the feeder keeps logging LC as `paper_only` research rows (out of any gate cohort) or stops entirely — operator's pick at that time; nothing in this spec pre-commits it.

### 2.6 What does NOT change

- No exit-rule changes, no marking changes (that is `docs/MARK_TRUSTWORTHINESS_SPEC.md`, separately signed).
- `live_execution.enabled` stays the hard switch; READY alone never places a trade.
- The historical checkpoint files and TSV rows are never rewritten; the TSV gains columns append-only.
- The affordable-vs-nominal dual reporting on the LC cohort continues.

---

## 3. Implementation plan (after sign-off, not before)

1. `compute_checkpoint` gains `n_eff`/ICC/DE computation and the posterior-band decision function, behind a config key `gate.version` (`1` = current behaviour, `2` = this spec). Default stays `1` until the operator flips it; the checkpoint prints what version-2 *would have decided* alongside version-1 for at least one week before the flip (shadow mode).
2. Short-premium cohort query + Arm A/Arm B computation (the reporting half ships now; the gate-status half activates with `gate.version: 2`).
3. `checkpoint_history.tsv`: append `spearman`, `p_spearman`, `n_eff`, `decision_v2` columns at the end; readers tolerate both widths.
4. EXTEND allowance persisted in the checkpoint state (counted per gate, reset never), so re-runs cannot re-grant extensions.
5. Tests: fixed synthetic cohorts pinning every band boundary (P₊ = 0.849/0.851, rank IC = 0.029/0.031, n_eff straddling 50), the sign-disagreement guard, EXTEND exhaustion → STOP, and a regression test that `gate.version: 1` output is byte-identical to today's.

## 4. Sign-off

Nothing above the line is active until this block is completed by the operator.

- [ ] Approved decision statistic: __________ (proposed: Spearman rank IC with sign-agreement guard)
- [ ] Approved bands: P₊ ≥ ____ READY / ≤ ____ STOP (proposed: 0.85 / 0.15); rank-IC floor ____ (proposed: 0.03)
- [ ] Approved EXTEND allowance: ____ extensions × ____ weeks (proposed: 2 × 2)
- [ ] Approved n threshold in effective-n terms: n_eff ≥ ____ (proposed: 50)
- [ ] Approved short-premium gate (cohort, two arms, affordable-only): yes / no / amended: __________
- [ ] Approved LC feeder ruling (§2.5): yes / no / amended: __________
- [ ] Date + initials: __________
