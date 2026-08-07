# Calibration Journal

A timestamped log of every `--calibrate --apply` event against `config.json`.
Each entry captures the IC snapshot, the shrinkage, the actual recommended
weight changes, and the rationale. Read top-down to see how the model has
evolved over time.

---

## 2026-08-07 — IC blend denominator: BUG FIX, weights move without a re-fit

**Context:** score/EV audit, `docs/SCORE_AUDIT_20260807.md` item 1. No
`--calibrate --apply` was run; `config.json` `composite_weights` are untouched.
What changed is how `load_ic_adjusted_weights` combines them with the cached
component ICs, so the *effective* weights move and this belongs in the journal.

**Defect:** the blend divided each surviving component's IC by the sum of IC
over survivors. With one survivor that ratio is 1.0 by construction, so the
component took the whole 0.30 reallocation regardless of its IC's size. Live on
2026-08-07 only `theta` cleared p < 0.10 (IC +0.082, p 0.021), so:

- `theta` ran at **0.3138** against a base weight of 0.0197 — **24.3%** of the
  composite, 16× its configured share.
- Doubling theta's measured IC moved its weight by **+0.0000**.
- An unrelated component crossing p = 0.10 stripped **26%** of it.

**Change:** denominator now spans all candidate components, not just eligible
ones. The p-gate decides eligibility; IC magnitudes decide the split.

**Effective weight change (config unchanged):**

| | before | after |
|---|---|---|
| theta share of composite | 24.3% | 13.3% |
| top three | theta 24.3%, vrp 13.6%, iv_velocity 10.6% | vrp 15.6%, theta 13.3%, iv_velocity 12.2% |

**OOS read:** expanding-window walk-forward over the closed ledger, 5 folds.
Mean OOS IC **+0.0383 (old) vs +0.0343 (new)**, difference −0.0040 — inside
noise on nested test sets of 173–564 trades. In-sample the old rule scores
higher (+0.094 vs +0.081), which is circular: those weights were fitted to
maximise IC on that ledger. Rank correlation old vs new: **0.933**.

**Decision: SHIP.** Not on a return claim — the OOS evidence cannot separate the
two rules, and that is stated. It ships because the old weight was an artifact
of the survivor count rather than of the evidence, and could swing by a quarter
of its size when an unrelated factor crossed a p-value. Returns unchanged as far
as this ledger can measure; semantics now defensible.

**Standing caveat, unchanged by this:** the ICs being blended come from a sample
whose own composite verdict is "NO SIGNIFICANT EDGE (IC −0.03, p 0.433)" and
whose top score quintile is the worst performer (avg −5.2%, win rate 41.2%).
Fitting weights on that sample is fitting noise under either denominator. Rank
by net-of-cost EV; `quality_score` breaks ties.

---

## 2026-05-29 — Long-Call v1 candidate calibration: EVALUATED, NOT ACTIVATED

**Context:** Phase 1 real-money-readiness work (see
`docs/superpowers/specs/2026-05-27-real-money-readiness-design.md`). Built a
true out-of-sample walk-forward harness (`src/walk_forward.py`) and an LC-only
weight calibration as a candidate.

**Command:** `python -m src.backtest_optimizer --strategy long_call --save
--mask-zero-variance --no-cv` (synthetic yfinance backtest, 25 tickers × 68
trades). Optimized weights extracted to `configs/weights/long_call_v1.json`,
then `config.json` restored to baseline.

**Walk-forward OOS read (real paper ledger, not synthetic):**
- Harness: `python -m src.walk_forward --strategy "Long Call"` over 94 closed
  LC trades, 5 folds (train=44, test=10, step=10).
- **Pooled OOS IC: +0.102 (p=0.480)**; fold IC mean +0.233; 4/5 folds positive;
  95% CI [-0.054, +0.536].
- This is the first *uncontaminated* IC for the system — the prior +0.023
  (p=0.73) was in-sample. Directionally encouraging, but underpowered: only 50
  out-of-sample observations, p far from significant.

**Decision: STAY ON BASELINE. Defer activation.**
- `config.json` → `auto_log.weight_profile` remains `null` (baseline weights,
  checksum unchanged).
- `configs/weights/long_call_v1.json` is written as a **candidate only** — NOT
  active. It exists so we can A/B it later without re-running calibration.
- Rationale: the optimized weights are fit on a tiny in-sample set and the OOS
  signal can't yet distinguish them from baseline (p=0.48). Activating now would
  risk overfitting and would contaminate the forward cohort with an unproven
  config. Baseline already produced the +0.10 OOS read.

**Revisit when:** the Phase 1 forward cohort (post-2026-05-27 LC trades,
`paper_only=0`) reaches ≥50 trades with a significant OOS read. At that point,
re-run walk-forward and A/B baseline vs `long_call_v1` before activating either.

**No revert needed** — baseline `config.json` was never changed (only snapshotted
and restored). Optimizer auto-backup: `config.bak.20260529-122248.json`.

---

## 2026-05-20 (evening) — multi-method ensemble calibration (long-call only, n=89)

**Command:** ad-hoc multi-method ensemble (not via
`src.backtester --calibrate --apply` — see "Methodology" below).
**Backup file:** `config.bak.20260520-ensemble-165352.json`
**Trigger:** Morning's surgical tuning (`config.bak.20260520-145310.json`) was
fit on n=33 post-2026-05-11 trades and boosted `pop` / `theta` while cutting
`spread`. On the larger n=89 long-call-only subset (filtered to match the
current `auto_log_skip_long_puts` flow), three independent methods agree those
moves were directionally wrong for long-calls. This calibration replaces the
surgical tuning with an ensemble-derived weight set whose in-sample composite
IC is +0.359 (p=0.0006) on the 89-trade long-call cohort, vs +0.269 for the
morning's surgical weights.

### Methodology — four-method ensemble

Per-factor verdict requires agreement across multiple independent statistical
methods, not just a single IC:

1. **Bootstrap IC, 2000 resamples** of the 89-trade long-call cohort. 95% CI
   per factor; `STRONG+` only if entire CI > 0.
2. **Walk-forward expanding-window IC** — three folds (train ≥ 30, test = 15).
   Looks for both meaningful test IC magnitude AND ≥ 50% train/test sign
   agreement.
3. **Ridge regression with 5-fold CV** for α selection, plus a 1000-resample
   bootstrap on the coefficients. Reports both β and `P(same sign)` under
   bootstrap.
4. **Leave-one-ticker-out** check — drops each of the top 8 tickers in turn,
   confirms no sign flip on the lead factors.

A factor earns `+++` only when all three statistical methods (bootstrap,
walk-forward, ridge) agree positive. `---` requires all three negative.

The synthetic backtester (`src.backtest_optimizer`) was also run at scale —
105 tickers × 5y × DE 300 trials × mask-zero-variance — with 7,471 synthetic
long-call trades. Its per-factor IC table did *not* agree with the real
paper-trade IC: synthetic top factors topped out at ±0.06 IC, vs ±0.26 on real
long-calls. Conclusion: the synthetic backtester's HV-as-IV proxy cannot
generate the long-call signal at any sample size. The ensemble is derived
from real paper trades only; synth was used as a non-validating cross-check.

### Weight changes (top deltas vs morning's surgical)

| Factor | Surgical → Ensemble | Δ | Why |
|---|---|---|---|
| `spread` | 0.0195 → **0.0768** | +0.057 | +++ across all 3 methods |
| `momentum` | 0.0378 → **0.0929** | +0.055 | +++ (validates mode-flip in `src.backtest_optimizer`) |
| `pop` | 0.0878 → **0.0354** | −0.052 | Undoes morning's +0.056 (n=33 was misleading) |
| `iv_velocity` | 0.0885 → **0.1373** | +0.049 | +++ across all 3 methods |
| `vega_risk` | 0.1062 → **0.0595** | −0.047 | Only `+` (bootstrap 85%) — not robust enough |
| `vrp` | 0.1366 → **0.1755** | +0.039 | Strongest +++ |
| `iv_rank` | 0.0646 → **0.0260** | −0.039 | Walk-forward sign-flips OOS |
| `iv_edge` | 0.0864 → **0.1195** | +0.033 | +++ |
| `theta` | 0.0488 → **0.0197** | −0.029 | Undoes morning's +0.034 |
| `em_realism` | 0.0000 → **0.0275** | +0.028 | Bootstrap STRONG+ on previously-zero factor |

### In-sample composite IC (validation on 89 long-calls)

| Weight set | Composite IC | p-value | Q1 (top 22) win | Q4 (bot 23) win | Skip-bot-25% book |
|---|---:|---:|---:|---:|---:|
| Morning surgical (replaced) | +0.269 | 0.011 | 59.1% | 26.1% | $13,455 |
| Pre-surgical (2026-05-11 cal) | +0.298 | 0.005 | — | — | — |
| **Ensemble (applied)** | **+0.359** | **0.0006** | **68.2%** | **13.0%** | **$16,963** |

The ensemble's bottom-quartile win rate of 13% is the headline diagnostic.
Skipping the ensemble's bot-25% on the 89-trade book would have improved P&L
from $13,089 to $16,963 — a +30% improvement in-sample.

### Caveats

- **In-sample number.** Out-of-sample expectation is roughly half: +10 to +18%
  improvement to total P&L, mostly via bottom-quartile filtering.
- **32-day window.** Could be regime-specific. Re-run methodology every ~30
  new closed long-calls.
- **Overrides the morning's 2026-05-20 surgical tuning's validation plan** (which
  asked for 2026-05-25 + ~30 new trades before next calibration). The override
  is justified because the surgical tuning was n=33 and is contradicted by 3
  independent methods on n=89.
- **Replaces, not stacks.** The morning surgical and the ensemble both edit
  the same `composite_weights` block, so they're mutually exclusive. The
  surgical is preserved in `config.bak.20260520-145310.json`.

### Validation plan

- Track the next ~30 closed long-call trades under these weights.
- Re-run `src.calibration_ensemble` (see below) after ~30 new closed trades to
  check stability of the top factors.
- If bot-quartile win rate stays < 25% OOS, the ensemble is validated.
- If bot-quartile win rate jumps to > 40% OOS, the ensemble is overfit and
  should be reverted via the procedure below.

### Revert procedure

```bash
cp config.bak.20260520-ensemble-165352.json config.json
# (or to the morning surgical):
# cp config.bak.20260520-145310.json config.json
rm -f ic_weights_cache.json
```

---

## 2026-05-11 — first IC-driven directional calibration

**Command:** `python -m src.backtester --calibrate --apply`
**Backup file:** `config.bak.20260511-111722.json`
**Trigger:** 182 closed paper trades (well past the 25-trade hard floor;
shrinkage λ = 182 / (182 + 60) = 0.752, IC weighted 75% vs current 25%).
**Scope:** `composite_weights` only (single-leg / directional bucket). Spread
weights and iron-condor weights left untouched — they live in separate
sections and are gated by their own 30-closed thresholds (Bear Call needs 10
more, IC needs 24 more).

### IC snapshot (top signals)

| Component | IC | Direction |
|---|---|---|
| vrp | +0.212 | strong positive |
| term_structure | +0.197 | strong positive |
| vega_risk | +0.168 | strong positive |
| gamma_theta | -0.159 | strong negative |
| pop | -0.143 | strong negative (counterintuitive — see notes) |
| spread | +0.142 | positive |
| iv_velocity | +0.130 | positive |
| iv_edge | +0.114 | positive |
| theta | -0.110 | negative |

### Biggest weight shifts (Δ ≥ 0.05)

| Component | Old | New | Δ |
|---|---|---|---|
| pop | 0.1300 | 0.0322 | **-0.098** |
| vrp | 0.0500 | 0.1401 | **+0.090** |
| term_structure | 0.0400 | 0.1287 | **+0.089** |
| vega_risk | 0.0300 | 0.1088 | **+0.079** |
| spread | 0.0100 | 0.0881 | **+0.078** |
| momentum | 0.1000 | 0.0387 | -0.061 |
| liquidity | 0.0800 | 0.0198 | -0.060 |
| ev | 0.0700 | 0.0174 | -0.053 |
| rr | 0.1000 | 0.0491 | -0.051 |

### Interpretation

The book's first IC-driven calibration was a vol-edge correction. The
priors (set by hand-tuned heuristics) over-weighted directional/PoP signals
(pop, momentum, rr, liquidity) and under-weighted volatility-surface
signals (vrp, term_structure, vega_risk, iv_velocity, spread). 182 closed
trades pulled the weights toward vol-edge dominance, which matches the
canonical options-trading principle: trade the surface, not the direction.

The negative IC on `pop` (-0.143) is the most striking finding. High-PoP
trades systematically underperform because the credit is small and tail
risk eats the realized P&L. The calibrator correctly identified this and
slashed `pop` from 0.13 → 0.03.

### Known issues flagged but NOT addressed by this calibration

**Six variance-zero scorers** (gamma_pin, max_pain, oi_change,
option_rvol, pcr, sentiment) — 151 rows but constant scores. The score
functions are returning a sentinel for every trade. They now sit at 0.000
weight (correct given they carry no information), but the underlying
scorer bugs should be investigated separately. Each is potentially a
useful signal once the scorer returns real values.

### Resulting weight table (full, sorted desc)

```
vrp                  0.1401
term_structure       0.1287
vega_risk            0.1088
iv_velocity          0.0907
iv_edge              0.0885
spread               0.0881
iv_rank              0.0662
rr                   0.0491
momentum             0.0387
skew_align           0.0349
pop                  0.0322
trader_pref          0.0293
iv_mispricing        0.0265
liquidity            0.0198
ev                   0.0174
theta                0.0149
catalyst             0.0136
gamma_magnitude      0.0074
gex                  0.0025
max_pain             0.0025
em_realism           0.0000
gamma_theta          0.0000
pcr                  0.0000
oi_change            0.0000
sentiment            0.0000
option_rvol          0.0000
gamma_pin            0.0000
```

Total budget: 0.9999 (≈ 1.000, conserved).

### Validation plan (post-apply)

- **Watch first 2 weeks** of auto-logged trades. New trades will be scored
  with the new weights; track win-rate and avg P&L by quality_score
  quintile. If the top quintile still under-performs vs lower quintiles,
  the recalibration didn't land — revert to backup.
- **Re-snapshot weekly** with `scripts/calibrate_snapshot.sh` to track IC
  stability. If next snapshot shows ICs collapsing toward zero, the
  current readings may be over-fit to the 2026-04 / 2026-05 regime.
- **Defer IC-bucket and spread-bucket calibration** until their
  respective sample counts clear 30. ETA: Bear Call ~2-3 weeks, Iron
  Condor late June 2026.

### Revert procedure if regression

```bash
cp config.bak.20260511-111722.json config.json
# or restore the in-DB calibration_marker file from before this run:
cat paper_trades.db.calibration_marker.json
```

---

## 2026-05-20 — surgical sign-mismatch tuning (manual, not from `--calibrate`)

**Command:** manual `composite_weights` edit (not `python -m src.backtester --calibrate --apply`)
**Backup file:** `config.bak.20260520-145310.json`
**Trigger:** Post-2026-05-11 cohort showed regression — 32 closed trades down
−$2,058 (34% win rate, avg −$64/trade) vs pre-cal cohort +$94/trade. IC of
`quality_score` vs realized return on the post-cal cohort was **−0.026 (p=0.71)**
— no signal. Quintile breakdown non-monotonic, top-half quality lost $2,616
while bottom-half made $558.

### Post-cal cohort IC by component (n=33)

Decomposed `pnl_pct` against each stored component score (sorted by IC):

| Component | IC | Pre-cal wt | Post-cal wt | Verdict |
|---|---:|---:|---:|---|
| gex | −0.377 | 0.010 | 0.003 | strong negative, tiny weight — leave |
| liquidity | −0.246 | 0.080 | 0.020 | cal already cut — ok |
| iv_mispricing | −0.235 | 0.050 | 0.026 | cal already cut — ok |
| skew_align | −0.213 | 0.020 | 0.035 | wrong-sign boost, small |
| **spread** | **−0.128** | 0.010 | **0.088** | **WRONG-SIGN, heavy weight — fix** |
| em_realism | −0.115 | 0.000 | 0.000 | n/a |
| ev | −0.078 | 0.070 | 0.017 | cal already cut |
| term_structure | −0.038 | 0.040 | 0.129 | low-IC, but matches direction in full ledger — keep |
| iv_edge | +0.085 | 0.080 | 0.088 | ok |
| momentum | +0.121 | 0.100 | 0.039 | mild signal, ok |
| vega_risk | +0.153 | 0.030 | 0.109 | cal correctly boosted |
| iv_rank | +0.198 | 0.080 | 0.066 | ok |
| iv_velocity | +0.205 | 0.050 | 0.091 | cal correctly boosted |
| trader_pref | +0.227 | 0.000 | 0.029 | cal correctly added |
| vrp | +0.228 | 0.050 | 0.140 | cal correctly boosted, strongest +IC |
| **pop** | **+0.253** | **0.130** | **0.032** | **cal CUT a positive signal — fix** |
| **theta** | **+0.265** | **0.060** | **0.015** | **cal CUT a positive signal — fix** |
| rr | +0.388 | 0.100 | 0.049 | strong positive but score std=0.004 → IC unreliable; leave |

**Sign-mismatch root cause:** 2026-05-11 calibration was fit against the full
182-trade ledger (≈50/50 long-calls vs credit spreads). The `auto_log_skip_long_puts`
filter has since funneled new logs into long-calls only. The weights that
performed well across the mixed ledger don't match the long-call profile —
specifically, `pop`/`theta`/`rr` matter much more for OTM long-calls than for
the credit-spread side, and `spread` is uninformative for long calls.

### Manual edits applied (delta ≥ 0.005)

| Component | From | To | Δ |
|---|---:|---:|---:|
| `spread` | 0.0881 | **0.0195** | **−0.069** |
| `pop` | 0.0322 | **0.0878** | **+0.056** |
| `theta` | 0.0149 | **0.0488** | **+0.034** |

All other components renormalized proportionally to preserve the 0.9999 budget.

### Caveats

- n=33 is small. With n=33, IC noise is ~±0.17 (one SD). The ±0.25 ICs are
  borderline-significant; ±0.39 (rr) is likely an artifact of low score variance
  (std=0.004).
- The signs of the three corrections are individually consistent with how each
  signal *should* behave for long-call buyers (you want high `pop`, you want
  cheap `theta`, you don't care about `spread`). Direction confidence > point-
  estimate confidence.
- The variance-zero scorer fix shipped the same day ([[project_variance_zero_scorer_fix]])
  means the next `--calibrate --apply` will have six previously-dead signals
  available; this manual tuning is the bridge until that next calibration.

### Validation plan

- Watch the next 2 weeks of auto-logged trades against this tuning.
- If the post-2026-05-20 cohort beats the post-2026-05-11 cohort
  (avg pnl_pct > 0, win-rate > 40%), the surgical fix is validated.
- Defer the next `--calibrate --apply` until ≥ 2026-05-25 AND ≥ ~30 new closed
  trades on these weights so we don't compound short-sample noise.

### Revert procedure

```bash
cp config.bak.20260520-145310.json config.json
rm -f ic_weights_cache.json
```

---

## 2026-07-31 — Exit marking basis changed (no weight change)

Not a calibration. Recorded here because it changes the **basis on which exits
are recorded**, and any step-change in the cohort's statistics after this date
needs its candidate explanation on the record.

**What changed** (`docs/MARK_TRUSTWORTHINESS_SPEC.md`, signed 2026-07-31,
commit 539c7e7):

1. Marks now prefer the **live bid/ask mid** over the last traded price. A mid
   is taken only from a two-sided, uncrossed book; anything else falls through
   to last trade -> daily close -> model.
2. The model fallback prices at the row's **stored `entry_iv`**, not a
   hardcoded 0.30. Marking an 80-vol name at 30 vol is what made model marks
   dangerous.
3. **A model-sourced mark can no longer fire a price-based exit.** The row
   stays OPEN and a warning names it. For multi-leg structures one modelled leg
   gates the whole structure — it is only as trustworthy as its worst leg. The
   pure-DTE time exit may still fire (its trigger never reads the mark) and
   stamps `(model mark)` into the exit reason.

**Expected effect on the numbers:** fewer spurious stop/take-profit exits on
illiquid contracts, and slightly later exits on quoteless ones. Bounded by the
deterministic expiry settlement, which is unchanged and still guarantees no row
hangs OPEN forever.

**No recorded exit was modified.** Exits before this date remain on the old
basis; that discontinuity is real and is why this entry exists.

**Also on this date, separately:** the Phase-1 gate moved to v2 (rank IC,
posterior bands, effective n) and resolved **STOP** at n=92. See
`status/DECISIONS.md`.

---

## 2026-08-06 — friction ceiling tightened 0.50 → 0.25

**Changed:** `config.max_friction_to_credit` 0.50 → 0.25.
**Backup:** `config.bak.20260806-224238-friction.json`.
**Applied by:** allocation-backtest evidence, not a calibration run.

**Why.** The allocation backtester measured the median crossing cost at exactly
**50% of credit** — $90 at mid, $45 actually received, over 400 real trades with
a median per-leg bid-ask of $0.35. A 0.50 ceiling therefore admitted roughly
half of all candidates, and the backtest says those are the ones that lose.

The supporting effect is the largest measured anywhere in this system:
restricting to names whose spreads are 3.6% of mid rather than 16-21% moved the
identical strategy from **-6.76% to -0.64%** return on capital, t=-17.84 over
10,363 trades. Full evidence in `docs/ALLOCATION_BACKTEST_FINDINGS.md`.

**Direction of risk.** This filter only ever REFUSES trades; it cannot add one.
Tightening it is the conservative direction — it removes candidates the evidence
says lose money, at the cost of a smaller cohort.

**Cohort impact, stated plainly.** Fewer auto-logged trades means the gates
accrue more slowly. That is a real cost and it is deliberate: the alternative is
continuing to accrue trades that the friction measurement says cannot profit.

**What was NOT changed, and why.** Six further proposals in
`docs/SCORER_IMPROVEMENTS.md` are documented and left unapplied:

- raising `composite_weights.spread` (0.0768) — would require renormalising all
  26 weights and re-ranks everything; larger blast radius than the evidence
  justifies in one step
- raising `iv_rank` (0.0260) and gating on it — suggestive only, clustered t=0.83
- a trend gate for short puts — clustered t=-1.42, not established
- the 50% take-profit on spreads — changes LIVE EXIT behaviour, which alters the
  open book and the gate cohort, and requires operator sign-off per DECISIONS.md
- tail-risk display, and ranking by something other than `quality_score`

Applying several at once would also be several more configurations against a
deflation bar that already reflects 34.
