# The Research Loop

How every scoring/strategy idea gets tested in this repo, so a finding is
either enforced or closed — never left as a vibe. This formalizes a pattern
that already exists organically in the git history (a `docs/PREREG_*.md` or
`*-design.md` spec PR, followed later by a `docs/*_RESULT_*.md` PR) so it
runs the same way every time instead of depending on someone remembering to
do it right.

**`docs/QUANT_RESEARCH_MANUAL.md` (and its `.pdf`/`.tex`) were deleted
2026-09-10.** Written 2026-05-11, never updated, and directly contradicted
by nearly everything below it: it called VRP the highest-conviction signal
(dead — see step 6 and `docs/HOLDOUT_20260809.md`), described an
auto-adapting weight system (found null — purged walk-forward, p≥0.46
everywhere), and predated the PoP seller-convention fix. If you find a copy
of it anywhere (a stale checkout, a search index), it's a historical
artifact, not a reference — don't cite it.

## The loop

### 1. Preregister before looking at outcomes
Write a short spec: the hypothesis, the exact population it applies to
(which strategy, which universe — index vs single-name is not a detail,
see `docs/PREREG_SECTOR_CONDITIONING_20260813.md`), the metric, and the
minimum sample size needed for a verdict. Commit it as its own PR
(`docs/PREREG_<name>_<date>.md` or `docs/superpowers/specs/<date>-<name>-design.md`)
*before* running the test. Looking at results first and writing the
hypothesis after is how a fit gets mistaken for a finding.

### 2. Define the observation unit before counting anything
Entry day? Ticker? Trade? Rows overstate confidence when trades cluster
(this repo has re-done the same significance test three times after
counting rows instead of clusters). Use `src/prereg_ranker.py::icc_oneway`
and `design_effect` to check how much clustering inflates nominal N into
`effective_n`, and `required_effective_n` to size the test up front.

### 3. Purge for leakage
No feature or label may see data from inside its own forward-looking
window. See `docs/superpowers/specs/2026-08-29-walk-forward-purging-design.md`
for the mechanism — this repo's walk-forward silently wasn't OOS before
that fix, and every prior IC quoted from it is void.

### 4. Report a clustered interval, not a point estimate
A verdict only counts if the interval excludes the null (0 for a lift, 1
for a profit factor). Use `src/prereg_ranker.py::cluster_bootstrap_ci` or
`src/alloc/validate.py::clustered_tstat`/`summarise`. The whole book's PF is
1.044 with CI [0.87, 1.24] — that interval containing 1 is *why* "no
measured edge" is correct even though the point estimate is above 1.

### 5. Check N against the prereg'd minimum before concluding anything
If effective N falls short, the verdict is INSUFFICIENT — full stop, not
"promising." `csp_index_only` closed this way at n=19 vs a preregistered
MIN_N=20 rather than being stretched into a read. `src/phase1_checkpoint.py`
and `src/short_premium_gate.py` use `GATE_V2_MIN_N_EFF=50` as their own
gate-specific floor — a reminder that MIN_N is chosen per hypothesis, not
copied from elsewhere.

### 6. Residualize against known effects
Before crediting a "new" signal, control for whatever known factor could be
driving it (credit richness, friction, sector). `src/prereg_ranker.py`'s
`negative_control` and `half_sample_ics` exist for exactly this. Every
single-feature signal tested this way here has died: `atm_iv`, `vol_of_vol`,
`iv_rank`, `skew_25d`, VRP/IV-RV (single-name and index). The one survivor
is `long_delta`, and only on a cost-challenged structure. Don't re-run any
of the dead ones without a materially different population or a new
confound to control for — see `docs/HOLDOUT_20260809.md` and
`docs/PREREG_ENSEMBLE_20260905.md` for what "materially different" needs to
look like.

### 7. Run it through the promotion gate before trusting it
`src/alloc/report.py::promotion_verdict` already gates on DSR (`MIN_DSR`),
PBO (`MAX_PBO`), and a *clustered* t-stat (`MIN_TSTAT`) together — see
`docs/DSR_PROMOTION_GATE_SPEC.md` for why the clustering has to apply to
every check in the gate, not just some of them (an inflated DSR promoted
things a clustered t-stat would have rejected). Don't build a new promotion
threshold from scratch; call this one.

### 8. Stage enforcement: off → report → refuse
Never flip a validated finding straight to blocking trades. Ship it gated
by a config mode (`off`/`report`/`refuse`, unrecognised value always falls
back to `off`) and watch report mode against **real, continuing** production
runs before enforcing — not a single backtest snapshot. Confirm the code
is actually merged and running before treating "report mode" as live: two
gates here sat on unmerged PRs for two days while treated as shipped
(`docs` note: check `gh pr view <n>` / `git merge-base --is-ancestor`, not
a commit message).

### 9. Write the result down, win or lose
Close every prereg with a result doc (`docs/*_RESULT_*.md`), even a null.
An unwritten null gets re-tested; a written one doesn't. This repo has
closed more nulls than confirmations — that's the discipline working, not
a failure rate to fix.

## What's actually queued for this loop right now

Feature-hunting on the volatility surface is exhausted (step 6's dead list
covers nearly everything derivable from IV alone). What's left that hasn't
been run through steps 1-9:

- **Cluster cap + spread negative-EV gate** (`config.json`:
  `cluster_cap_mode`, `spread_negative_ev_mode`) — in report mode as of
  2026-09-10, step 8. Check accumulated data before flipping to refuse.
- **Insider/EDGAR filings** (`src/insider/edgar.py`) — built, never
  preregistered or tested for edge (step 1 has never happened for this
  module).
- **Catalyst calendar** (`src/catalyst/`) — 27% coverage, zero validation
  (steps 1-7 never run).
- **Portfolio construction** (correlation-aware sizing) rather than pick
  selection — the book's per-trade scoring may be fine while its
  concurrent-exposure construction (0.497 mean correlation vs 0.257
  baseline) is the actual leak.

New score-formula ideas on the existing inputs (IV, skew, vol-of-vol,
Greeks) should be treated as very low prior — that space has been searched
hard and come back empty every time.
