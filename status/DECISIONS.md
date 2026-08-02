# DECISIONS — the judgment calls and why

A short log of the non-obvious choices we made, so future-us remembers the reasoning.

---

## 2026-08-01 — Short-premium family promoted out of paper_only

**Why:** the short-premium gate's Arm A reads READY, but every row in its cohort
was `paper_only=1`, so the evidence was formally disqualified from authorising
anything. The operator ruled that these rows should count.

**What changed:** 285 rows (Bull Put 131, Bear Call 135, Short Put 109... 
counted before promotion: 81/104/100 flipped) set to `paper_only=0`, and
`auto_log.paper_only_strategies` reduced from
`[Bear Call, Iron Condor, Bull Put, Short Put]` to `[Iron Condor]` so future
rows log clean. Backup taken first:
`backups/paper_trades.db.bak.20260801-223429`.

**Scope was deliberately narrow, because `paper_only` means two different
things in this ledger.** For the credit family it is a STRATEGY-SELECTION
quarantine (the 2026-05-29 "trade only Long Calls" decision). For Long Call and
Long Put rows it marks DTE CONTAMINATION (2026-06-03) — those rows are bad
data, not merely out of scope. Only the first was promoted: Long Call's 41 and
Long Put's 26 contaminated rows are untouched, as is Iron Condor, which is not
part of the short-premium gate's family.

**What this does NOT change, and the distinction matters:** promoting a row
changes its CLASSIFICATION, not the evidence it carries. The binding caveat is
untouched — the window is under two months with no volatility shock, and short
premium's characteristic failure is a single tail event, not a slow bleed. Every
large loss in the wider ledger is a cash-secured put at $32k-$83k of collateral,
all excluded by the affordability cap, so this cohort has never actually taken a
big loss. The gate now says so in place of the paper_only caveat rather than
simply dropping a line.

**Nothing is armed.** `python -m src.execution.pipeline` still reports DISARMED
on two blockers: `live_execution.enabled=false`, and `gate=STOP` — because the
execution pipeline reads the LONG CALL gate, which resolved STOP. Wiring the
pipeline to authorise off the short-premium gate instead is a separate decision
that has not been made and is not implied by this one.

---

## 2026-07-31 — Auto-log refuses credit trades the spread would eat

**Why:** building the short-premium gate surfaced that **31 of 188 logged
short-premium trades carried round-trip friction exceeding the entire credit
received** — micro-spreads with $57 of median capital at risk against roughly
$65 of spread. Those cannot profit at any win rate. The finding was worth
nothing as a note in a document: the screener would keep suggesting them and the
feeder would keep logging them, so every future cohort would carry the same
contamination.

**Decision:** `config.auto_log.max_friction_to_credit = 0.50`, enforced in
`PaperManager.log_trade` beside the affordability gate. Refusals increment
`untradeable_rejected` and print, so a feeder that has gone quiet names the gate
that held it back rather than looking broken. `allow_untradeable=True` bypasses
it for a deliberate manual entry; `null` disables it.

**This is the mirror of the 2026-07-29 affordability gate.** That one refuses
positions too LARGE for the account; this one refuses positions too SMALL to
survive their own market. Both are the same error — measuring a population no
rational trader would open — at opposite ends of the size range.

**Scope, deliberately narrow:** credit structures only. A debit trade has no
credit to compare against, so `_friction_to_credit_ratio` returns None and the
guard never fires on Long Call or Long Put. A missing credit also returns None
rather than 0.0 — an unrecorded credit is a row the guard should not judge, not
a free trade.

**Cost accepted:** the feeder will log fewer credit trades, and the ones it
drops are disproportionately the cheap-looking small ones. That is the intended
trade. Watch `untradeable_rejected` for a fortnight to confirm it is gating
rather than starving, the same way the affordability gate was watched.

**Not retroactive.** No logged row was deleted or altered; the existing cohort
still contains the 31, which is why the short-premium gate applies the same
filter at read time and reports how many it excluded.

---

## 2026-07-31 — Short-premium gate: a different statistic, and a tradeability filter

**Why a new gate at all:** the Long Call gate resolved STOP, and Long Call is
not what the book earns on. Validating one strategy to authorise going live and
then trading another is a mismatch between the experiment and the decision it
licenses (spec section 2.4).

**Why not the same machinery:** short-premium returns are the mirror image of
long-call returns — many small wins, occasional large losses (66% win rate,
skew -0.25). So Arm A asks about the MEDIAN net return-on-risk, not an IC, and
gets a **bootstrap** rather than a Fisher-z posterior, because "does this family
make money" is a question about a location parameter, not a correlation.

**The bootstrap resamples ENTRY DAYS, not trades.** 188 trades landed on 27
entry days, up to 33 in one day. Resampling trades would treat correlated
observations as independent and manufacture confidence; a test pins that
clustered resampling widens the interval relative to ignoring clustering.

**The finding that changed the design.** Re-costing on measured per-structure
spreads moved the MEAN by -12.7pp while barely moving the median, because the
cohort is full of micro-spreads: median capital at risk is $57, and a Bull Put
with $26.50 at risk carries $64.80 of round-trip friction. **31 of 188 trades
have friction exceeding the entire credit received** — they cannot profit at any
win rate. Measuring them is measuring a population no rational trader would
open, exactly like the unaffordable trades of 2026-07-29, but at the other end
of the size range.

**Decision:** exclude trades whose round-trip friction exceeds 50% of the credit
(`TRADEABILITY_MAX_FRICTION_RATIO`). On what remains (n=140) the median
(+28.1%), mean (+19.1%) and capital-weighted return (+18.6%) all agree in sign,
and the verdict is robust from a 25% to a 100% threshold. The loose 50% setting
is the conservative choice: tightening it improves the result.

**A coherence guard, mirroring the Long Call gate's sign guard.** Arm A will not
fire READY when the median and the capital-weighted return disagree in sign. A
positive median with a negative book is the signature failure of short premium —
picking up pennies in front of a steamroller — and it is exactly what the raw,
unfiltered cohort showed (+12.6% median against -0.6% mean).

**Current verdict: Arm A READY (posterior 100%), Arm B EXTEND (rank IC +0.091).**
Arm B does not veto Arm A; it decides whether live trading would use the scorer
to pick contracts or trade the family on structural rules.

**This gate is REPORTING ONLY and does not authorise anything yet.** Every row
in its cohort is `paper_only=1` — the family was quarantined in 2026-05-29 as a
strategy-selection choice, not a data-quality one. Promoting a quarantined
cohort to a validation cohort is an operator decision. And the window is under
two months with no volatility shock: short premium's characteristic failure is
a single tail event, so a high posterior here means "no evidence against", not
"the tail has been survived". Both caveats print with every run.

---

## 2026-07-31 — Gate v2 adopted: rank IC, posterior bands, bounded EXTEND

**Why:** the v1 gate could not answer its own question. It decided on the
Pearson IC of returns floored at -100% and driven above by a handful of
take-profits — the distribution Pearson handles worst — and on the current
cohort Pearson and Spearman disagreed in SIGN. Its READY arm demanded
`IC>=0.08 AND p<0.05`, which at n=50 needs an observed IC of ~0.286, 3.6x the
stated bar. And its EXTEND had no exit: an IC drifting between 0.03 and 0.08
extended forever, so "keep gathering" was a permanent answer rather than a
decision. Three states, one of them an infinite loop.

**Decision:** `config.gate.version = 2`. The gate now reads the **Spearman rank
IC**, sized in **effective n** (entry-day clustering: ICC 0.080, design effect
1.27, so 92 nominal trades are 72.5 effective), and decides on the Bayesian
posterior `P(true rank IC >= 0.08)` — READY at >=0.85, STOP at <=0.15, EXTEND
between, granted **at most twice**, after which it must resolve. A
sign-agreement guard withholds READY whenever rank and Pearson disagree in
sign, so real money is never authorised on a statistic its counterpart
contradicts. Spec: `docs/GATE_REDESIGN_SPEC.md`, signed by the operator
2026-07-31.

**v1 is not deleted.** `decide_v1` is preserved verbatim and both verdicts are
printed on every checkpoint and in `GATE_STATUS.md`. A superseded rule that
vanishes cannot be audited, and whether the two agree is itself evidence.

**First verdict: STOP, from both rules.** n=92, Pearson -0.065, Spearman
-0.132, posterior 4%. The redesign did not rescue the long call — it reached
the same conclusion on a sounder statistic and showed its working. That is the
outcome the phased approach was built to produce: "we proved there is no edge"
is worth more than losing real money to find out.

**What did NOT change:** `live_execution.enabled` is still the hard switch, and
READY alone never places a trade. The affordable-subset and short-premium
cohorts remain reporting-only; wiring the short-premium family to `decide_v2`
is the next step and its own decision.

---

## 2026-07-29 — Auto-log refuses positions bigger than the account; gate untouched

**Why:** The feeder had no size limit — the budget filter only ever applied in
"Budget scan" mode, and DISCOVER is documented "no budget limit". The result:
capital at risk per logged trade ranged $22 to $83,650 against a $750 budget, and
**every dollar of the book's loss sat in trades the account could not have opened**.
Over the cohort window, trades inside the budget are +$3,283 (+6.3% of capital
risked, n=247); trades above it are −$19,741 (n=160). Long Call alone goes
−$21,718 → +$183 once the unaffordable half is dropped. So the headline
"Long Call bleeds $17.6k" was a sizing artifact, not a signal result.

**Decision:** `config.auto_log.max_capital_at_risk = 750` (= `default_budget`), enforced
in `PaperManager.log_trade` — the single chokepoint all eight log sites funnel
through. Rejections print and increment `unaffordable_rejected` so a gated feeder
is visibly gated, never silently starved. `allow_unaffordable=True` bypasses it for
deliberate manual entries. Risk per structure is defined once in `src/capital_risk.py`
and stored per row (schema v16); ad-hoc `max_loss_usd or entry_price*100` was
costing a cash-secured put at the *credit received* rather than the collateral,
understating a WFC 77.5 put ~50×. Set the key to `null` to disable.

**Cost, accepted deliberately:** this refuses 108 of 242 historical Long Calls, 94 of
99 Short Puts and 116 of 160 Iron Condors, so cohort accrual slows markedly. Measuring
a population the account cannot trade faster is not worth having.

**Explicitly NOT changed:** the gate. `phase1_checkpoint` now *reports* the affordable
subset beside the nominal one (currently IC +0.086 p=0.50 n=64 nominal vs +0.119
p=0.52 n=31 affordable — neither resolves anything at this n), but the READY/EXTEND/
STOP rule still reads the nominal cohort. Re-pointing the gate is a human call to be
made on its own, per the 2026-06-07 no-silent-gate-change decision.

---

## 2026-06-07 — Built Phase 3 execution stack now (inert), not after READY

**Why:** Building the execution layer (`src/execution/`: sizing, exits, ticket,
slippage, pipeline) *before* the gate fires removes the ~2-week build tax between a
READY verdict and the first trade, without weakening discipline: every live ticket
is gated behind BOTH `gate==READY` AND `config.live_execution.enabled` (default
false), enforced by data in `pipeline.build_ticket`/`arm_status`, not by remembering
to pass a flag. Mirror-mode only (system prints a ticket, human places it, slippage
tracked) — explicitly NO broker API. Exits reuse `paper_manager._normalize_exit_rules`
so there's one source of truth. A STOP verdict shelves reusable code, not capital.
Runbook: `docs/GO_LIVE_RUNBOOK.md`. Arming check: `python -m src.execution.pipeline`.

## 2026-06-07 — Power analysis: n=50 gate is underpowered for a modest edge

**Why:** `scripts/validation_power_analysis.py` → `docs/VALIDATION_POWER.md` shows
that at n=50 the smallest IC significant at p<0.05 is ~0.28, so the `p<0.05` clause
binds, not the `IC≥0.08` floor; detecting a ~0.10 edge frequentist-clean needs ~780
trades. Decision: **leave thresholds unchanged for now** (a READY at n=50 legitimately
means a strong edge), but read every gate result alongside this doc, and revisit
adopting a Bayesian tie-breaker (n≥50 AND P(true IC≥0.08) ≥ 0.85) once n≥50. No
silent gate change — this is the basis for that future human call.

## 2026-06-07 — Retire cron; self-healing maintenance at screener startup

**Why:** Cron silently died ~2026-05-20 (lost Full Disk Access) and went unnoticed
for ~12 days; a month of attempts never made it reliable on this Mac (FDA +
Login-Items friction). Rather than keep fighting it, `src/maintenance.py` now runs
the jobs at **screener startup**, crash-isolated so a failure can never stop the
screener: auto-log (once per clock-window/day, weekdays, in-window) and the weekly
checkpoint (≥7 days). Exit-enforcement was *already* running inline at startup via
`PaperManager.update_positions()`, so maintenance deliberately does **not** re-run
it (would mean a second ~60s scan per boot); instead startup now appends to
`logs/enforce_exits.log` after enforcing, so the automation-health check reflects
reality instead of false-flagging it stale.

**Trade-off accepted:** the cohort only fills on days the screener is run. Made
visible by a new startup line: `Forward cohort: X/50 closed clean | open: Y |
weeks: Z | gate: <DECISION>` (reuses `phase1_checkpoint.compute_checkpoint`, so the
cohort filter has one definition). Throttle state in `logs/.maintenance_state.json`.

Also added `config.json → live_execution.enabled` (default **false**) — the hard
switch that Sub-project C (Phase 3 execution stack) will gate live tickets behind.

Plan/spec: `docs/superpowers/{specs,plans}/2026-06-07-*` (local-only, gitignored).

---

## 2026-06-03 — Cohort DTE floor of 30, and reset the contaminated cohort

**Why:** All 15 forward-cohort Long Calls had been logged at 14–27 DTE, which is *inside*
the 21-DTE time-exit window — so every one force-closed at the 3-day min-hold floor. The
gate was measuring 3-day returns, not swings (the IC −0.65 was one bad 3-day semiconductor
week). Fix: Long Calls under a DTE floor now log as `paper_only=1` (data only, out of the
gate). The floor is **horizon-aware** — if `cohort_min_dte` is unset it derives from
`time_exit_dte + cohort_min_runway_days` (21 + 9 = 30), so it can never silently drift below
the time-exit. We chose the entry-side floor over exempting Long Calls from the time-exit:
it's surgical (no live exit-rule change, no effect on other strategies) and keeps the
eventual real-money risk profile unchanged.

**Reset:** reclassified all 15 contaminated trades to `paper_only=1`
(`scripts/reclassify_cohort_horizon.py`, DB backed up first). Forward cohort → 0 clean trades.
0 trustworthy trades beats 15 noisy ones; Phase 2 restarts honest. The 100 historical Long
Calls behind the +0.10 OOS read were left untouched.

---

## 2026-06-03 — Surface silent automation failure at startup

**Why:** Cron died ~May 20 and went unnoticed for ~12 days. The screener now runs an
automation-health check at startup (`src/health.py`) that warns when auto-log /
exit-enforcer / weekly-checkpoint go stale, inferred from artifacts they already produce.
Observability is cheap; silent data rot is expensive.

---

## 2026-05-29 — Trade only Long Calls to start

**Why:** Across 225 closed paper trades, Long Calls were the only strategy with a positive
profit factor (1.46x). Bear Calls (0.41), Long Puts (0.50), Iron Condors (0.48) all lose money.
No point risking capital on strategies that bleed on paper. Others stay on for data, quarantined.

---

## 2026-05-29 — Phased, gated approach with a hard kill criterion

**Why:** The biggest trap in trading systems is building elegant execution infrastructure around
a signal that's actually a coin flip. So we prove the edge *first* (Phase 2), and only build the
real-money machinery (Phase 3) if it clears the bar. If at week 6 the edge isn't there, we STOP —
and "we proved there's no edge" is itself worth more than losing real money to find out.

---

## 2026-05-29 — Did NOT activate the tuned Long-Call weights

**Why:** A calibration produced an optimized Long-Call weight profile
(`configs/weights/long_call_v1.json`). We kept it as a *candidate* but left the screener on
**baseline** weights. The optimized weights are fit on a tiny in-sample set, and the out-of-sample
signal can't yet tell them apart from baseline (p=0.48). Activating now risks overfitting and would
contaminate the forward cohort with an unproven config. Revisit when the cohort hits 50+ trades.

---

## 2026-05-29 — The +0.023 IC was contaminated; +0.10 is the real read

**Why:** The old "IC = 0.023" number scored trades that the calibration had already fit to
(in-sample) — meaningless for predicting the future. The new walk-forward harness fits weights on
older trades and tests on newer ones it never saw (out-of-sample), which is the only honest way to
estimate real-world skill. That honest number is +0.10 — better, but still not yet significant.
