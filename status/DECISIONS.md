# DECISIONS — the judgment calls and why

A short log of the non-obvious choices we made, so future-us remembers the reasoning.

---

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
