# ROADMAP — the path to real money

**Goal:** Deploy real money on Long-Call options, but only after the screener proves a real,
statistically significant edge on trades it has never seen.

**Strategy decision:** Trade *only Long Calls* to start. They are the one strategy with a
positive track record (PF 1.46x). Bear Calls, Long Puts, Iron Condors, etc. lose money and
are quarantined — still logged for data, but excluded from the real-money decision.

---

## Phase 1 — Truth-machine ✅ DONE (2026-05-29)

Built the tooling to *measure* edge honestly.
- Cohort quarantine (only Long Calls count toward the real-money decision).
- Out-of-sample walk-forward IC harness (no data leakage — the old +0.023 was contaminated).
- Weekly gate checkpoint + startup banner.
- Tuned Long-Call weight candidate evaluated, deliberately left inactive (anti-overfitting).

**Outcome:** First honest out-of-sample read = **+0.10 IC (p=0.48)**. Promising, unproven.

---

## Phase 2 — Forward Observation ⏳ IN PROGRESS (started 2026-05-29)

Let the data accumulate. No new code — just patience and the weekly checkpoint.
- Keep auto-logging Long Calls.
- Each Sunday the checkpoint reports the gate decision.

**The gate, at 50+ closed cohort trades:**
- IC ≥ 0.08, p < 0.05 → **READY** (go to Phase 3)
- 0.03 ≤ IC < 0.08 → **EXTEND** (2 more weeks, max 2 extensions)
- IC < 0.03 at week 6 → **STOP** (no edge — pivot, do not deploy)

**The hard rule:** if it says STOP, we honor it. Not deploying is a valid, money-saving outcome.
Estimated duration: ~4–6 weeks.

> **2026-06-07 — read `docs/VALIDATION_POWER.md` before trusting the gate.** A power
> analysis showed n=50 is *underpowered* for a modest edge: at n=50 the smallest IC
> that is significant at p<0.05 is ~**0.28**, so the `p<0.05` clause (not the
> `IC≥0.08` floor) is what binds. A real-but-modest 0.10 edge would need ~780 trades
> to clear a frequentist gate. **Implication:** a READY at n=50 means a *strong*
> edge — size accordingly. Recommended addition (not yet adopted): a Bayesian
> tie-breaker (n≥50 AND P(true IC≥0.08) ≥ 0.85). Thresholds unchanged pending a
> human decision once n≥50.

---

## Phase 3 — Execution Stack ✅ BUILT 2026-06-07, 🔒 INERT until gate fires READY

Built ahead of the gate so READY → first trade is same-day (the gate stays the only
switch). All live output is gated behind BOTH `gate==READY` AND
`config.live_execution.enabled` (default false); until both, every ticket is DRY RUN.
Check arming with `python -m src.execution.pipeline`. Mirror mode only — no broker API.
- **Position sizing** ✅ `src/execution/sizing.py` — half-Kelly informed, hard caps
  (2% risk, 10% cost per trade).
- **Exit rules** ✅ `src/execution/exits.py` — TP / SL / time-exit, reusing
  `paper_manager._normalize_exit_rules` (one source of truth).
- **Order ticket** ✅ `src/execution/ticket.py` — mirror-mode ticket + the hard switch.
- **Slippage tracker** ✅ `src/execution/slippage.py` — record real-vs-paper fills.
- **Go-live runbook** ✅ `docs/GO_LIVE_RUNBOOK.md`.
- Tests: `tests/test_execution.py`, `tests/test_execution_pipeline.py` (19, green).

---

## Definition of "done" (real money flowing)

1. Forward cohort proved IC ≥ 0.08, p < 0.05 on 50+ unseen trades (read alongside
   `docs/VALIDATION_POWER.md` — a READY at n=50 implies a strong edge).
2. Sizing + exit rules shipped and tested. ✅ (Phase 3 built 2026-06-07, inert.)
3. You've run mirror mode for a couple of weeks and slippage is acceptable.
4. You flip `live_execution.enabled`, follow `docs/GO_LIVE_RUNBOOK.md`, and place a
   real Long Call sized by the system with rules-based exits.

Until step 1, everything else waits.
