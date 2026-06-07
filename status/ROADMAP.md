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

---

## Phase 3 — Execution Stack 🔒 GATED (only if Phase 2 fires READY)

Built only once the edge is proven. This is the "actually trade real money" layer:
- **Position sizing** — Kelly-fraction, account-aware, hard cap (e.g. 2% risk per trade).
- **Exit rules** — automatic take-profit / stop-loss / time-stop / IV-crush stop.
- **Mirror mode** — the system emits an order ticket; you place it manually (no broker API yet),
  and we track real-vs-paper slippage.
- **Go-live runbook** — the checklist for putting actual capital in.

Estimated duration: ~2 weeks of build, once unlocked.

---

## Definition of "done" (real money flowing)

1. Forward cohort proved IC ≥ 0.08, p < 0.05 on 50+ unseen trades.
2. Sizing + exit rules shipped and tested.
3. You've run mirror mode for a couple of weeks and slippage is acceptable.
4. You click "buy" on a real Long Call, sized by the system, with rules-based exits.

Until step 1, everything else waits.
