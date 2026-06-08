# STATUS — where we are right now

**Last updated:** 2026-06-03
**Current phase:** Phase 2 — Forward Observation (cohort **reset clean**)
**Real money deployed?** ❌ Not yet. Waiting on the gate (see below).

> **2026-06-03 — horizon bug FIXED, cohort reset clean.** The 2026-06-01 heads-up
> is resolved. Confirmed empirically: **all 15** forward-cohort Long Calls had been
> logged at 14–27 DTE — every one started inside the 21-DTE time-exit, so they
> force-closed at the 3-day min-hold floor. The gate was measuring 3-day returns,
> not swings (that's where the IC −0.65 came from — one bad semiconductor week at
> 3 days, not real signal).
>
> **What changed today:**
> 1. **Cohort DTE floor (30).** Long Calls under 30 DTE now auto-log as
>    `paper_only=1` (data only, excluded from the gate). The floor is horizon-aware:
>    if `cohort_min_dte` is unset it derives from `time_exit_dte + 9` so it can never
>    drift below the time-exit. (`apply_auto_log_allowlist`)
> 2. **Reclassified the 15 contaminated trades** to `paper_only=1`
>    (`scripts/reclassify_cohort_horizon.py`, DB backed up). Forward cohort is now
>    **0 clean trades** — honest reset. Phase 2 effectively restarts from here.
> 3. **Startup automation-health warning.** The screener now warns if auto-log /
>    exit-enforcer / weekly-checkpoint have gone stale — so a silent cron death
>    can't go unnoticed for days again. It currently flags the **exit-enforcer
>    (~21d stale)**.
>
> Still standing from 2026-06-01: the auto-close enforcer bug (expired/legacy
> multi-leg trades left OPEN) was patched + tested. Scheduled jobs (cron) stopped
> firing ~May 20 — see "What's blocking", item 4 (now also surfaced at startup).

---

## The one number that matters

**Out-of-sample Long-Call IC: +0.102** (p = 0.48)

- "IC" = how well the screener's score predicts actual returns on trades it never trained on. 0 = coin flip, higher = real skill.
- +0.10 is **directionally encouraging** — far better than the old in-sample 0.023 that turned out to be contaminated.
- BUT p = 0.48 means it is **not yet statistically trustworthy**. With only ~50–100 test observations, this could still be luck.
- **We do not risk real money until this clears the bar:** IC ≥ 0.08 with p < 0.05 on 50+ *fresh* closed trades.
- This number is the read on the **100 historical** Long Calls (untouched). The **forward
  gate cohort is now n=0** after today's clean reset — decision: **GATHERING**.

---

## The gate (what unlocks real money)

```
Forward cohort = Long Calls, closed, logged on/after 2026-05-27, cohort-eligible.

  < 50 closed trades ............... GATHERING  (keep logging — we are here)
  ≥ 50 & IC ≥ 0.08 & p < 0.05 ...... READY      → build execution layer, go live
  ≥ 50 & 0.03 ≤ IC < 0.08 .......... EXTEND     → gather 2 more weeks
  ≥ 50 & IC < 0.03 at week 6 ....... STOP       → no edge; pivot, don't deploy
```

When this fires READY or STOP, a banner appears at the top of the screener on startup.

---

## Cohort progress

| Metric | Count |
|--------|-------|
| Forward cohort — **closed, clean** (counts toward the 50) | **0** (reset 2026-06-03) |
| Forward cohort — open, clean | 0 |
| Reclassified to data-only today (sub-30-DTE contamination) | 15 |
| Historical closed Long Calls (used for the OOS read above) | 100 (untouched) |

**The cohort is now empty and clean.** That is the correct, honest state: 0 trustworthy
trades is worth more than 15 noisy ones. From here every cohort trade is a ≥30-DTE Long
Call with real swing runway before the time-exit.

**Rough timeline:** at the current logging pace — and now that only ≥30-DTE calls qualify —
reaching 50 closed clean cohort trades takes ~5–7 weeks. Bias selection toward 30–45 DTE
calls to keep the cohort filling.

---

## What's live and working

- ✅ Long Calls auto-log into the validation cohort; all other strategies are quarantined (logged for data, excluded from the real-money decision).
- ✅ **Cohort DTE floor (≥30, horizon-aware)** — only swing-runway calls enter the gate; sub-30-DTE calls log as data-only.
- ✅ Out-of-sample walk-forward harness — gives an honest, leak-free IC.
- ✅ Weekly checkpoint computes the gate decision and writes a report.
- ✅ Startup banner surfaces READY/STOP so you can't miss it.
- ✅ **Startup automation-health warning** — flags auto-log / exit-enforcer / checkpoint if they go stale (catches silent cron death).
- ✅ Baseline scoring weights (a tuned Long-Call candidate exists but is **deliberately not active** — see DECISIONS.md).

---

## What's blocking real money (in plain terms)

1. **Not enough fresh trades** — 0 of 50 clean cohort trades (just reset). This is now just time.
2. **Edge not proven yet** — the horizon bug is fixed, so the cohort will finally measure swings, not 3-day noise. Let the clean sample grow.
3. **No execution layer yet** — position sizing and exit rules (Phase 3) are intentionally NOT built until the edge proves out, to avoid wasting effort on an unproven signal.
4. **Automation now runs at screener startup (cron retired 2026-06-07)** — cron
   was silent since ~2026-05-20 (lost Full Disk Access) and a month of attempts
   never made it reliable, so it is abandoned. `src/maintenance.py` now runs
   auto-log (once per window/day, in-window weekdays) and the weekly checkpoint
   (≥7 days) at screener startup; exit-enforcement already ran inline at startup
   and now logs that it did. Trade-off: **the cohort only fills on days you run
   the screener** — the new `Forward cohort: X/50 …` startup line makes that cost
   visible. See `status/DECISIONS.md` (2026-06-07).

---

## Your next action

- [ ] **Run the screener regularly.** Startup now auto-runs exit-enforcement, the
  weekly checkpoint, and (in-window, weekdays) auto-log — cron is retired and no
  longer needed. The cohort only fills on days you run it, so running it *is* the
  discipline. Watch the `Forward cohort: X/50` line tick up.
- [ ] Check the Sunday `reports/checkpoint_*.md` (or this file) weekly.

Nothing else to do but let the data accumulate. The discipline *is* the strategy.
