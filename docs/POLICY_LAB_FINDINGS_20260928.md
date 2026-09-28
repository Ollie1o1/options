# Counterfactual Policy Lab — Findings

**Run:** 2026-09-25 | **Written:** 2026-09-28
**Preregistration:** `docs/PREREG_POLICY_LAB_20260925.md` (committed `aa093a4`, before any sweep ran)
**Raw output:** `docs/POLICY_LAB_RESULT_20260925.md` (committed `bc66b1b`)
**Code:** `src/policy_lab/`, branch `feat/policy-lab`

## Result

**Null. Nothing promoted. No config change is authorised.**

750 exit policies replayed over 105,392 real Bull Put positions across 1,870
ticker-day clusters, under crossing costs, with 2,000 permutations and 10,000
bootstrap draws. Runtime 5 minutes 6 seconds.

| verdict | count |
|---|---|
| promote | **0** |
| reject | 119 |
| underpowered | 631 |

The best policy the lab could measure — `sp_tp0.65_slnone_dtenone_hold7` —
is worth **+0.0032 of capital at risk per trade (+0.32%)**, against a
preregistered detection floor of **+0.012 (+1.2%)**. It falls short by a
factor of four. Of the 119 measurable policies, 65 have a positive mean
difference and none clears the bar.

## What the result actually says

**No exit-policy knob in a 750-cell grid is worth more than roughly 1.2% of
capital at risk per trade, on 1,870 independent ticker-days, net of modelled
friction.**

That bound is the deliverable. It is materially different from the
twenty-five prior experiments in this repository, which returned "p >= 0.46"
or "CI contains zero" — statements indistinguishable from never having
looked. This one states the size of the effect it has ruled out.

## Why 631 policies are "underpowered" rather than "rejected"

They are dominated by the `stop_mult=1.0` family. Under the replay engine's
formula a credit structure stops when `close >= entry_price * stop_mult`, so
1.0 fires the moment the credit stops shrinking — it is not a stop at all.
Those policies bear no resemblance to the baseline, the paired difference
therefore fails to cancel the market factor, and the variance ratio lands at
0.70 against a 0.5 threshold.

The lab refused to score them rather than reporting confident nonsense. That
distinction is deliberate: "we could not measure this" and "we measured it and
it failed" are different claims, and collapsing them loses the one that tells
you to go and get better data. The `stop_mult=1.0` cells are a defect in the
grid I preregistered, not a finding — and the grid stays frozen, because
trimming it after seeing results is precisely the preregistration violation
`docs/RESEARCH_PROCESS.md` step 1 exists to prevent.

## One observation for a future preregistration — NOT a finding here

**Every top-ranked measurable policy carries `slnone` — no stop at all.**

This is consistent with the ledger's own record: across 187 closed trades the
stop-loss family cost **-$87,265** while take-profits earned **+$97,601**.
Two independent lines of evidence point the same way.

It is not a finding from this sweep. The effect is below the detection floor
this document declared in advance, and acting on it would be exactly the
error this project was built to prevent. It is a well-motivated hypothesis
for a **smaller, separately preregistered grid**, where fewer trials buy more
sensitivity. That is the legitimate route to more power, chosen before seeing
results rather than after.

## What was corroborated, and what was not

**Corroborated: the friction assumption.** Every Corpus A cost figure rests
on a modelled half-spread of 0.095, because 100% of its Bull Put marks carry
no two-sided quote. The chain archive's real single-name quotes — a
completely separate dataset — measure **0.0981**. Agreement to 0.003 from
independent sources.

**Not corroborated: sign agreement.** Corpus B was to provide an independent
check that policy differences point the same way on real per-leg quotes. It
cannot, and the reason is a data-provenance limit rather than a defect:
`chain_snapshots.snap_date` does not label a contemporaneous price. A snapshot
dated 2026-06-10 carries `last_trade_time=2026-06-08T09:57:41`; one dated
2026-09-15 carries `2026-09-11T11:19:46`. On the single date 2026-06-10 there
are 7,707 distinct `last_trade_time` values spanning February to June.
Meanwhile `trades.date` is date-only and fills happened at scan time. The two
sources are different observations of the same contracts at different moments,
and no baseline, cost model or loader change can close that gap.

**So this null has its friction assumption verified and its direction
unverified.** That is a real gap and it is stated here rather than omitted.

## Limits, all preregistered

- **One regime.** 26 trading days. SPY's worst drawdown across the entire life
  of this book is **-4.49%**, against a long-run **-55.2%**. Nothing here speaks
  to a volatility event, which remains the dominant risk to a short-premium
  book.
- **Modelled friction.** 100% of path points were widened from mid by a
  single-leg half-spread applied to a two-leg round trip, which likely
  **under-states** friction — the optimistic direction.
- **Corpus A's own P&L is not ground truth.** It reproduces at 79.7%, and
  stop-loss positions reproduce worst. The paired comparison never reads
  recorded P&L, so this does not bias `d`; but no figure from
  `candidate_positions.pnl_pct` should be quoted.
- **Path depth.** Median 5.75 marks per position, so coarse thresholds are
  answerable and intraday timing is not.
- **Long premium is untrustworthy here.** No recorder baseline was ever
  recovered for Long Call or Long Put on Corpus A; they fall back to an
  intended policy with the same degenerate-DTE risk that was fixed for short
  premium. Do not read long-premium rows from this corpus.
- **Survivor count is not a discovery count.** 643 of 750 policies cleared the
  family-wise gate. The grid is a Cartesian product of heavily correlated
  cells, so once the best clears, most neighbours follow. The permutation test
  bounds the chance that *the best* policy is spurious; it says nothing about
  how many distinct effects exist.

## What this does not authorise

No real money. No change to the auto-log path. No change to any selection
rule. No config change at all — nothing was promoted.

## Recommended next step

The exit-policy lever has now been measured and bounded. Against the original
three directions, the evidence has moved:

The **benchmark-first rebuild** is now the strongest remaining candidate.
Exit policy was the best-motivated alternative — it owned 100% of the book's
P&L variance and had never been tested — and it does not pay above 1.2% per
trade. That materially strengthens the case that the binding constraint is not
which trades to take or when to leave them, but that **nothing in this system
is measured against a null strategy**. A book that returned +0.067% on capital
at risk while SPY returned 18.8% still has no benchmark recorded anywhere.

The cheap, honest follow-up inside this lab is a **smaller preregistered grid
testing the no-stop hypothesis** — four to six cells rather than 750, where
the detection floor falls far enough to resolve a +0.3% effect.
