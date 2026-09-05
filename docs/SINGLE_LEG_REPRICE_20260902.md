# Single-Leg Closed Book, Repriced Under the Measured Spread Surface (2026-09-02)

Task A of the 2026-09-02 work order: does the single-leg book still show
positive expectancy once friction is measured per-contract (the fitted
spread surface, PR #87) instead of assumed? Produced by
`scripts/reprice_single_leg_book.py`, run against the live ledger on
2026-09-02. Reporting only — no scoring, exit-path, or gate changes.

## What this answers

The book's headline profit factor has never been repriced against
contract-level friction — only against per-strategy medians
(`scripts/cost_model_report.py`) or against a flat $0.05 constant. This
reprices every single-leg closed trade's cost assumption using the
delta/DTE/open-interest surface fitted from 606,986 archived quotes, and
reports net profit factor beside the gross figure, with a 95% CI
bootstrapped and clustered on entry day. This is the number the rest of the
2026-09-02 work order (Task B, Task C) is conditioned on: **if net PF < 1.0
on capital at risk, stop and report before doing anything else.**

## Cohort

- **497 single-leg trades priced**, **429 multi-leg trades refused** — 46% of
  the 926 closed trades this script can see. Multi-leg is refused because
  `entry_price` on a spread is a **net credit** across legs, not a single
  leg's mid (`net_credit IS NOT NULL` is the structural test, not the
  strategy name — this repo shipped a defect where every Bear Call was
  labelled "Bull Put" for months, so the name cannot be trusted for this).
  Pricing a net credit on a leg-calibrated surface would report a number
  that is not a spread cost at all.
- By strategy: 312 Long Call, 108 Short Put, 77 Long Put (0 Short Call
  closed to date).
- **Open interest coverage: 80/497 (16%) join a real archived quote** on
  their own entry date and price off the surface's exact cell. The
  remaining **417/497 (84%) have genuinely unknown open interest** — not
  merely omitted from one lookup — so two figures are reported for them: a
  **central** estimate (the OI-collapsed marginal across every liquidity
  bucket) and a **conservative** bound (the most-illiquid-bucket pin). The
  true cost lies between the two; known-OI rows are identical in both.

## Method

**This does not use `execution_costs.reprice_pnl_pct`**, though the work
order named it. Verifying that function against how single-leg trades are
actually closed showed it would inject a modelling error: `reprice_pnl_pct`
adds back an "old" friction fraction computed from a `CostModel`, which is
only valid when the friction actually charged was a strategy-level constant.
`scripts/cost_model_report.py` already excludes single-leg trades from its
own reprice for exactly this reason — their close path
(`paper_manager._get_spread_slippage`) charges 30% of the **live** quoted
bid-ask width at the moment of exit, floored at $0.05/share and capped at
$0.50 — a per-trade, per-exit-time market observation with no `CostModel`
equivalent. An expired single-leg trade charged **zero** friction
historically, not one side (`paper_manager.py`'s `dte<=0` settlement branch
never subtracts anything; 0 of the 497 rows here hit that path, but the code
handles it correctly rather than assuming it away).

Instead, the **gross** (pre-friction) return is recomputed directly from
`entry_price` and `exit_price` — both stored on every closed row — using
exactly the formula that produced them
(`paper_manager._evaluate_short_single_leg_exit` /
`_evaluate_long_single_leg_exit`: short → `(entry−exit)/entry`, long →
`(exit−entry)/entry`). That needs no cost-model assumption at all. Only the
**new**, surface-measured round-trip friction is then subtracted
(`execution_costs.round_trip_friction`, one side only for a hold-to-expiry
close), isolating the cost assumption for real.

**Open interest convention**: the ledger has no `open_interest` column. A
two-sided archived-quote join on `(symbol, strike, expiration, type, entry
date)` supplies real OI where it exists. Where it doesn't, the conservative
figure (`SpreadSurface.relative(..., open_interest=None)`) resolves to the
most **illiquid** bucket — an upper bound on friction, not a lower one — and
the central figure (`SpreadSurface.oi_collapsed_relative`) is the genuine
OI-collapsed marginal. Both are fit on the archive's 15 liquid symbols and
applied to a book spanning far more tickers, which understates friction in
the same direction for both — the direction that flatters a book.

## Result

| basis | denominator | gross PF | repriced PF | 95% CI (clustered on entry day) | contains 1.0? |
|---|---|---:|---:|---|---|
| central (OI-collapsed) | entry premium | 1.216 | **1.103** | [0.851, 1.419] | yes |
| central (OI-collapsed) | capital at risk | 1.216 | **1.089** | [0.836, 1.412] | yes |
| conservative (bucket-0 pin) | entry premium | 1.216 | **0.947** | [0.727, 1.221] | yes |
| conservative (bucket-0 pin) | capital at risk | 1.216 | **0.957** | [0.733, 1.241] | yes |

n=497 single-leg trades, 4,000-resample bootstrap, seeded (reproducible).

**Every interval contains 1.0.** This is the same character as the
book-wide finding already on record (PF 1.044, CI [0.87, 1.24], on the
whole book's capital-at-risk basis) — no edge either way, not a loss and not
a gain. The point estimate moves with the OI assumption: central sits just
above 1 on both denominators (1.089–1.103), conservative sits just below on
both (0.947–0.957). Given open interest is unknown for 84% of this cohort,
neither point estimate should be read as more than a lead — the interval is
the honest answer, and it says "no evidence of edge" regardless of which OI
assumption is used.

## Per-strategy cost drag

```
  strategy        n  avg drag $  avg drag % of credit
  Long Call     312       27.08                 2.4%
  Long Put       77       18.73                 2.7%
  Short Put     108       16.73                 3.0%
```

Cost drag (central estimate) runs 2.4–3.0% of credit per trade, fairly even
across the three strategies — none of them is being singled out by the
repricing the way `cost_model_report.py` found for the multi-leg book (where
a flat constant undercharged Bull Put 3.2x and overcharged Bear Call 2x).
That asymmetry was a property of a flat per-strategy constant; the surface,
conditioned on delta/DTE/OI, does not reproduce it here.

## What this does not show

- Medians (and the surface's cell values) hide the tail — the cost of
  exiting a losing position in a fast market is not the median quote of a
  calm one.
- Nothing here re-examines entry or exit timing, only what the round trip
  was charged.
- **This reprices only the 54% of the closed book that is single-leg.** The
  multi-leg 46% needs per-leg mids the ledger does not record yet (Task C
  item 1: recording per-leg bid/ask/mid at entry and exit) — until then this
  number is single-leg's cost drag, not "the book's."
- 84% of this cohort has unknown open interest, fit on 15 liquid archive
  symbols applied to a wider-ticker book; the true friction likely sits
  toward the central estimate rather than the conservative one, but that is
  a judgment call stated here, not a measured fact.

## Next step

Every CI here contains 1.0 and the central capital-at-risk point estimate
(1.089) is not below 1.0, so per the work order's stated condition this does
**not** block the rest of the plan: proceed to Task B's pre-registration
(does the gate add value — regression discontinuity on gate thresholds),
carrying forward the finding that the single-leg book, like the book as a
whole, shows no established edge in either direction once friction is
priced honestly.
