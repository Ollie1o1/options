# Preregistration — friction-to-credit admission sweep

**Written:** 2026-09-28, before any return in this cohort has been computed.
**Prior work:** `docs/POLICY_LAB_FINDINGS_20260928.md`,
`docs/BENCHMARK_RESULT_20260928.md`

## Why, and where the hypothesis came from

Tracing the gap between the accepted-candidate cohort (+4.24% mean return on
capital at risk) and the live book (+0.35% over the same period) showed:

- **Take-profits reconcile almost exactly**: live +23.53% vs modelled +23.62%.
- **Time exits diverge**: modelled +3.95%, live -7% to -11%.
- **Mechanism: friction is a fixed dollar cost.** Round-trip crossing is
  ~$14/contract either way, but a take-profit's average |P&L| is $96 while a
  time exit's is $53.56. Friction is a haircut on a win and a quarter of a
  flat outcome.

So the hypothesis is **mechanical, not fitted**: positions whose round-trip
spread is large relative to their plausible payoff should be the ones that
time-exit near flat and get converted into losses by cost.

**Honest provenance note.** This hypothesis was formed by looking at the
existing data, so this sweep is not an independent confirmation — it is a
powered test of a mechanism suggested by that data. Unlike the no-stop case
there is no held-out strategy available (Bear Call's accepted cohort is far
smaller), so the result must be read as suggestive, and a genuine confirmation
requires forward observations.

## Hypothesis

**H1.** Among accepted Bull Put candidates, restricting to those whose
round-trip crossing cost is a smaller fraction of mid credit produces a higher
mean return on capital at risk, with a ticker-day-clustered 95% CI excluding
zero for the difference against the unrestricted cohort.

**H2 (the decision-relevant one).** That improvement is large enough to offset
the reduced trade count — i.e. tightening raises *total* return given capacity,
not merely the per-trade mean.

## Friction measure (frozen)

`round_trip_pct` is **NULL for 100%** of accepted Bull Put candidates, so it
cannot be used. Friction is computed instead from the per-leg quotes stored in
`candidates.features_json`, which are present on **300/300** sampled accepted
rows:

```
mid_credit      = (short_bid+short_ask)/2 - (long_bid+long_ask)/2
cross_in        = short_bid - long_ask      # sell the spread
cross_out       = short_ask - long_bid      # buy it back
friction_ratio  = (cross_out - cross_in) / mid_credit
```

Measured distribution on accepted candidates: **p10 0.051, median 0.100, p90
0.211.** Note this is materially lower than the "27% of credit" figure in
project memory, which came from mid-vs-cross on 177 live ledger rows — a
different measurement on a different population.

## Grid (frozen, 5 cells)

Thresholds **0.075, 0.100, 0.150, 0.200, none**, each compared against the
unrestricted accepted cohort.

**0.05 is excluded in advance: it yields 15 clusters, below the lab's
MIN_CLUSTERS=20 bar.** Including it would generate an "insufficient" verdict
that adds nothing.

Cohort sizes, measured before any return was computed:

| threshold | n | clusters | % of cohort |
|---|---|---|---|
| 0.075 | 298 | 45 | 12% |
| 0.100 | 660 | 83 | 26% |
| 0.150 | 1,402 | 151 | 55% |
| 0.200 | 2,070 | 193 | 81% |
| none | 2,566 | 218 | 100% |

## Statistics — deliberately NOT the policy lab's paired design

This is an **unpaired subset comparison**: a tightened cohort is a different
set of positions, not the same positions scored two ways. The paired
variance-cancellation that makes the policy lab work (`sd(d)/sd(r)` = 0.172)
**does not apply here**, and no pairing will be claimed.

Each threshold's mean return gets a `(symbol, entry_date)`-clustered bootstrap
CI at n_boot=10000. The difference against the unrestricted cohort is reported
with its own clustered CI over the shared clusters.

## Declared in advance

- **The prior is that this is real but small.** The mechanism is arithmetic
  (cost is cost), so some improvement is near-certain; the question is size.
- **Power falls as the threshold tightens** — 45 clusters at 0.075 against 218
  unrestricted. **A null at the tighter cells is weak evidence of absence**, and
  will be reported as underpowered rather than as a refutation.
- **Tightening reduces trade count**, so H2 can fail while H1 succeeds. If the
  per-trade mean rises but total return falls, the honest recommendation is to
  leave the gate alone.
- **All friction figures are computed from stored entry quotes**, not from
  realised fills. They describe what the spread was, not what was paid.
- One regime, 26 trading days, SPY's worst drawdown -4.49%.

## What promotion would authorise

A config-only change to `auto_log.max_friction_to_credit` (currently 0.25).
Nothing here authorises real money or touches any selection rule.
