# Pre-registered ranker test — REGISTRATION

**Frozen 2026-08-19 15:42 UTC. Immutable.**

Written before any outcome existed. Every term below was fixed in advance;
that is the only thing that makes the eventual result mean anything.

## Hypothesis

**H1.** Among gate survivors, `ev_net` predicts `pnl_pct`.

Statistic: rank IC on ranks demeaned within `(entry_date, strategy)` cells.
Interval: percentile cluster bootstrap resampling `contract_key`.

## Parameters

```
feature: ev_net
outcome: pnl_pct
cells: entry_date, strategy
cluster: contract_key
target_ic: 0.08
power: 0.8
alpha: 0.05
n_boot: 10000
seed: 20260819
min_cell_rows: 3
assumed_icc: 0.11
mean_cluster_size_at_registration: 2.4821
design_effect: 1.1630
n_star_effective: 1224.2
n_star_nominal: 1424
deadline: 2026-11-19
```

## Decision rule

One look, at `n_star_nominal` closed survivor positions or `deadline`,
whichever comes first.

| outcome | condition | consequence |
|---|---|---|
| PASS | 95% CI lower bound > 0 | ranking may be *proposed* again, as its own change |
| FAIL | CI contains 0 | refuse-don't-rank stands; entries stay random |
| INVERTED | CI entirely below 0 | treated as FAIL; motivates a NEW registration |
| UNDERPOWERED | n < n_star_nominal at deadline | treated as FAIL |

There is no EXTEND state — that is the trap that let the LC gate run forever.
PASS does **not** authorise real money.

`n_star` is powered FOR `target_ic`, so "CI lower bound > 0" and "the effect is
meaningful" cannot disagree. That is exactly where the LC gate failed:
`IC >= 0.08 AND p < 0.05` at a trigger of n >= 50 made the 0.08 decorative,
because detecting 0.08 needs n around 1224.

## Secondary, no decision authority

`quality_score`, `carry`, `delta` — reported beside the primary. A secondary
result may motivate a new registration; it can never move this one.

## Guards

- Negative control: outcome shuffled within cells must return null.
- Sign consistency across halves, split at the median `entry_date`. Reported.
- Effective n reported, never nominal.

## Result

Run 2026-08-27. One look, as registered.

```
decision: FAIL
n: 2137
rank_ic: -0.09753975752667834
ci_low: -0.20330892835792652
ci_high: 0.012209833547679834
design_effect: 3.1078172712059975
effective_n: 687.620864907135
negative_control_mean: -0.0017945803694571236
negative_control_p95_abs: 0.052002531437921114
half1_ic: 0.1179401757532451
half2_ic: -0.13129115823965107
secondary_quality_score_ic: 0.06407421261521777
secondary_carry_ic: 0.15186578573640974
secondary_delta_ic: -0.14762395362725333
```
 `scripts/prereg_ranker_test.py` writes here, once.
