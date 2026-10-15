# Index vs single-name Bull Puts in the shadow corpus — REGISTRATION

**Frozen 2026-10-15, before any outcome (`pnl_pct`, `exit_price`, `exit_reason`)
of this population was queried in this session.** Task 2 of the 10-15 strategic
reset.

## Provenance

What was looked at before freezing. These are **entry fields and bookkeeping
only**:

- Task 1 (same day) read the EV gate's code and ENTRY fields of `candidates`.
  The finding: the spread EV prices both legs at ONE flat realised vol, so put
  skew reads as the seller overpaying for the long wing. On the `CREDIT SPREADS`
  board the negative-EV gate passes 5.2% of index Bull Puts and 13.2% of
  single-name ones. That finding motivates this test.
- Counts, symbol lists, scan-date ranges and `status` closure rates per arm
  (feasibility table below).

No earlier session had run an index-vs-single-name outcome cut of this corpus
either. The adjacent forward gate test (`project_forward_gate_test_20261015`)
measured passed-vs-refused **within** (symbol, expiry, day) cells, pooled over
symbols. It reported nothing split by index. It still shapes priors: the gate
separates Bull Puts at +0.18 AUC forward, but that result is fragile.

## Question

The live Bull Put book is 100% single-name. The negative-EV gate refuses most
index Bull Puts, and task 1 says it does so partly because of skew.
**Do the index Bull Puts it refuses do better, on capital at risk, than the
single-name Bull Puts it passes?** If yes, the gate is pointed at the wrong
population.

## Population

Rows of `candidates` (alias c) joined to `candidate_positions` (alias p) on
`(scan_id, board, contract_key)`, where:

- `c.strategy_name = 'Bull Put'` and `c.board = 'CREDIT SPREADS'`. Rows are
  duplicated verbatim on `TOP PICK`; one board only.
- Arm: `c.gate_passed = 1` (**passed**), or `c.refused_by = 'negative_ev'`
  (**refused**). Friction refusals and every other reason are out: they never
  reach the EV gate, so they say nothing about it.
- `p.status = 'CLOSED'` and `p.pnl_pct IS NOT NULL`.
- Scan date (`substr(c.scan_id,1,10)`) **≤ 2026-10-08**. At this cutoff every
  arm is ≥ 99.5% closed, so differential censoring is removed rather than
  modelled. After it, closure is 0-45% and differs by arm.
- `p.exit_date NOT IN ('2026-08-29','2026-08-30')`. These are the known
  weekend-quote artifacts, never repaired.

Groups:

| group | symbols |
|---|---|
| **index** | SPY, QQQ, IWM, DIA, VOO, VTI (broad US equity index ETFs) |
| other ETF | GLD, SLV, TLT, IEF, EEM, XL* — **excluded from the primary**; tertiary only |
| **single** | everything else |

## Outcome — capital at risk, never a raw mean of `pnl_pct`

```
e = p.entry_price                       (signed credit at the `limit` fill; > 0)
w = features_json.spread_width
R = pnl_pct * e / (w - e)               (P&L per dollar of max loss)
```

- Rows are dropped unless `0 < e < w`.
- R is clipped to `[-1, e/(w-e)]`, the physical bounds of a vertical. The
  clipped count is reported.

**What R measures:** the sim exits at the first mark on or after day 3 whenever
DTE ≤ 21 (`time_exit_dte=21`, `min_days_held=3`), or earlier on TP +50% / stop
−100% of premium. Most of this population enters under 21 DTE (index median
≈ 10). **So R is mostly a ~3-trading-day return, at mid, on capital at risk.**
It is not a hold-to-expiry return.

The exit is marked at **mid**, with no exit cost. Index spreads are tighter, so
this omission **flatters the single-name arm**. That bias runs against the
hypothesis.

## Units and estimator

| level | key | role |
|---|---|---|
| row | `(scan_id, board, contract_key)` | decision instance, never the unit |
| arm-day | `(arm, scan_date)` | mean over SYMBOLS of each symbol's mean R that day (symbol-equal weight) |
| pair | `scan_date` | days where both arms of a contrast have ≥ 1 row |
| cluster | ISO week of scan_date | resampling unit |

The two arms of the primary hold different symbols, so they can't be matched
within a symbol. They are paired by **day**, because the market's move is the
dominant shared shock for a long-delta position. Clusters are **weeks**: days
within a week share overlapping 3-day holds. Index has only ~4 effective
symbols, so a symbol bootstrap of the index arm is not viable.

**Primary statistic:**
`D = mean over paired days of [R̄(index, refused, d) − R̄(single, passed, d)]`.

**CI:** 10,000 week-cluster bootstrap resamples, percentile 95%, seed 20261015.

## Decision rule (primary)

| result | verdict |
|---|---|
| CI lower > 0 | **misdirected** — refused index beats passed single-name |
| CI upper < 0 | **gate right** — passed single-name beats refused index |
| otherwise | **inconclusive** |

Reported alongside, with no override: each arm's own level, mean R with the
same week bootstrap. If both are negative, "misdirected" means "loses less", and
it will be said that way.

**Authority.** Even "misdirected" authorises **no gate change** on its own: ~8
weekly clusters, one regime, a 3-day horizon. It authorises a forward
re-run on scans after 2026-10-08 (around 2026-11-15) and a gate redesign
proposal (skew-aware real-world distribution). Those are inputs to the
2027-01-15 decision rule (task 5).

## Secondary (no decision authority)

- **S1, skew mechanism.** For each row, compute the skew penalty from entry
  fields:
  - `(flat-IV value − credit) / w`
  - flat-IV value = BS put-spread value with both legs at the short leg's
    `entry_iv`, r = 4%, q = 0, S backed out from the short leg's mid at that IV.
  - Quintile edges are taken on the pooled index + single primary population.

  Report symbol-equal-weighted mean R by quintile in the **refused** arm, and
  `Q5 − Q1` with a week bootstrap CI.
  - If the penalty prices risk that really exists, Q5 is below Q1.
  - If it is spurious, there is no decline.

  Repeated within single-name only, to strip the index/single confound.
- **S2, gate within index.** Index passed vs index refused, the same day-paired
  estimator. Expected: insufficient (3 symbols, 16 days).
- **S3, all four arms.** Levels: index/single × passed/refused.

## Sensitivities (no decision authority)

1. Drop `exit_date = '2026-10-12'` (the Sunday 10-11 repair batch).
2. Row-weighted instead of symbol-equal.
3. Include the 08-29/30 exits.
4. Other-ETF group in place of index.
5. Mature-only: scan dates ≤ 2026-10-01.

## Feasibility at freeze (counts only)

Scans ≤ 2026-10-08, before the exit-date and `0<e<w` filters:

| group | arm | rows | closure | scan days |
|---|---|---|---|---|
| index | refused | 5,504 | 1.000 | 35 |
| index | passed | 194 | 1.000 | 16 |
| single | refused | 12,756 | 0.997 | 35 |
| single | passed | 1,436 | 0.995 | 35 |
| other ETF | refused / passed | 1,118 / 400 | 1.000 | 34 / 30 |

DB at freeze: `candidate_positions` 610,801 rows, latest exit 2026-10-15,
sha256 of `data/candidates.db` begins `247b714dbea31d9f`. The DB is live, so
the analysis reports its own fingerprint.

## Prior, stated before running

Weak expectation of **inconclusive**. Eight weeks of a mostly rising market
favour every long-delta arm, and 3-day returns at mid are noisy. Moderate
expectation that S1 shows **no** decline from Q1 to Q5. Index skew is
documented as overpriced (the skew premium) in the literature, which would make
the penalty spurious.

Estimator: `scripts/index_vs_single_bull_put.py`.
