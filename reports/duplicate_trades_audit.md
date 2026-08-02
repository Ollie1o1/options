# Duplicate-trade audit — candidate rows in the paper ledger

> ## RULED 2026-08-01 — 17 of 18 groups are NOT duplicates
>
> **Verdict: one true double-log in 882 rows — group 9 (WFC, entry_id 91).**
> It is marked `duplicate_of=90` and excluded from every cohort and from the
> published track record. The other 17 groups stand as genuine trades.
>
> **The test that settled it: did the flagged day log anything else?** The
> failure mode this audit was built to catch is the catch-up replay documented
> at `auto_log.dedup_window_days` — the feeder re-logging the PREVIOUS day's set
> the next morning. A replay would make the repeats most of what that day
> contains. They are not:
>
> | day | logged | flagged | fresh |
> |---|---|---|---|
> | 2026-04-18 | 15 | 6 | 9 |
> | 2026-04-19 | 10 | 6 | 4 |
> | 2026-06-08 | 33 | 5 | 28 |
> | 2026-06-09 | 16 | 5 | 11 |
> | 2026-07-09 | 24 | 1 | 23 |
>
> Every flagged day carries a full batch of unrelated fresh trades. On
> 2026-06-09 only 5 of the previous day's 33 reappear. That is a normal scan in
> which a deterministic screener re-picked a few of the same contracts, not a
> replay of the previous day.
>
> **Why they look identical anyway.** `entry_price` and `capital_at_risk` are
> bit-identical across the pairs because the quoted bid/ask had not moved, so
> the mid recomputes to the same float. `entry_iv` differs in every one of those
> groups — one day less to expiry re-prices the vol off an unchanged quote. The
> apparent identity is a stale market, not a stale log.
>
> **Group 9 is the exception, and the reason is structural.** Its two rows were
> entered on the SAME day (2026-04-26, adjacent entry_ids 90/91) with
> bit-identical `entry_iv`. The screener ran once that day; one snapshot cannot
> produce two independent decisions. It is the only same-day, same-contract,
> bit-identical-IV pair in the entire ledger.
>
> **Impact is smaller than the headline suggested.** The $2,959.10 figure below
> assumed all 18 groups were duplicates. The ruling removes **$64.70**, on a
> Short Put with $7,598 of collateral — already above the $4,000 affordability
> cap, so it was never in the short-premium gate cohort or the affordable track
> record. No gate verdict moves.
>
> Reversible: `python scripts/rule_duplicate_trades.py --undo`.

- Generated: 2026-07-31 13:49:20
- Database: `paper_trades.db` (opened read-only; this audit never writes)
- Rows scanned: **882**
- Match key: same `(ticker, strategy_name, strike, expiration)` with `entry_price` equal to the cent, entered within **3 days** of each other

## Summary

- Candidate duplicate groups: **18**
- Rows involved: **38**
- Excess rows (every row past the first in its group): **20**
- P&L carried by those excess rows: **$2,959.10** (the amount double-counted if the groups are true duplicates)

**Nothing has been deleted or edited.** A match is a candidate, not a verdict: a contract legitimately re-entered at the same price a day later looks identical at this resolution. The operator rules on each group; only then does anything change.

A group whose rows also share an exit price and exit date is marked `identical exits` — that is the strongest tell, because two genuinely separate positions would have to be closed by the same sweep at the same mark to look that way.

## Candidates

### 1. DIA Bear Call $529 exp 2026-07-24 @ $0.43

2 rows between 2026-07-09 and 2026-07-10 (span 1d) — 2 closed, 0 open. Excess P&L $-23.60.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 574 | 2026-07-09 | CLOSED | $0.43 | $0.43 | 2026-07-12 | -52.6% | $-22.60 | 1 | baseline |
| 598 | 2026-07-10 | CLOSED | $0.43 | $0.44 | 2026-07-13 | -54.9% | $-23.60 | 1 | baseline |

### 2. ABBV Long Call $260 exp 2026-08-21 @ $8.30 — **identical exits**

2 rows between 2026-07-07 and 2026-07-08 (span 1d) — 2 closed, 0 open. Excess P&L $-506.10.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 520 | 2026-07-07 | CLOSED | $8.30 | $3.75 | 2026-07-14 | -61.0% | $-506.10 | 1 | baseline |
| 540 | 2026-07-08 | CLOSED | $8.30 | $3.75 | 2026-07-14 | -61.0% | $-506.10 | 1 | baseline |

### 3. SPY Bear Call $745 exp 2026-07-17 @ $0.49

2 rows between 2026-06-24 and 2026-06-26 (span 2d) — 2 closed, 0 open. Excess P&L $-48.10.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 475 | 2026-06-24 | CLOSED | $0.49 | $1.81 | 2026-06-24 | -104.1% | $-51.00 | 1 | baseline |
| 505 | 2026-06-26 | CLOSED | $0.49 | $0.74 | 2026-07-07 | -99.2% | $-48.10 | 1 | baseline |

### 4. SPY Bear Call $745 exp 2026-06-18 @ $0.46 — **identical exits**

2 rows between 2026-06-08 and 2026-06-09 (span 1d) — 2 closed, 0 open. Excess P&L $45.50.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 357 | 2026-06-08 | CLOSED | $0.46 | $0.00 | 2026-06-10 | +100.0% | $45.50 | 1 | baseline |
| 361 | 2026-06-09 | CLOSED | $0.46 | $0.00 | 2026-06-10 | +100.0% | $45.50 | 1 | baseline |

### 5. QQQ Bull Put $705 exp 2026-06-18 @ $0.56 — **identical exits**

2 rows between 2026-06-08 and 2026-06-09 (span 1d) — 2 closed, 0 open. Excess P&L $52.40.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 353 | 2026-06-08 | CLOSED | $0.56 | $0.00 | 2026-06-09 | +93.6% | $52.40 | 1 | baseline |
| 360 | 2026-06-09 | CLOSED | $0.56 | $0.00 | 2026-06-09 | +93.6% | $52.40 | 1 | baseline |

### 6. ORCL Bull Put $200 exp 2026-06-18 @ $1.22

2 rows between 2026-06-08 and 2026-06-09 (span 1d) — 2 closed, 0 open. Excess P&L $-127.50.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 358 | 2026-06-08 | CLOSED | $1.22 | $0.70 | 2026-06-11 | +24.4% | $29.90 | 1 | baseline |
| 362 | 2026-06-09 | CLOSED | $1.22 | $3.05 | 2026-06-11 | -104.1% | $-127.50 | 1 | baseline |

### 7. NVDA Bull Put $205 exp 2026-06-26 @ $1.77

2 rows between 2026-06-08 and 2026-06-09 (span 1d) — 2 closed, 0 open. Excess P&L $-9.10.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 359 | 2026-06-08 | CLOSED | $1.77 | $2.12 | 2026-06-11 | -32.2% | $-57.10 | 1 | baseline |
| 364 | 2026-06-09 | CLOSED | $1.77 | $1.64 | 2026-06-15 | -5.1% | $-9.10 | 1 | baseline |

### 8. INTC Bull Put $106 exp 2026-07-02 @ $0.50

2 rows between 2026-06-08 and 2026-06-09 (span 1d) — 2 closed, 0 open. Excess P&L $50.00.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 330 | 2026-06-08 | CLOSED | $0.50 | $0.00 | 2026-06-08 | +64.8% | $32.40 | 1 | baseline |
| 363 | 2026-06-09 | CLOSED | $0.50 | $0.00 | 2026-06-09 | +100.0% | $50.00 | 1 | baseline |

### 9. WFC Short Put $77.5 exp 2026-05-15 @ $1.52 — **identical exits**

2 rows between 2026-04-26 and 2026-04-26 (span 0d) — 2 closed, 0 open. Excess P&L $64.70.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 90 | 2026-04-26 | CLOSED | $1.52 | $0.76 | 2026-04-28 | +42.6% | $64.70 | 1 | — |
| 91 | 2026-04-26 | CLOSED | $1.52 | $0.76 | 2026-04-28 | +42.6% | $64.70 | 1 | — |

### 10. T Long Put $26 exp 2026-05-15 @ $0.53 — **identical exits**

2 rows between 2026-04-24 and 2026-04-26 (span 2d) — 2 closed, 0 open. Excess P&L $-16.30.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 66 | 2026-04-24 | CLOSED | $0.53 | $0.48 | 2026-04-27 | -30.8% | $-16.30 | 1 | — |
| 83 | 2026-04-26 | CLOSED | $0.53 | $0.48 | 2026-04-27 | -30.8% | $-16.30 | 1 | — |

### 11. ORCL Long Call $180 exp 2026-05-15 @ $6.55

3 rows between 2026-04-24 and 2026-04-26 (span 2d) — 3 closed, 0 open. Excess P&L $-715.20.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 71 | 2026-04-24 | CLOSED | $6.55 | $5.45 | 2026-04-27 | -23.0% | $-150.60 | 1 | — |
| 76 | 2026-04-25 | CLOSED | $6.55 | $3.38 | 2026-04-28 | -54.6% | $-357.60 | 1 | — |
| 84 | 2026-04-26 | CLOSED | $6.55 | $3.38 | 2026-04-28 | -54.6% | $-357.60 | 1 | — |

### 12. GS Long Call $940 exp 2026-05-15 @ $22.25

3 rows between 2026-04-24 and 2026-04-26 (span 2d) — 3 closed, 0 open. Excess P&L $-1,635.60.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 67 | 2026-04-24 | CLOSED | $22.25 | $22.00 | 2026-04-27 | -5.7% | $-126.30 | 1 | — |
| 78 | 2026-04-25 | CLOSED | $22.25 | $19.95 | 2026-04-28 | -14.9% | $-331.30 | 1 | — |
| 85 | 2026-04-26 | CLOSED | $22.25 | $10.22 | 2026-04-29 | -58.6% | $-1,304.30 | 1 | — |

### 13. ORCL Long Call $180 exp 2026-05-01 @ $5.35

2 rows between 2026-04-18 and 2026-04-19 (span 1d) — 2 closed, 0 open. Excess P&L $598.60.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 2026-04-18 | CLOSED | $5.35 | $6.00 | 2026-04-21 | +5.9% | $31.60 | 1 | — |
| 22 | 2026-04-19 | CLOSED | $5.35 | $11.67 | 2026-04-23 | +111.9% | $598.60 | 1 | — |

### 14. ORCL Long Call $180 exp 2026-05-08 @ $6.90

2 rows between 2026-04-18 and 2026-04-19 (span 1d) — 2 closed, 0 open. Excess P&L $602.30.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 2026-04-18 | CLOSED | $6.90 | $7.70 | 2026-04-21 | +5.4% | $37.30 | 1 | — |
| 23 | 2026-04-19 | CLOSED | $6.90 | $13.35 | 2026-04-23 | +87.3% | $602.30 | 1 | — |

### 15. ORCL Long Put $170 exp 2026-05-15 @ $7.45 — **identical exits**

2 rows between 2026-04-18 and 2026-04-19 (span 1d) — 2 closed, 0 open. Excess P&L $-106.00.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 14 | 2026-04-18 | CLOSED | $7.45 | $6.85 | 2026-04-24 | -14.2% | $-106.00 | 1 | — |
| 24 | 2026-04-19 | CLOSED | $7.45 | $6.85 | 2026-04-24 | -14.2% | $-106.00 | 1 | — |

### 16. MU Long Call $480 exp 2026-05-15 @ $25.05 — **identical exits**

2 rows between 2026-04-18 and 2026-04-19 (span 1d) — 2 closed, 0 open. Excess P&L $1,603.70.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2026-04-18 | CLOSED | $25.05 | $42.10 | 2026-04-23 | +64.0% | $1,603.70 | 1 | — |
| 28 | 2026-04-19 | CLOSED | $25.05 | $42.10 | 2026-04-23 | +64.0% | $1,603.70 | 1 | — |

### 17. MU Long Call $470 exp 2026-05-01 @ $18.80

2 rows between 2026-04-18 and 2026-04-19 (span 1d) — 2 closed, 0 open. Excess P&L $1,570.70.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 2026-04-18 | CLOSED | $18.80 | $14.61 | 2026-04-21 | -27.7% | $-520.30 | 1 | — |
| 26 | 2026-04-19 | CLOSED | $18.80 | $35.52 | 2026-04-23 | +83.5% | $1,570.70 | 1 | — |

### 18. MU Long Call $475 exp 2026-05-08 @ $22.25

2 rows between 2026-04-18 and 2026-04-19 (span 1d) — 2 closed, 0 open. Excess P&L $1,558.70.

| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |
|---|---|---|---|---|---|---|---|---|---|
| 21 | 2026-04-18 | CLOSED | $22.25 | $17.40 | 2026-04-21 | -26.4% | $-586.30 | 1 | — |
| 27 | 2026-04-19 | CLOSED | $22.25 | $38.85 | 2026-04-23 | +70.1% | $1,558.70 | 1 | — |

## What to do with this

1. ~~Rule on each group~~ — **done 2026-08-01, see the ruling at the top.**
2. True duplicates stay in the ledger — the record is what was traded, and rewriting it silently is worse than the double-count it fixes. The ruling therefore MARKS (`duplicate_of`, schema v17) rather than deletes: cohort and track-record queries exclude marked rows, the ledger keeps them.
3. The auto-log dedup guard (`auto_log.dedup_window_days`) refuses new entries matching the same key inside the window, so this list should stop growing from the automated feeders regardless of how the existing rows are ruled on.

### A note for the next run of this audit

The match key that produced these 18 groups has a known false-positive mode: an
unchanged bid/ask reproduces the same mid to the last bit, so a deterministic
screener re-picking a contract on a quiet day is indistinguishable from a
double-log on `(ticker, strategy, strike, expiration, entry_price)` alone. Two
cheap discriminators separate them, and a future version of this audit should
compute both rather than leaving the reader to:

- **Same-day, bit-identical `entry_iv`** — one snapshot, so at most one decision.
  This is the only pattern that is conclusive on its own.
- **Batch share of the flagged day** — what fraction of that day's log is
  flagged. A replay approaches 100%; the groups here ran 15-40%.

