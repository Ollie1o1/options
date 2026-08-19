# Candidate Forward Marks — design

Date drafted: 2026-08-19
Status: **DRAFT, awaiting operator review.** No implementation yet.
Scope: sub-project 2 of the ranker-validation sequence. Gives the recorded
candidates — the refused ones especially — outcomes. It does not analyse them,
does not produce a verdict, and changes no gate, ranker or entry decision.

Depends on sub-project 1 (`docs/CANDIDATE_RECORD_SPEC.md`, merged as PR #49).

---

## 1. What is missing

`data/candidates.db` now records every pre-gate candidate with its refusal
reason, rank position and taken flag. On the first live scan: 786 rows, 424
refusals, EV on 776 of them.

**None of them have outcomes.** A refused candidate is a decision with no
recorded consequence, which is the same hole the ledger has — the ledger just
hides it better by containing only what was taken. Until the refused population
is marked forward under the same rules the book uses, the questions in §8 of the
sub-project 1 spec stay unanswerable.

## 2. The constraint that shaped this

`data/chain_archive.db` was the obvious source and cannot do the job.
`config.json → data_archive.symbols` is a **hardcoded 15-symbol list**, while
scans cover 78:

| | |
|---|---|
| chain_archive symbols | 15 |
| candidate symbols | 78 |
| overlap | **14** |
| candidate rows reachable | **41.5%** |

And the reachable subset is mega-cap tech plus index ETFs — a biased sample,
which would reintroduce exactly the selection bias PR #50 removed from the
entry path. Widening the archive to whole chains for 78 symbols is roughly 5x
its current growth with the disk already at 92%.

The way out is that whole chains are not needed. The contracts requiring marks
are known exactly, because they are in `candidates.db`: 393 distinct contracts
across 157 distinct `(symbol, expiration)` pairs over three scans. That is a
targeted fetch, and `PaperManager.update_shadow_marks` already established the
batching — one chain request per `(ticker, expiration)`, shared by every row on
that pair.

**Marks come from `PaperManager._fetch_chain_quotes`**, the same call that marks
real shadowed trades. Same basis, so a candidate's outcome and a real trade's
outcome are comparable rather than merely similar.

## 3. One amendment to sub-project 1

**`candidates.strategy_name` is NULL on 100% of recorded rows.** Discovery-board
rows carry `type = 'call' | 'put'`, and `_strategy_of` correctly refuses to read
an option type as a strategy name. Exit rules are per-family, so marking cannot
proceed without one.

The mode is embedded in `scan_id` (`...|Discovery scan|...`), but parsing a
composite key to recover a field is the kind of thing that works until someone
changes the separator.

So: **add a `mode TEXT` column to `candidates`**, populated from the scan
context, which already receives `mode` in `candidate_record.scan(mode)`. The
marker derives the strategy with
`trade_analysis.strategy_label_for_mode(mode, opt_type)` — the same function
the auto-log path uses, not a second copy of the mapping — then maps that
strategy onto the family `config.exit_rules` is keyed by:

| strategy | `config.exit_rules` family |
|---|---|
| Long Call, Long Put | `long_option` |
| Bull Put, Bear Call, Iron Condor | `spread` |
| Short Put, Short Call | `short_premium` — **not supported in v1, §5.5** |

Rows recorded before this column exists carry `mode IS NULL`. They get **no
position at all**, not an `UNMARKABLE` one: a row with no derivable family is
not a decision this can simulate, and filling the table with thousands of inert
placeholders would bury the rows that mean something. They are not backfilled
(§7).

## 4. Schema

Two new tables in `data/candidates.db`. Marks and outcomes are different kinds
of thing and are kept apart: a mark is an observed fact about a contract, a
position is a derived simulation of a decision.

```sql
CREATE TABLE candidate_marks (
  contract_key TEXT NOT NULL,
  mark_date    TEXT NOT NULL,
  bid REAL, ask REAL, mid REAL,
  source       TEXT NOT NULL,        -- 'live_quote'
  PRIMARY KEY (contract_key, mark_date)
);

CREATE TABLE candidate_positions (
  scan_id      TEXT NOT NULL,
  board        TEXT NOT NULL,
  contract_key TEXT NOT NULL,
  family       TEXT,                 -- long_option | spread | short_premium
  entry_date   TEXT NOT NULL,
  entry_price  REAL,
  status       TEXT NOT NULL,        -- OPEN | CLOSED | UNMARKABLE | UNSUPPORTED
  exit_date    TEXT,
  exit_price   REAL,
  exit_reason  TEXT,                 -- take_profit | stop_loss | time_exit | expired
  pnl_pct      REAL,
  PRIMARY KEY (scan_id, board, contract_key)
);
CREATE INDEX idx_pos_status ON candidate_positions(status);
CREATE INDEX idx_pos_contract ON candidate_positions(contract_key);
```

One position per `(scan_id, board, contract_key)` — one per decision instance,
matching the primary key of `candidates` itself. The same contract recorded on
three consecutive scans is three positions at three entry prices with three rank
positions, sharing one stream of marks.

**There is no `pnl_usd`.** Sizing a position that was never taken means
inventing a sizing rule, and `pnl_pct` is what the ledger comparison needs.

**NULL means not recorded, never zero** — the same rule as the ledger.

## 5. The daily run

One entry point, `mark_candidates(today)`, on the existing maintenance
heartbeat beside the shadow-mark job.

1. **Open positions** — every `candidates` row with no position yet gets one.
   `entry_price` is the `limit` fill computed by `execution_truth.structure_fill`
   from the recorded `bid`/`ask`. Real entries record `fill_policy='limit'`
   (194 rows; the 810 NULLs are pre-policy legacy), so a candidate priced any
   other way would not be comparable to the book. A row that cannot be priced
   opens as `UNMARKABLE` rather than at a guessed price.
2. **Mark** — group open positions by `(symbol, expiration)`, one
   `_fetch_chain_quotes` call per pair, write one `candidate_marks` row per
   contract at mid. Mid, because `update_shadow_marks` marks the real book at
   mid.

   `candidate_positions` deliberately does not duplicate `symbol`,
   `expiration`, `strike` or `opt_type` — they are joined from `candidates` on
   `(scan_id, board, contract_key)`, which is that table's primary key. The
   alternative is parsing them back out of `contract_key`, and a key is an
   identity, not a storage format. One duplicated-and-drifting copy of a
   contract's identity is exactly the class of defect this project keeps
   finding.
3. **Resolve** — apply the exit rules **read from `config.json → exit_rules`**,
   never copied into source: `time_exit_dte 21`, `min_days_held 3`, and the
   per-family take-profit/stop blocks (`long_option.take_profit 1.0` /
   `stop_loss -0.5`, `spread.take_profit 0.5` / `stop_loss -1.0`, and the
   DTE-banded `short_premium` block). A divergence between these rules and the
   book's makes every counterfactual incomparable, which is why they are read
   rather than restated.
4. **Expire** — a position past its expiration with no exit closes `expired` at
   its final mark.

### 5.5 short_premium is not supported in v1, and says so

The `short_premium` block does not decide on price alone. `stop_loss_on_strike_breach`
needs the **spot**, and `stop_loss_delta_multiple` needs the position's current
**delta** — neither of which a bid/ask mark carries. Applying only that block's
price rule would produce a number that looks like a short-premium outcome and
is not one: the same defect shape as every measurement bug this repo has
removed.

So a `short_premium` position opens with `status='UNSUPPORTED'` and
`exit_reason='needs_spot_and_delta'`, and is never marked or resolved.

The cost is small and bounded. `auto_log.allowed_strategies` is **Long Call and
Bull Put** — `long_option` and `spread` — so every family that can currently
reach the book is fully supported. Short Put and Short Call are switched off on
their own evidence. Supporting them means storing spot and delta on each mark;
that is a follow-up, deliberately not smuggled into v1.

Failure-safe and failure-visible, the same discipline as sub-project 1: never
raises into maintenance, every failure counted and written to `recorder_errors`,
and a `--health` line that is loud at zero. A marker that returns cleanly and
writes nothing is precisely how four months of shadow-mark data went missing.

## 6. Boundaries

- **Never touches `paper_trades.db`.** It is opened read-only or not at all.
- **Never influences a scan, a gate, a ranker or an entry.** Nothing in the
  scan path reads these tables.
- **Never invents a price.** No mid-from-last, no synthetic quote. Unquotable
  is a recorded state.

## 7. Explicitly out of scope

- **Any analysis or verdict.** Sub-project 3. The test, its power analysis and
  its acceptance bar must be written before the data is large enough to peek at,
  and must pair within `(scan_day, strategy)` cells and cluster on
  `contract_key` — the book already shows ICC ~0.08–0.11 and a design effect of
  1.23–1.27 from batch entries, and the whole-book carry correlation (+0.104)
  reverses within strategy (Iron Condor −0.282).
- **Backfilling the 2,354 rows already recorded.** They predate the `mode`
  column, so their family would have to be inferred. Seeding the dataset with
  guessed families to gain three days of history is a bad trade.
- **Widening `chain_archive`.** Rejected in §2.
- **Crypto candidates.** The crypto auto-log is off (2026-08-18) and its
  monoculture defect is unrelated.

## 8. Testing

Against a temp database, with the quote fetcher **injected** so no test touches
the network. No test names the real ledger or the real `candidates.db`.

1. A candidate becomes exactly one position per `(scan_id, board, contract_key)`,
   and a second run does not duplicate it.
2. The same contract on three scans produces three positions sharing one
   mark stream.
3. `entry_price` equals the `limit` fill of the recorded bid/ask — asserted
   against `execution_truth`, not a hand-computed constant.
4. Take-profit, stop-loss, time-exit and expiry each fire, **with thresholds
   read from a fixture config**, and a test asserts the code reads
   `config.exit_rules` rather than carrying its own numbers.
5. `min_days_held` suppresses an exit that would otherwise fire on day 1.
6. An unquotable candidate opens `UNMARKABLE` and is never marked or resolved.
7. A quote-fetch failure for one `(symbol, expiration)` does not stop the
   others, and increments the error counter.
8. The family is derived through `strategy_label_for_mode`; a `Short Put` opens
   `UNSUPPORTED` and is never marked; a row with `mode IS NULL` produces **no
   position row at all** rather than defaulting to a family.
8b. `candidate_positions` carries no `symbol`/`expiration`/`strike` column —
   asserted on the schema, so the identity cannot be duplicated and drift.
9. `--health` reports marks in the last 7 days and is loud at zero.
10. **The run is driven end-to-end against a temp database with a stub fetcher
    and the resulting rows asserted** — a source grep is not a rendering test.

## 9. What this buys

Once it has run for a quarter, the refused population has outcomes and the
questions from sub-project 1 become ordinary queries — with the added property,
from PR #50, that the *entered* set is an unbiased draw from the survivors. The
comparison "taken vs refused, same board, same day" becomes valid rather than
confounded by the selection rule that produced it.
