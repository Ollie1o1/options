# Candidate Record — design

Date drafted: 2026-08-18
Status: **DRAFT, awaiting operator review.** No implementation yet.
Scope: sub-project 1 of the ranker-validation sequence. Records candidates.
Does not mark them forward, does not analyse them, does not change any gate,
ranker, or entry decision.

---

## 1. Why

The ranking key that selects real entries has an outcome sample of **n = 2**.

`options_screener.py:6731` (structures) and `~6895` (single legs) rank a board
with `rank_by_verdict` → `candidate_verdict.rank`, cut it with `.head(_top_n)`,
and write the survivors to the ledger. The sort key is
`(verdict.passed, ev_per_contract, -round_trip_pct, quality_score)`
(`candidate_verdict.py:230`). `ev_per_contract` is persisted as
`trades.entry_ev_net`, and that column is populated on 26 rows in total, all
since 2026-08-11, of which **2 are closed with a `pnl_pct`**.

Measured on the 881 closed, non-duplicate trades (2026-08-18):

| ordering | where used | Spearman vs `pnl_pct` |
|---|---|---|
| `quality_score` | tie-break only | n=881, rho **−0.032**, p=0.345 |
| carry `\|theta\|/premium` | board display order, `pick_ranking._carry_key` | n=842, rho **+0.104**, p=0.002 |
| `ev_per_contract` | **auto-log entry selection** | **n=2** |

The carry row is a warning, not a result. Whole-book it is significant; within
strategy the sign flips — Iron Condor **−0.282** (p=0.001), Long Put −0.070,
Long Call +0.084 (ns), Bull Put +0.172 (p=0.067). It is strategy mix, not
discrimination. Any future test of a ranker must be paired within
`(scan_day, strategy)` cells or it will reproduce exactly this artifact.

So the repo replaced a ranker *measured to be bad* with one that is
*unmeasured*. That is probably an improvement and is not evidence of one.

**And the ledger can never settle it.** The ledger contains only what the
ranker chose. Rank position is not recorded, and every row is a top-5 pick, so
in-ledger rank variance is near zero. There is no table of refused candidates:
`shadow_until` (207 rows) tracks trades already taken, after exit. Grading a
filter using only what it let through is not possible.

This spec builds the missing dataset.

---

## 2. What is recorded

A new standalone `data/candidates.db`. `paper_trades.db` stays at schema v22
and is not migrated: a research table whose shape will churn must not be able
to damage the book, and the join is cheap in pandas or via `ATTACH`.

```sql
CREATE TABLE candidates (
  scan_id        TEXT NOT NULL,   -- one scan invocation
  ts             TEXT NOT NULL,   -- ISO8601, when the row was recorded
  board          TEXT NOT NULL,   -- 'discover' | 'spreads' | 'condors' | 'top' | ...
  contract_key   TEXT NOT NULL,   -- deterministic candidate identity, §3
  symbol         TEXT,
  strategy_name  TEXT,
  expiration     TEXT,
  strike         REAL,
  opt_type       TEXT,

  bid REAL, ask REAL, premium REAL, theta REAL, delta REAL,

  ev_net REAL, ev_gross REAL, ev_cost REAL, ev_noise REAL,
  quality_score REAL, round_trip_pct REAL, friction_pct REAL,

  rank_pos       INTEGER,         -- position in the ranked frame, 1-based
  refused_by     TEXT,            -- gate key, NULL when it cleared
  gate_passed    INTEGER,         -- 1/0
  gating_failed  INTEGER NOT NULL DEFAULT 0,   -- §4
  auto_logged    INTEGER NOT NULL DEFAULT 0,
  entry_id       INTEGER,         -- FK into paper_trades.trades when taken

  features_json  TEXT,            -- the ~35 churning scorer columns, verbatim
  PRIMARY KEY (scan_id, board, contract_key)
);
CREATE INDEX idx_cand_ts       ON candidates(ts);
CREATE INDEX idx_cand_strategy ON candidates(strategy_name, ts);
CREATE INDEX idx_cand_logged   ON candidates(auto_logged) WHERE auto_logged = 1;

CREATE TABLE recorder_errors (
  ts TEXT NOT NULL, scan_id TEXT, board TEXT, traceback TEXT
);
```

Fixed columns carry identity and everything the gates and ranker actually
consume, so the analysis queries are indexed SQL. `features_json` carries the
tail that changes whenever a weight profile changes, so a new scorer is never a
migration. NULL keeps its ledger meaning: **not recorded**, never zero.

Volume: ~250 pre-gate rows per board, a few boards per scheduled scan, daily.
Order 1–2M rows/year, a few hundred MB. Retention is a documented prune query,
not a mechanism — YAGNI until the rows have been marked forward.

---

## 3. Interface

New module `src/candidate_record.py`. One purpose. It is imported by
`options_screener`; it imports nothing from `options_screener`, so there is no
cycle.

```python
scan(mode) -> ContextManager[str]         # sets the scan_id ContextVar
record_board(result, *, board) -> int     # pre-gate rows + refusal reasons
mark_ranked(board, ordered_rows) -> int   # rank_pos, upserted
mark_logged(board, row, entry_id) -> None # auto_logged=1 + FK to trades
contract_key(row) -> str
```

`scan_id` propagates through a `contextvars.ContextVar` set once per scan
rather than a parameter threaded through every call site. One scan_id spans
every board that scan produces — it is opened around the scan, not around a
board — and both hooks read it. That is what joins the gate record to the
auto-log record.

`board` is the label the caller already passes to `gate_and_report(df, board)`.
It is free text from the recorder's point of view; no enum is imposed, because
a new board must be recordable without a code change here.

`contract_key` is the deterministic identity: `symbol|expiration|opt_type|strike`
for a single leg, and `symbol|expiration|strategy|` plus every leg strike in a
fixed order for a spread or condor. It is also the key sub-project 2 will use
to find the contract in `data/chain_archive.db` for forward-marking, so it must
be stable across scans and must distinguish two structures that differ only in
one leg.

---

## 4. The two hooks, and one honesty fix

**Hook A — `gate_and_report`** (`options_screener.py:793`). It already holds
everything: `result.kept`, `result.refused` with its `refused_by` column, and
`result.reasons`. Recording happens from the `BoardResult`, never from a
re-derived frame.

`gate_board`'s failure-safe branch (`pick_ranking.py:319`) returns
`BoardResult(kept=df, refused=empty)` — **byte-identical to a board on which
every candidate legitimately cleared every gate.** Recording that as "all
passed" would write false data into the one table whose entire purpose is
settling a scientific question. So `BoardResult` gains one field,
`gating_failed: bool = False`, set `True` in that except branch and persisted on
every row of that board. This is the only change to existing behaviour in this
spec, and it changes no decision — only what is recorded about one.

**Hook B — the auto-log ranker** (`options_screener.py:6731` structures,
`~6895` single legs). Writes `rank_pos` across the whole ranked frame, then
`auto_logged` and `entry_id` for the rows actually inserted.

That frame is derived independently of the gated board — `_log_src` is picked
from `picks` / `_credit_spreads` / `_iron_condors` and is explicitly **not**
gated (see the comment at `options_screener.py:6718`, which is deliberate: G5
must not freeze its own training set). It is also filtered by the auto-log
allowlist and by the per-scan budget cap *before* the top-N cut, so rows are
removed for reasons the board never records.

Therefore hook B **upserts** on `(scan_id, board, contract_key)`: update when
the gate record exists, insert when it does not, and **count the inserts**.
That counter measures the board/auto-log divergence — the same structural split
that produced the "cleared the gates showed ungated rows" defect on 2026-08-18.
Allowlist and budget removals are recorded as `refused_by` values
`allowlist_drop` and `budget_displaced`, so a candidate that never reached the
top-N cut is distinguishable from one that reached it and lost.

---

## 5. Failure semantics

This project has already lost four months of shadow-mark data to a `NameError`
under a bare `except: pass`. `update_shadow_marks` returned cleanly and wrote
nothing, and nothing noticed until someone queried for rows that did not exist.
That is the exact failure this table is most exposed to.

So recording is **failure-safe and failure-visible**:

- every entry point is wrapped and never propagates into a scan; a broken
  recorder must not be able to stop a scan or change a pick
- every failure increments a counter, logs at WARNING, **and** writes the
  traceback to `recorder_errors`
- `maintenance --health` grows one line: candidate rows recorded in the last
  7 days and the `recorder_errors` count over the same window, with a red
  banner when the row count is zero

A silent zero must be loud. An exit code is not an outcome
(`feedback_verify_outcomes_not_exit_codes`), and neither is a clean return.

---

## 6. Tests

Against a `tmp_path` database. No test names the real ledger — `PaperManager`
migrates on init and `chdir` is not a sandbox.

1. Round-trip: record a board, read it back; kept and refused counts and every
   refusal reason match the `BoardResult`.
2. Refused rows are persisted with their reason. This is the whole dataset;
   a table of survivors only would be worthless.
3. The `gating_failed` path records `gating_failed=1`, **not** an all-passed
   board.
4. Upsert: hook B updates `rank_pos` on an existing gate row and does not
   duplicate it.
5. Auto-log rows with no gate record are inserted and counted in the
   divergence counter.
6. An unwritable database does not raise into the caller, increments the error
   counter, and writes a `recorder_errors` row.
7. `contract_key` is stable across repeat scans and distinguishes two condors
   differing in exactly one leg.
8. `features_json` round-trips the tail columns, and absent columns stay NULL
   rather than becoming 0.
9. **A real scan path runs against a temp database and rows land.** A source
   grep is not a rendering test, and an allowlist entry is a claim about
   behaviour that must be tested by running.

---

## 7. Explicitly out of scope

- **Forward-marking the refused** — sub-project 2. It depends on `contract_key`
  and `data/chain_archive.db` (490,048 rows, 15 symbols, 30 snapshot dates,
  2026-06-10 to 2026-08-18).
- **Any analysis or verdict.** The pre-registered test, its power analysis, and
  its acceptance bar are sub-project 3 and must be written before the data is
  large enough to peek at.
- **Any change to a gate, a ranker, a sort key, or an entry decision.** This
  spec observes. `gating_failed` is recorded, not acted on.
- **Retention enforcement.** A documented prune query only.

---

## 8. What this buys

After roughly one quarter of scans, the questions that are currently
unanswerable become ordinary queries:

- Does `rank_pos` predict realised return within `(scan_day, strategy)` cells?
- Do the top-5 taken beat the candidates refused on the same board, same day?
- Do the six gates in `pick_ranking.GATES` still hold out of sample, or was G5
  a three-month artifact?
- How often does the auto-log frame contain rows the board refused?

None of these can be asked of the ledger, because the ledger is the answer
sheet with every wrong answer erased.
