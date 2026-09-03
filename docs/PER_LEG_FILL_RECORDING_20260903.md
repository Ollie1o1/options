# Per-Leg Fill Recording — What Shipped and What It Unblocks (2026-09-03)

Task C item 1 of the 2026-09-02 work order: record per-leg bid/ask at entry
and exit for multi-leg structures, so a future session can reprice the 46%
of the closed book Task A had to refuse
(`docs/SINGLE_LEG_REPRICE_20260902.md`: `trades.entry_price` on a spread is
a net credit across legs, not any single leg's mid). This session does not
reprice anything and does not attempt any κ₀/κ₁ impact-model fitting, per
the work order's own instruction — there are zero real fill records to fit
against yet.

## What shipped

- **Schema v23** (`_SCHEMA_VERSION` 22 → 23): 24 nullable `REAL` columns on
  `trades` — 8 generic (`short`/`long`, Bull Put/Bear Call) and 16
  role-qualified (`short_put`/`long_put`/`short_call`/`long_call`, Iron
  Condor), each split `_entry`/`_exit`. NULL on every legacy row and every
  single-leg row, by construction — nothing backfills a value.
- **Entry-side threading**: `src/options_screener.py` already computed
  per-leg `short_bid`/`short_ask`/`long_bid`/`long_ask` (and the 8 condor
  equivalents) at scan time — the comment beside that code says exactly why
  ("so the candidate can be priced at what it would actually FILL for") —
  but the auto-log payload builder silently dropped them before the
  database write. `log_spread`/`log_iron_condor` now forward them into the
  new `_entry` columns; `log_trade`'s `INSERT` persists them.
- **Exit-side capture**: the live exit-enforcement loop
  (`PaperManager.update_positions`) already fetches a full per-leg bid/ask
  chain per `(ticker, expiration)` before collapsing each leg to a single
  mid/last/close/model mark. No new network call, no change to which exit
  rule fires or when — verified against the full
  `tests/test_mark_trustworthiness.py` suite (21 tests, all passing
  unchanged) plus a full `scripts/test.sh` run. A leg the loop could not
  chain-quote (model-marked, or a genuine chain gap) leaves its `_exit`
  columns NULL rather than a fabricated value; the `dte <= 0`
  expiry-settlement branch is untouched, since it settles at intrinsic from
  spot and never crosses a chain.

## Two of the work order's three named files were wrong

The work order said to touch `src/candidate_record.py`, `src/alloc/fills.py`,
and `src/structure/`. Verified before writing any code — the same "verify
these yourself first" discipline as Task A's `reprice_pnl_pct` correction:

- **`src/structure/`** is confirmed shelved and disconnected from the live
  ledger — its own `__init__.py` docstring states *"Display-only. Never
  writes to paper_trades.db and never touches the Phase-1 gate."*
- **`src/alloc/fills.py`** is part of the allocation backtester's simulated
  fill model over `data/dolt_options.db` historical quotes — unrelated to
  the real `paper_trades.db` ledger.
- **`src/candidate_record.py`** writes to `data/candidates.db`, a separate
  database from `paper_trades.db`; it already stores per-leg quotes for
  multi-leg candidates in `features_json` (used by Task B's gate RD study),
  but that was never the blocker here.

Nothing in this PR touches any of the three. The real blocker was
`src/options_screener.py`'s auto-log payload builder silently dropping
fields it already had in scope, and `src/paper_manager.py`'s exit loop
never persisting bid/ask it already fetched — neither named in the work
order.

## Coverage caveat — read before expecting data

**Only new trades logged or closed from this point forward carry this
data.** Nothing backfills historical rows. As of this session, the real
ledger (`paper_trades.db`) has **1 currently-open multi-leg position** and
6 open single-leg positions — so the first exit-side multi-leg data point
depends on that one position (or a new one) closing after this PR merges.
Entry-side data starts immediately on the next multi-leg auto-log.

Two gates surfaced while testing this that are worth knowing about
independent of this PR — both pre-existing, neither touched here:

- `PaperManager.log_trade`'s tradeability gate (`_friction_to_credit_ratio`)
  refuses to log a multi-leg trade at all if round-trip friction exceeds
  the configured ceiling. It tries to *measure* that friction from real
  quotes first, but only from a pre-built `trade_dict["legs"]` list (a
  different shape than the flat `short_bid`/`short_put_bid`/etc. keys this
  PR threads through) — without that key it silently falls back to a flat
  per-share estimate. The two are unrelated inputs to two different gates;
  this PR does not wire the flat quotes into `"legs"`, since doing so would
  change what this gate refuses, which is out of this task's scope
  (recording, not gating).
- Position sizing refuses a trade whose risk-per-contract does not fit
  inside the per-trade budget (2% of book equity, ≈$1,000 today) — a
  reminder for anyone hand-constructing a large-notional multi-leg test
  fixture, not a defect.

## Next step

Not this session. Once enough multi-leg positions have closed with exit
columns populated, a future session can extend
`scripts/reprice_single_leg_book.py` (or a sibling script) to reprice the
multi-leg book the way Task A repriced single-leg — likely not for weeks,
given today's coverage. `src/execution/slippage.py` (Task C item 2 — wiring
the never-used real-fill recorder) is a separate, independent PR, not
folded into this one.
