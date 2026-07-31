# Mark Trustworthiness Spec — for operator sign-off

Date drafted: 2026-07-31
Status: **DRAFT — no code changes until the sign-off block is completed.** Changes live exit behaviour, so per the exit-rules constraint this is a deliberate, journaled decision, not a cleanup.
Covers idea: `mark-fallback-hardcoded-vol`.

## The problem

`PaperManager.update_positions` marks each open leg through a three-step fallback (`_fetch_option_price`, src/paper_manager.py:1181–1212):

1. `fast_info.last_price` — the last *trade*, which can be hours or days stale on illiquid contracts;
2. daily-close history (`tkr.history(period="1d")`) — same staleness, one day granularity;
3. `american_price(option_type, S, K, T, r, **0.30**)` — a model price at a hardcoded 30 vol **regardless of the name**.

The resulting price goes into `option_price_cache` as a bare float, and the exit checks (take-profit / stop-loss / delta rules, call sites ~1427–1446) consume it with no knowledge of where it came from. Two distinct failures:

- **A fabricated mark can write a permanent exit.** Marking an 80-IV name at 30 vol misprices it severely; if that model price crosses a stop or TP threshold, a real exit row is recorded in the ledger — which the gate cohort, the track record, and every P&L statistic then read forever. The row's *actual entry IV is stored in the schema* (`entry_iv`, v16) and is simply not used.
- **Stale-trade preference.** Preferring last trade over the live bid/ask mid is inconsistent with the repo's own quote-freshness rules; the single-leg close path already reads the live spread (`_get_spread_slippage`, line 1003).

## The change (three parts, one decision — they jointly define "what is a trustworthy mark")

### 1. Model fallback uses the row's stored entry IV

`american_price(..., sigma)` takes `sigma = row.entry_iv` when present and > 0, falling back to 0.30 only when `entry_iv` is NULL (pre-v16 rows never backfilled with IV). The mark should also log at debug level which sigma it used.

Entry IV is itself an approximation of current IV — vol moves after entry — but it is name-specific and strictly dominates a global constant. No claim of precision is made; see part 2 for why precision isn't required.

### 2. A model-fallback mark can never fire an exit

`option_price_cache` values become `(price, source)` where `source ∈ {"last", "close", "model"}`. Behaviour by consumer:

- **Unrealized P&L display:** all sources allowed (a model mark beats a blank).
- **Exit checks (TP, stop, delta-based, strike-breach priced off the mark):** if any leg of the row's mark chain has `source == "model"`, **skip the exit evaluation for that row this run** and log one warning line naming the row and leg (throttled to once per row per run). The row stays OPEN and is re-evaluated next run when a market price may exist.
- **Deterministic expiry settlement (dte <= 0 intrinsic path) is unaffected** — it prices off spot, not the option mark, and remains the terminal guarantee that no row hangs OPEN forever. This is what bounds the worst case of part 2: a chronically quoteless contract exits at expiry settlement rather than never.
- Time-based exits that do not depend on the mark's level (pure DTE rules) remain allowed; they record the model mark as the exit price ONLY if no market mark exists, and stamp the exit row's `exit_reason` with a `(model mark)` suffix so the ledger carries the provenance.

### 3. Prefer bid/ask mid over last trade

Mark preference order becomes: **live mid → last trade → daily close → model**. Mid is taken only when bid and ask are both present, both > 0, and not crossed (reuse the crossed-quote guard convention from the 2026-07-13 audit fixes); a one-sided or crossed book falls through to last trade. Implementation note: per-leg bid/ask comes from the option-chain fetch for the (ticker, expiration) pair — batch one chain call per pair rather than per leg, mirroring how `_get_spread_slippage` reads the book; no extra per-leg requests.

## What this changes in the record

- Post-change exits are on a different marking basis than pre-change exits. **A dated entry goes into docs/CALIBRATION_JOURNAL.md at merge**, and the checkpoint that first includes post-change exits notes it, so any IC step-change has its candidate explanation on the record.
- Expected effect direction: fewer spurious stop/TP exits on illiquid names, slightly later exits on quoteless contracts (bounded by expiry settlement). No retroactive edits to any existing exit.

## Test plan (asserted before merge)

1. **No exit from a model mark:** synthetic open row whose only obtainable mark is model-sourced and crosses the stop threshold → after `update_positions`, row is still OPEN and the skip warning was logged.
2. Same row with a market mark crossing the stop → exits exactly as today (regression).
3. Fallback sigma: row with `entry_iv = 0.80` → model path called with 0.80; row with NULL `entry_iv` → 0.30.
4. Mid preference: bid/ask present and sane → mid used; crossed book → last trade; nothing → close → model, in order.
5. Expiry settlement still fires on dte <= 0 regardless of mark source.
6. Full suite via `scripts/test.sh` green.

## Sign-off

- [ ] Part 1 (entry-IV fallback sigma): approved / amended: __________
- [ ] Part 2 (model marks never fire exits; provenance-tagged cache): approved / amended: __________
- [ ] Part 3 (mid preferred over last trade): approved / amended: __________
- [ ] Calibration-journal entry at merge: acknowledged
- [ ] Date + initials: __________
