# Marking structures — design

Date drafted: 2026-08-19
Status: **IMPLEMENTED 2026-08-19.** See the branch `fix/mark-structures`.
Scope: a defect fix in `src/candidate_marks.py` (shipped in PR #51) plus the
health check that should have caught it. Prerequisite for sub-project 4, the
removal test.

---

## 1. The defect

`candidate_marks.mark_open` finds a contract's current quote by looking up
`(round(strike, 4), opt_type)` in the fetched chain. For a spread or a condor
**both of those columns are NULL**: `candidate_record` promotes only a single
`bid`/`ask`/`strike`/`opt_type` set to fixed columns, and per-leg data survives
in `features_json`.

So every structure position opens `OPEN`, is never marked, and can never
resolve. Measured on the live database:

| refusal reason | marked | unmarked |
|---|---|---|
| `friction` | 0 | **1,892** (100%) |
| `condor_universe` | 0 | **8** (100%) |
| `negative_ev` | 426 | **704** (62%) |
| kept | 356 | **206** (37%) |

**2,802 of 3,592 open positions — 78% — will never resolve.**

Entry pricing is not affected: `legs_for` / `entry_price_for` already read legs
out of `features_json`. Only the marking side is blind.

## 2. Why it stayed invisible, and the check that fixes that

`health_lines` goes CRITICAL only when marks are **totally** zero. 782 marks
existed, so it read `OK` while 78% of the book was dead.

> A health check that tests for total silence does not catch partial silence.

That is the same lesson as the LaunchAgent outage one level down, and it is the
more valuable half of this fix: the marking bug is one bug, but a health line
that cannot distinguish "nothing to do" from "most of it silently failing" will
hide the next one too.

## 3. One leg spec, not two

`candidate_marks._LEG_QUOTES` and `candidate_record._LEG_STRIKES` already
describe the same legs in two places:

```
_LEG_QUOTES ["Bull Put"] = (("short", "sell"), ("long", "buy"))
_LEG_STRIKES["Bull Put"] = ("short_strike", "long_strike")
```

In every entry `strike_field == prefix + "_strike"`, so the second map is
derivable from the first. Marking needs a third fact — each leg's **option
type** — and adding it as a third parallel map would make three structures that
must agree and will eventually not. Two copies of a contract's identity
drifting apart is the defect shape this project keeps finding.

So `_LEG_QUOTES` is replaced by a single `_LEG_SPEC`:

```python
_LEG_SPEC = {
    "Bull Put":    (("short", "put",  "sell"), ("long", "put",  "buy")),
    "Bear Call":   (("short", "call", "sell"), ("long", "call", "buy")),
    "Iron Condor": (("short_put",  "put",  "sell"), ("long_put",  "put",  "buy"),
                    ("short_call", "call", "sell"), ("long_call", "call", "buy")),
}
```

Quote fields are `f"{prefix}_bid"` / `f"{prefix}_ask"`, the strike field is
`f"{prefix}_strike"`. A test asserts `_LEG_SPEC` and
`candidate_record._LEG_STRIKES` still describe the same legs in the same order,
so the remaining cross-module pair cannot drift silently.

`legs_for` — the **entry** pricing path — consumes `_LEG_QUOTES` today and is
repointed at `_LEG_SPEC`, reading the same `f"{prefix}_bid"` / `f"{prefix}_ask"`
fields it already reads. Its behaviour must not change, and a test pins entry
prices for a Bull Put and an Iron Condor across the refactor. A silent shift in
entry pricing would corrupt every position already open.

`candidate_record._LEG_STRIKES` is **not** changed — `contract_key` depends on
it, and rewriting the identity function of a table with 8,700 rows in a defect
fix is not a trade worth making.

## 4. Marking a structure

New `marking_legs(row) -> Optional[List[dict]]`, returning
`[{"strike": float, "opt_type": str, "side": "buy"|"sell"}, ...]` in
`_LEG_SPEC` order.

- **Structure** (`strategy_name` in `_LEG_SPEC`): strikes read from
  `features_json`, option type and side from the spec. Any leg whose strike is
  missing or unparseable makes the whole call return `None`.
- **Otherwise** — a single leg, or a strategy the spec does not know: one leg
  from the fixed `strike` / `opt_type` columns, with `side = "sell"` when the
  strategy name starts with `Short` and `"buy"` otherwise, matching `legs_for`.
  Returns `None` when either column is NULL, which is exactly today's
  behaviour for those rows.

An unrecognised structure therefore degrades to "unmarkable", never to a
half-priced guess.

`mark_open` then:

1. Groups open positions by `(symbol, expiration)` — **unchanged**, and this is
   why the fix needs **no extra network calls**: every leg of a structure
   shares one expiration, so the existing chain request already covers them.
2. For each position, resolves every leg against that chain.
3. If **any** leg is unquoted, writes no mark for the whole structure — the
   same refusal `legs_for` already applies at entry. A structure priced from
   one real quote and one guess is not a price.
4. Computes the net mark with `execution_truth.structure_fill(legs, "mid")` and
   stores `mid = abs(price)`.

`structure_fill` returns a price signed from the trader's cash perspective, and
`pnl_pct` already derives direction from the sign of the **entry**. The mark is
therefore stored as a positive premium value, matching what `pnl_pct` expects
for its second argument and what single-leg marks already store.

Storage for a structure mark: `mid` set, **`bid` and `ask` NULL**, and
`source = 'live_quote_structure'`. NULL because a structure has no single
two-sided quote — inventing one would be a number describing something other
than its label. The distinct `source` means a structure mark can never be
mistaken for a single-leg one in later analysis.

## 5. Health: catch partial silence

`health_lines` gains one number: **open positions carrying zero marks**, and
goes CRITICAL when it is non-zero.

```
cand marks   782 marks / 3592 open / 0 closed in 7d      [CRITICAL]
   2802 OPEN POSITIONS HAVE NEVER BEEN MARKED — they cannot resolve
```

The existing total-zero alarm stays; it catches a different failure (the marker
not running at all). Both are needed: this defect had a healthy total and a
dead majority.

## 6. Deliberately NOT changed

**The exit-pricing asymmetry.** Entries are priced at `limit`
(mid + 0.35 x half-spread); exits mark at `mid` (0 x half-spread). A simulated
round trip is therefore charged roughly a fifth of a crossed one, which
flatters wide-spread candidates — precisely those the `friction` gate refuses.

This is real and it biases the removal test toward concluding the friction gate
is wrong. It is not fixed here because `candidate_positions` is already
accruing a series under the **frozen** registration in
`docs/PREREG_RANKER_TEST.md`, and changing mark pricing mid-flight would change
the meaning of that series after its terms were fixed. Re-pricing marks is a
decision about a live experiment, not a bug fix to be folded into one.

It is recorded here so the removal-test spec designs around it explicitly
rather than inheriting it silently.

**No backfill.** The 2,802 unmarked positions begin accruing marks from the
next run. Their price path has a gap before that date, which matters only for a
position that hit and reversed a take-profit inside the gap. Backfilling would
mean reconstructing historical chains this project does not have for 64 of its
78 symbols — see the chain_archive coverage finding.

## 7. Testing

Against a temp database with the quote fetcher **injected**; no test touches
the network, the real ledger, or the real candidate database.

1. A Bull Put position is marked, and its `mid` equals
   `abs(structure_fill(quotes, "mid").price)` computed independently in the
   test — asserted against the library, not a hand-typed constant.
2. An Iron Condor is marked from all four legs.
3. A structure with **one** leg missing from the chain gets **no** mark.
4. A structure mark stores `bid`/`ask` NULL and
   `source='live_quote_structure'`.
5. **One chain call still serves a whole structure**, asserted by counting
   fetcher invocations — the fix must not multiply network calls.
6. Single-leg marking is byte-for-byte unchanged: same mid, same `bid`/`ask`,
   same `source='live_quote'`.
7. A marked structure position **resolves** — take-profit fires on a credit
   spread whose mark falls, proving the sign convention end to end.
8. `_LEG_SPEC` and `candidate_record._LEG_STRIKES` describe the same legs in
   the same order, for every strategy in both.
8b. **Entry pricing is unchanged by the refactor**: `entry_price_for` returns
   the same value for a Bull Put and an Iron Condor as it did against
   `_LEG_QUOTES`, asserted against `execution_truth.structure_fill` computed
   independently. A silent shift here would corrupt every open position.
8c. A row whose `strategy_name` is not in `_LEG_SPEC` and whose fixed
   `strike`/`opt_type` are NULL yields no marking legs — unmarkable, not a
   guess.
9. `health_lines` goes CRITICAL with an unmarked open position and names the
   count.
10. `health_lines` stays OK when every open position is marked.
11. A full suite leaves the real `data/candidates.db` untouched.

## 8. Out of scope

- The removal test itself (sub-project 4). This unblocks it.
- Exit re-pricing (§6).
- `short_premium`, still `UNSUPPORTED` — its stops need spot and delta.
- Any change to `contract_key`, the recorder schema, or a gate.
