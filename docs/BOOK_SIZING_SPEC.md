# Position sizing for the paper book — design

Date drafted: 2026-08-19
Status: **BUILT 2026-08-19.** `src/book_sizing.py`, `config.position_sizing`,
wired at `src/paper_manager.py::log_trade`. See §8 for what shipped and the
three places the build departed from this design.
Scope: make position size a decision instead of an accident of option premium.
Prerequisite for: any P&L number this system reports meaning what its label says.

---

## 1. The gap

**Every one of the 1,011 rows in `paper_trades.db` carries `quantity = 1.0`** — a
schema-migration default that nothing has ever written. So position size is set
by the option's *premium*, which is a function of share price and implied
volatility, and has nothing to do with the pick.

Measured 2026-08-19 over 889 closed trades:

| | as sized | equal-weighted |
|---|---|---|
| P&L | **+$8,198** | **−$2,266** |
| profit factor | 1.061 | **0.997** CI [0.844, 1.178] |

**The book's entire headline profit is an artifact of which trades happened to
be large.** Per trade there is no edge. Until size is a decision, every P&L
figure the system reports is mostly a statement about option premiums.

Capital committed ranged **55×** ($60 to $3,285) inside the post-repair cohort
alone. See [`project_book_edge_is_a_sizing_artifact`] in the memory store.

## 2. Why `src/execution/sizing.py` is not the answer

It already exists and is already wired — to the *mirror-mode / real-money*
path (`execution/pipeline.py`, `preflight.py`, `ticket.py`, `leverage/`). It is
**not** reusable here, and this is the trap this spec exists to avoid:

```python
"""Account-aware position sizing for long-call mirror-mode execution."""
risk_per_contract = (entry_price - stop_price) * CONTRACT_MULTIPLIER
cost_per_contract = entry_price * CONTRACT_MULTIPLIER
```

Both are **long-premium formulas**. A Bull Put *receives* a credit and risks
`width − credit`; it never pays `entry_price × 100`, and `entry − stop` is
meaningless for it. **Bull Put is now the only auto-logged strategy**
(`allowed_strategies = ["Bull Put"]`, PR #55), so wiring `size_position` into
the book would connect a function to the one strategy it cannot price.

`src/capital_risk.py::capital_at_risk` already bounds risk correctly per family
— credit spreads as `width − credit`, short puts as collateral, naked calls as
`None` — and already accepts a `quantity` argument that nothing ever sets. That
is the right primitive and it is already here.

**`src/execution/sizing.py` is left untouched.** Repointing it is a separate
job with four other consumers.

## 3. The decisions, and the numbers behind them

Every parameter below was chosen against measured consequences, not defaults.

| decision | chosen | why |
|---|---|---|
| account basis | **book's own equity, compounding** | sizing should shrink as the book loses |
| opening balance | **$50,000** from `2026-08-05` | see below |
| per-trade risk | **2%** of equity | standard fractional-risk; 5% was rejected as aggressive |
| below 1 contract | **refuse the trade** | a position too big to size is one the account cannot afford |
| concurrent cap | **10%** of equity | closes the gap the July sizing replay flagged in its own caveat |
| legacy open book | **grandfathered** | the 122 open positions were never sized |

### Why $50,000 and not $25,000

$25,000 was the first choice. Measured against real trades it **stops the book
completely**, so it was rejected:

```
equity  = 25,000 − 9,890 (realised since restart) = $15,110
budget  = 2% = $302 per contract
recent Bull Puts risk $392–$1,468 per contract  ->  ALL FOUR REFUSE
```

The only trades that *would* size at $302 were the July micro-spreads risking
$25–150 — which are exactly what the `max_friction_to_credit` gate refuses as
uneconomic. **Sizing and the friction gate would admit disjoint sets, leaving
nothing.** At $50,000:

```
equity  = 50,000 − 9,890 = $40,110
budget  = 2% = $802 per contract
concurrent cap = 10% = $4,011  (~5 positions at full risk)

2026-08-19 DIA   risk/contract $  749  ->  1 contract
2026-08-18 CRM   risk/contract $1,468  ->  0  REFUSED (correctly: too big)
2026-08-18 WMT   risk/contract $  392  ->  2 contracts
2026-08-17 NVDA  risk/contract $  772  ->  1 contract
                                    11 of 12 size to >= 1
```

CRM refusing is the feature working, not a miscalibration.

### Why the legacy book is grandfathered

Open risk across the 122 currently-open positions is **$176,323**, against a
10% cap of $4,011 — breached **117×**. Those positions were opened unsized
under the old regime; holding them against a sized-era cap compares two
different books and would refuse every new trade for months. The concurrent cap
therefore counts only positions opened **on or after `sizing_start_date`**.

## 4. Architecture

One new module with one responsibility, and one call site.

### `config.json` → new `position_sizing` block

```
opening_balance      50000
equity_basis_date    "2026-08-05"    # the frozen book-restart split
sizing_start_date    null            # set to the merge date when this ships
max_risk_pct         0.02
max_open_risk_pct    0.10
enabled              true
```

Carries a `_note` recording §3 in prose, matching the house convention that
every config decision states its evidence inline.

### `src/book_sizing.py`

```python
@dataclass(frozen=True)
class SizingDecision:
    contracts: int          # 0 means refuse
    reason: str             # "risk_capped" | "concurrent_capped"
                            # | "below_one_contract" | "unbounded_risk"
                            # | "disabled" | "no_equity"
    risk_per_contract: float
    equity: float

def book_equity(conn, cfg) -> float
    # opening_balance + SUM(pnl_usd) of CLOSED trades since equity_basis_date

def open_risk(conn, cfg) -> float
    # SUM(capital_at_risk) over OPEN trades with date >= sizing_start_date

def size(risk_per_contract: Optional[float], equity: float,
         open_risk: float, cfg: dict) -> SizingDecision
    # PURE. No I/O. The whole decision, fully testable without a database.
```

`size` is deliberately pure and separate from the two queries: it is the part
with the arithmetic worth testing exhaustively, and this project has already
shipped one defect because a decision hid behind a default argument where no
test could see it.

**Rule:**
```
if not enabled            -> contracts = 1        (today's behaviour, explicit)
if risk_per_contract None -> contracts = 0        "unbounded_risk"
n = floor(equity * max_risk_pct / risk_per_contract)
headroom = equity * max_open_risk_pct - open_risk
n = min(n, floor(headroom / risk_per_contract))
if n < 1                  -> contracts = 0        "below_one_contract"
```

### `src/paper_manager.py::log_trade`

The single chokepoint — memory records it as the one place capital decisions
cannot be routed around.

1. `risk_per_contract = capital_at_risk(..., quantity=1)` — **from
   `capital_at_risk`, never from entry premium.**
2. `size(...)` → decision.
3. `contracts == 0` → **do not insert.** Log the reason and `return False`.
   Verified: `log_trade` already returns `False` on its three existing
   refusals (over-budget at `src/paper_manager.py:1308`, untradeable at 1275
   and 1288) and `True` after the insert at 1432. Sizing is a fourth refusal of
   exactly the same shape, so it needs no new return convention.
4. Otherwise set `trade_dict["quantity"] = contracts` and recompute the stored
   `capital_at_risk` at that quantity, so the column keeps meaning what it says.
5. **`allow_unsized` is a key in `trade_dict`, not a function parameter** —
   read as `trade_dict.get("allow_unsized")`, exactly like the existing
   `allow_unaffordable` and `allow_untradeable` flags. It bypasses sizing for a
   deliberate manual entry and logs at quantity 1.

Place the sizing check **after** the existing budget/tradeability refusals, so
a trade that fails a cheaper gate is still refused for that reason rather than
for its size — refusal reasons stay diagnostic.

## 5. Consequences to expect

- **This changes the cohort.** Trades that would have been logged now refuse,
  so pre- and post-sizing book statistics are not comparable. That is what
  `sizing_start_date` is for — freeze it and split on it, exactly as the
  2026-08-05 restart is used.
- **Historical rows are not rewritten.** They keep `quantity = 1.0`.
- **The book will log fewer trades.** Roughly 1 in 12 recent Bull Puts refuses
  on size, plus whatever the concurrent cap blocks beyond ~5 open positions.
- **P&L becomes interpretable for the first time.** Equal-weighted and as-sized
  results converge in meaning once size is deliberate.

## 6. Testing

Against a temp database; no test names the real ledger, the real config, or the
network.

1. `size` refuses when `risk_per_contract` is None — unbounded risk is not
   sizeable.
2. `size` returns `floor(budget / risk)` inside both caps.
3. `size` refuses below one contract rather than rounding up to 1.
4. The concurrent cap binds: with open risk near the ceiling, contracts is
   reduced, and at the ceiling the trade refuses.
5. `enabled: false` yields exactly 1 contract — today's behaviour, explicit.
6. **A Bull Put sizes from `width − credit`, not from its credit**, and the two
   give *different* contract counts on the same row. This is the defect
   `src/execution/sizing.py` would have introduced; assert the difference so a
   future edit cannot quietly reintroduce premium-based sizing.
7. `book_equity` = opening balance + realised P&L since the basis date, and
   **excludes** trades closed before it.
8. `open_risk` **excludes** positions opened before `sizing_start_date`
   (grandfathering), asserted with one legacy and one sized position.
9. `log_trade` inserts no row when sizing refuses, and the ledger row count is
   unchanged.
10. `log_trade` stores `capital_at_risk` computed at the sized quantity, not at
    quantity 1.
11. `allow_unsized=True` inserts at quantity 1 despite a refusal.
12. A full suite leaves the real `paper_trades.db` untouched.

## 7. Out of scope

- `src/execution/sizing.py` and its four consumers (§2).
- Rewriting historical `quantity` values (§5).
- Kelly sizing. Half-Kelly needs a win probability, and this system's ranker is
  disproven (OOS IC −0.12, rho −0.030 p 0.38 on 880 closed). A fractional-risk
  rule that needs no forecast is the honest choice while no forecast is
  trustworthy.
- Any change to a gate, the allowlist, or the exit rules.
- `reports/TRACK_RECORD.md`, which still publishes the as-sized headline. It
  should gain the equal-weighted figure, but that is a reporting change, not
  this one.

---

## 8. What was built, and where it departed from this design

Built 2026-08-19 on `feat/book-sizing`. Every decision in §3 shipped as
written; the arithmetic in §3 reproduces exactly against the real ledger
(equity $40,109.62, budget $802.19, DIA 1 contract, WMT 2, NVDA 1, CRM refused).

Three departures, each forced by something this design did not know:

1. **`book_equity` splits on ENTRY date, not exit date.** §6 test 7 says
   "excludes trades closed before it", which entry-dating also satisfies — a
   trade closed before the basis was necessarily entered before it. Entry-dating
   is what the 2026-08-05 restart means everywhere else in this system, and it
   is the population §3's numbers were measured on: exit-dating gives -$4,281
   over 64 trades instead of the -$9,890 over 25 that the $50,000 balance was
   chosen against.

2. **`sizing_start_date` is 2026-08-20, not the 2026-08-19 build date.** Six
   positions were auto-logged unsized earlier on the 19th carrying $4,567 of
   risk — already past the $4,011 concurrent ceiling. Dating the sized era to
   the 19th would have held those trades against a rule they were never subject
   to and refused every new entry until they closed: the §3 grandfathering
   argument in miniature. The boundary is the first day sizing governs an entry.

3. **Realised P&L had to be made quantity-aware first**, which §7 did not
   anticipate. `_sanitize_close_values` computed
   `pnl_usd = entry_price x pnl_pct x multiplier` and no equity exit path scaled
   it, so every realised dollar described ONE contract. Invisible while every
   row carried `quantity = 1.0`; the moment sizing writes 2, a two-lot winner
   books at half its value — into `book_equity`, which is what sizes the next
   position. Fixed at all four close paths plus the eight per-contract dollar
   figures in the portfolio view. `pnl_pct` is deliberately untouched: a return
   is a return at any size, and it feeds the IC sample.

Also found and fixed on the way: **`quantity` was never written to the table at
all.** The INSERT had no such column, so every row inherited the migration's
`DEFAULT 1.0` — including the crypto screener's carefully computed fractional
quantities, which a backfill script had been patching in afterwards.

Two consequences to expect that §5 did not name:

- **A Bull Put with no recorded `spread_width`/`max_loss_usd` is now REFUSED**
  rather than sized off its credit. The live auto-log path always records one;
  two test fixtures did not, and now carry `allow_unsized` for the same reason
  they already carried `allow_unaffordable`.
- **The concurrent cap cannot bind until 2026-08-20.** Between the merge and
  midnight the per-trade cap applies alone.
