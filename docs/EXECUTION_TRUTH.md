# Execution truth — what a fill actually costs — 2026-08-04

Every entry in this book was priced at the bid/ask MID. `options_screener.py:2160`
sets `premium = mid`, spreads are built as `short_leg['premium'] - long_leg['premium']`
(`:2505`), and `paper_trading.slippage_per_share` is charged only on the way out.
**Entry friction has always been modelled as exactly zero.**

This document records what that cost, and what was built to stop it.

## 1. The measurement

`data/chain_archive.db` holds real CBOE quotes (bid, ask, size, greeks) for 15
symbols since 2026-06-10. 30 logged Bull Puts fall on a day and expiry it
covers, so they can be re-priced against the quote that actually existed.

| | at mid | crossed |
|---|---:|---:|
| median credit, $2.50-wide spread | $1.05 | $0.60 |
| median credit / width | 0.420 | 0.267 |
| **breakeven win rate** | **58.0%** | **73.3%** |

Crossing costs **$0.35/share — 27% of the credit collected**. 3 of the 30
spreads had no credit left at all once crossed. The book's Bull Put win rate is
70.4%, which clears 58% comfortably and misses 73.3%.

This reproduces `docs/DOLT_CREDIT_SPREADS.md` (Bull Put −1.89% mean on real
crossed marks) from an unrelated data path, which is the strongest evidence
either number has.

## 2. What it does NOT prove

Stated because the obvious next step is wrong.

- **The archive cannot restate the book.** It covers 194 of 907 rows: Bull Put
  30/131, Bear Call 41/135, Iron Condor 38/137, Long Call 26/238. Anything said
  about the other 713 is modelled, not measured, and the two are never pooled.
- **Logged credits are NOT the archived mid.** They run ~11% higher (median
  +$0.10). The archive is CBOE snapshotted between pre-market and mid-session;
  the scanner reads yfinance intraday. That comparison mixes source and timing
  and says nothing about fill policy. The evidence that entries are booked at
  the mid is the code, not this data. What the fixture does support: **30 of 30
  logged credits sit at or above what crossing would have paid.**
- **n is 30–41 per line.** These are precise *price* measurements — a
  mechanical quantity — but the win rates they are compared against are small
  samples.

## 3. Restated, per structure

`python -m scripts.restate_execution --report`, rows the archive prices exactly.
p* is the median of each trade's own breakeven, not the breakeven of the median
credit — with widths spanning $1 to $29 those differ by up to 7 points.

| strategy | n | win% | p*@mid | p*@limit | p*@cross | verdict at cross |
|---|---:|---:|---:|---:|---:|---|
| Bear Call | 41 | 63.4% | 59.0% | 60.8% | 65.0% | −1.6 pts |
| Iron Condor | 38 | 63.2% | 63.4% | 63.8% | 64.9% | −1.7 pts |
| Bull Put | 30 | 66.7% | 58.4% | 65.1% | 73.4% | **−6.7 pts** |

Bull Put has the highest win rate and the worst honest verdict, because its
credit is small relative to the spread it must cross. A day-clustered bootstrap
on *logged* (mid) prices ranked Bull Put the best line in the book at +26.7%
return on risk, CI [+13.2, +42.1]. **That ranking does not survive honest
fills.** Both readings are in the record; the mid one should not be cited alone.

## 4. What was built

- `src/execution_truth.py` — quotes to fills. `leg_fill`, `structure_fill`,
  `breakeven_win_rate`, `edge_report`, `gate`. Pure; knows nothing about the
  ledger or the scanner. Three policies: `mid | limit | cross`.
- `src/execution_restate.py` — ledger rows to priced legs, strategy-aware.
  Refuses rather than half-prices: 13 of 187 logged condors stored only their
  put legs and pricing those as two-leg structures would halve their friction.
- `scripts/restate_execution.py` — backfill and report. Writes only the v18
  columns; `entry_price`, `net_credit` and `pnl_usd` are read-only to it.
  `--dry-run`, `--undo`, `--report`.
- **Schema v18** — `entry_price_mid`, `entry_price_fill`, `entry_price_cross`,
  `fill_policy`, `fill_source`. `entry_price` keeps its v17 meaning forever.
  `fill_source` is `live_quote | modeled | unknown`; never pool them.
- **`_friction_to_credit_ratio` now measures.** The auto-log guard that refuses
  trades whose spread eats their credit estimated friction as
  `2 x $0.05 x n_legs`. Where the payload carries real leg quotes it now uses
  them; where it does not, the flat estimate stands unchanged. On the measured
  Bull Puts the flat number understated the round trip by ~3.5x.

## 5. The one assumption

`DEFAULT_LIMIT_K = 0.35` — a worked limit order is assumed to concede 35% of
the half-spread. It is set from the measured $0.35 mid-to-cross slip, not
derived from filled orders, because no filled orders exist yet. **This is the
only number here that is a guess.** It is isolated in one constant, config-
overridable, and retired the moment real fills are recorded.

## 6. Verified

2812 tests green (`scripts/test.sh`), up from 2743, no regressions. Backfill
round-tripped on the live ledger: `--undo` then re-apply left a SHA-256
fingerprint of `(entry_price, net_credit, pnl_usd, pnl_pct, exit_price, status,
capital_at_risk)` across all 908 rows byte-identical.

The golden test (`tests/test_execution_truth.py::ArchivedQuotesTest`) runs
against `tests/fixtures/bull_put_archived_quotes.json` — 30 real quote pairs
extracted from the archive. If the fill model drifts, those numbers move and
the suite fails.

## 7. What this changes

Nothing about the signal, which was never the binding problem. The screener has
been ranking candidates on a price nobody gets. The levers that attack friction,
measured on the archive:

| lever | worth (points of breakeven win rate) |
|---|---|
| fill near mid instead of crossing | 3–8 |
| 61–120 DTE instead of weekly | ~5 |
| wider spreads | 2–5 |

Longer-dated is the one the book has never tested: **maximum entry DTE across
all 824 closed trades is 58 days.** Note that neither corpus can validate it —
DoltHub tops out at 67 DTE and the archive has only 22 snapshot days — so it
has to be measured forward.
