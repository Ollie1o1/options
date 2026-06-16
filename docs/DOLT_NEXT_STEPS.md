# DoltHub Real-Marks Research — Handoff / What's Left

**Last updated:** 2026-06-15. Read `docs/DOLT_OPTIONS.md` (usage) and
`docs/DOLT_RESEARCH.md` (findings) first — this doc is the forward-looking to-do.

---

## Where things stand (TL;DR)

We wired 7 years of real EOD option marks (free DoltHub `post-no-preference/options`,
no auth) into a backtest + research stack, and used it to find what actually works
on real prices. The honest bottom line:

- **Buying premium (long calls) is ~breakeven** after real costs. Cost is the wall.
- **Selling premium beats buying it** almost everywhere (positive median vs negative).
- **The edge is segment-specific** (this is the key result):
  - **Index (SPY): sell put credit spreads — PF 4.29** (defined risk, the best & safest).
  - **Semis (NVDA/AMD): sell naked puts — PF 1.17** (the spread version loses there).
  - **Tech (AAPL/MSFT/GOOG): STAND DOWN** — nothing has an edge.
- A **recommender** (`dolt_research --recommend`) reproduces these verdicts from the
  backtest on demand. Long calls remain a candidate; the system picks them when they win.

All committed to `main`, 86 tests green. Nothing is wired into the LIVE trading path yet.

## The toolkit that exists (all CLIs, all read the local cache)

```bash
PY="PYTHONPATH=$PWD ~/.venvs/options/bin/python"
$PY -m src.dolt_options --audit          # data trust: per-symbol coverage, EMPTY flags
$PY -m src.dolt_options --probe          # sanity: fetch one chain
$PY -m src.backtester ... --price-source dolt   # real-marks walk-forward
$PY -m src.dolt_cohort  --symbols ...     # long-call cohort (gate strategy)
$PY -m src.dolt_short   --symbols ...     # short puts/calls
$PY -m src.dolt_spread  --symbols ...     # put credit spreads (defined risk)
$PY -m src.dolt_validate --symbols ...    # price-slice scorer IC vs real returns
$PY -m src.dolt_research --sweep | --segments | --recommend --symbols ...
$PY -m src.dolt_earnings --iv-crush AAPL  # earnings IV-crush study
```
Real universe with data: **AAPL, SPY, MSFT, NVDA, GOOG, AMD**. (QQQ, IWM are NOT in the
dataset — `--audit` shows them EMPTY. Always check `--audit` before adding a symbol.)

---

## What still needs to be done (prioritized)

### P0 — make the edge trustworthy before any real money
1. ~~**Walk-forward / holdout the WINNING strategy (index put spreads).**~~ **DONE 2026-06-15.**
   `train_test` now takes a `strategy` arg (`STRATEGIES` registry: long_call / short_put /
   put_spread) and a `--strategy` CLI flag, so any strategy can be holdout-validated, not just
   long calls. Result for the index put spread (SPY, baseline):
   - `--train-test baseline --strategy put_spread --symbols SPY`
   - **TRAIN 22-23** (incl. 2022 bear): n=19, win=47%, avg +1.6%, med −0.2%, **PF 4.19**
   - **TEST 2024** (held out):          n=12, win=33%, avg +2.2%, med −0.1%, **PF 4.41**
   - **Verdict: PF SURVIVES the holdout** (4.19→4.41) and survives 2022 in-train. BUT it is
     NOT proven: **n is tiny (12 OOS trades)**, win rate is LOW (33%) and **median is negative**
     both periods — the positive PF rides a handful of large wins vs many small losers, a
     fragile distribution on n=12. Promising, still not deployable on its own. Next: more
     entries (denser cadence / more index data) and the P0.2 drawdown stress before sizing.
2. **Stress the bull-market bias.** 2022–2024 was net-bullish after the dip — short premium's
   friend. The one scenario to respect is a sustained crash. Find the worst drawdown windows
   in the data and report the strategy's behavior there explicitly.

### P1 — turn the signal into a deployable strategy
3. **Position sizing + risk layer.** Every backtest is 1 contract; returns are per-contract on
   premium/max-risk. Add fixed-fractional (e.g. risk 1–2% of equity per trade) sizing and
   report portfolio-level equity curve / max drawdown, not just per-trade PF.
4. **VRP / regime timing.** Sell index put spreads only when vol is RICH (IV ≫ realized, or
   elevated VIX). The `entry_filter` hook + `dolt_research` filters already support this —
   add a `realized_vol` helper and a `vrp_rich` filter, then re-run the index spread. Likely
   pushes PF 4.29 higher and avoids selling cheap vol.
5. **Assignment / margin mechanics.** Short puts can be assigned; spreads have margin. Model
   early assignment and the actual capital tied up so the equity curve is real.

### P2 — wire it into the system (the user explicitly wants long/short guidance live)
6. **Surface the recommender verdict in the live screener.** When scanning a name, show
   "LONG / SHORT / STAND DOWN" from `dolt_research.recommend()` (or a cached per-segment
   lookup) so the day-to-day tool reflects what the backtest learned. Keep long calls as a
   candidate — do NOT hard-pivot to short-only.
7. **Strategic real-money decision (USER call, then code).** The gate / cohort feeder /
   real-money path are built around long calls (breakeven). The edge is short index put
   spreads. Decide whether the real-money path should evolve toward the validated strategy.
   This is a decision first, code second.

### P3 — more research (the brainstormed ideas not yet built)
8. **Earnings IV-crush SELLING.** We measured ~24% post-earnings IV crush and used it as a
   filter; the bigger trade is to SELL that vol pre-earnings (strangle/IC) and buy it back
   after. Data is ready (`dolt_earnings` + before/after chains).
9. **Term-structure / calendar + skew trades.** Full surface is available on real marks.
10. **Broaden the basket** where `--audit` confirms data, ideally adding a few more names per
    segment to firm up the (currently thin) per-segment samples.

### P4 — loose ends / cleanups
11. **Wire `dolt_slippage` into the live EV/preflight.** The real per-contract slippage model
    exists but isn't used by the live system, whose EV still assumes a flat 7%/side.
12. **Dead `sentiment` scorer** (zero variance) — fix or remove (separate from DoltHub work).
13. **Optional: real risk-free rate** from DoltHub `rates/us_treasury` instead of hardcoded
    0.045 in the backtesters.

---

## Gotchas a fresh session MUST know (so mistakes aren't repeated)

- **QQQ & IWM are NOT in the options dataset** — they cache as all-empty and silently produce
  zero trades. ALWAYS run `dolt_options --audit` before trusting a symbol. The guard
  `dolt_options.symbol_has_data()` + a `[warning]` in the runners now catches this.
- **Split names need raw prices.** yfinance is split-ADJUSTED; DoltHub strikes are raw.
  `dolt_stocks.close_history()` un-adjusts via the DoltHub `split` table. Without it, NVDA/
  GOOG/AMD get skipped by the moneyness guard. The cohort/short/spread runners already use it.
- **DoltHub has day-gaps + a ~30s query deadline.** Scope queries by date; use `get_chain_near`
  (snaps to nearest day with data). Rate limits (HTTP 403) are handled with backoff →
  `DoltRateLimited`; harnesses return PARTIAL results instead of crashing. Don't run two
  API-heavy jobs at once (they compete and 403).
- **Returns are per-trade on premium / max-risk**, NOT portfolio returns. PF is comparable
  across strategies; avg/median are not directly comparable between long (on premium paid)
  and short/spread (on premium collected / max risk). Sizing (P1.3) fixes this.
- **Backtest from a DB snapshot if something else is writing the cache** (`sqlite3 .backup`)
  to avoid "database is locked" / disk I/O errors.
- **Data is EOD only, ends 2026-06-12, price/Greek slice only** (no news/sentiment historically).
  It's a research/validation set, not a live feed.

## How to verify the stack still works (run after pulling)

```bash
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
  tests.test_dolt_options tests.test_dolt_validate tests.test_dolt_slippage \
  tests.test_dolt_cohort tests.test_dolt_earnings tests.test_dolt_stocks \
  tests.test_dolt_short tests.test_dolt_spread tests.test_dolt_research
# expect: OK (~86 tests). Then:
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.dolt_options --audit   # confirm coverage
```

Tests are **unittest** (venv has no pytest). Prefix live runs with
`OPTIONS_MAINTENANCE_CHILD=1`. Commits must have ZERO AI attribution.
