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

**Update 2026-06-15 (P0–P4 worked through):** the index put-spread edge is now **holdout-validated**
(PF 4.19→4.41 OOS) with a **capped tail** (defined-risk wing holds even when selling at a market
top). BUT the new **portfolio layer is the headline**: sized responsibly the single-index spread
makes **~0.3% CAGR / ~0.7%/yr on deployed capital** (only ~31 trades/3yr, 27 open at once) — the
per-trade PF was hiding a **capacity wall**. The verdict is now surfaced live (display-only) in the
screener; the basket is broadened (META/AMZN/TSLA have data); earnings-vol-selling + skew tooling is
built (earnings result pending a data-fetch pass); real slippage replaced the flat 7% in the
backtester. The one OPEN item is the strategic real-money decision (P2.7) — **yours to make**.

All committed to `main`, **99 tests green**. Nothing is wired into the LIVE trading path (only the
display-only verdict line). Real money stays OFF until the gate fires / you decide.

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
Real universe with data: **AAPL, SPY, MSFT, NVDA, GOOG, AMD, META, AMZN** (+TSLA available;
META/AMZN/TSLA confirmed via live probe 2026-06-15). (QQQ, IWM are NOT in the dataset —
`--audit` shows them EMPTY. Always check `--audit`/probe before adding a symbol.)

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
2. ~~**Stress the bull-market bias.**~~ **DONE 2026-06-15.** New `drawdown_on()` + `in_drawdown(min_dd)`
   filter in `dolt_research` (selling INTO weakness = the worst case for short premium).
   SPY put spread, 2022-2024:
   - **Selling into an established drawdown is the BEST sub-sample**, not the worst:
     baseline PF 4.29 → in_drawdown_10% **PF 5.43** (win 50%, avg +2.3%). Down 10% ⇒ vol is
     rich ⇒ you collect more ⇒ mean-reversion helps. (Confirms the P1.4 VRP thesis.)
   - **The genuine worst case is selling at a TOP as the market begins to fall**: the
     2022-H1 top→fall window is n=4, win 25%, avg **−0.3%**, **PF 0.01** — a wipeout for
     those trades, BUT the defined-risk wing capped the damage to −0.3% avg on max risk (no
     blow-up). That cap is exactly why the spread beats naked puts in a crash.
   - Full 2022 bear *year* is still net positive (PF 8.23) because most of it is selling into
     already-elevated vol. **Takeaway: tail is capped by design; a VRP/drawdown timing filter
     (P1.4) would skip the top-selling losses.** All small-n — qualitative, not proven.

### P1 — turn the signal into a deployable strategy
3. ~~**Position sizing + risk layer.**~~ **DONE 2026-06-15.** New `src/dolt_portfolio.py`:
   `equity_curve(trades, start_equity, risk_frac)` (sequential fixed-fractional compounding) +
   `max_concurrent()`, CLI `python -m src.dolt_portfolio --strategy put_spread --symbols SPY`.
   Required exposing `trades` from `dolt_spread._summarize`.
   **HEADLINE FINDING — the portfolio view destroys the per-trade story:**
   - SPY put spread 22-24, risk 2%/trade: 31 trades → end +1.1%, **CAGR +0.3%**, maxDD −0.2%,
     **maxConcurrent = 27**.
   - PF 4.29 looked great because it's per-trade on max-risk. But you only get **31 trades in
     3 years on one index**, and 27 are open AT ONCE (35-DTE spreads opened weekly stack up).
     To truly cap portfolio risk at 2% you'd size each trade at ~0.07% → returns vanish.
   - **Conclusion: the index put-spread edge is real per-trade but has NEGLIGIBLE portfolio
     CAPACITY as a standalone single-name strategy (~0.3% CAGR).** This directly reframes P2.7:
     you cannot build a real-money program on this alone. Capacity (more names/segments,
     P3.10) — not a better filter — is the binding constraint.
4. ~~**VRP / regime timing.**~~ **DONE 2026-06-15.** Added `realized_vol(ctx, lookback)` +
   `vrp_rich(min_ratio)` (entry only when entry_iv / realized_vol ≥ ratio). SPY spread, 22-24:
   - `vrp_rich_1.1`: **PF 4.29 → 4.87** (n=23) — light filter helps modestly, right sign.
   - `vrp_rich_1.2` / `1.3`: n collapses to 13–17 and PF goes noisy (2.1 / 3.44) — overfiltering
     a thin sample. Holdout of 1.2: train PF 1.55 (n=11) → test 2.9 (n=6), too thin to trust.
   - **Verdict: the VRP thesis has the right sign but the index sample is too small to push
     hard. Use a LIGHT threshold (~1.1×) at most.** The `in_drawdown` filter (P0.2, PF 5.43)
     is actually the stronger version of the same "sell rich vol" idea. Both confirm: the edge
     concentrates in elevated-vol entries. Real lever is more data (P3.10), not a tighter filter.
5. ~~**Assignment / margin mechanics.**~~ **DONE 2026-06-15.** `dolt_portfolio.margin_profile()`:
   for a defined-risk spread, margin/contract = max_risk×100; walks the calendar for PEAK
   simultaneous capital across overlapping trades + return-on-peak-capital + an
   early-assignment proxy (stop-loss exits = short leg breached). SPY spread 22-24:
   - **peak capital ≈ $59,754** (27 concurrent), **return-on-peak-capital +2.2% over 3y**
     (~0.7%/yr on deployed capital). Confirms the P1.3 capacity wall in dollars.
   - **assignment-risk = 0/31 stop-loss exits** — the defined-risk wing means the short leg is
     never force-closed ITM in-sample; early-assignment risk is negligible for the spread
     (this is the whole reason to pay for the wing vs naked puts). Naked-put margin (reg-T,
     ~20% notional, real assignment into shares) is flagged in the function docstring as the
     separate case if the short-put strategy is ever sized.

### P2 — wire it into the system (the user explicitly wants long/short guidance live)
6. ~~**Surface the recommender verdict in the live screener.**~~ **DONE 2026-06-15.** New
   `src/dolt_verdict.py`: cached per-segment LONG/SHORT/STAND-DOWN lookup (recommend() is too
   slow/rate-limited to run live), with built-in validated defaults so it works offline +
   `--build` to refresh into `data/dolt_verdicts.json`. Wired as ONE display-only line in
   `pick_context.context_lines` (the existing failure-safe overlay), shown only for symbols in
   a known DoltHub segment, and it carries the capacity caveat (index = real per-trade edge but
   ~0.3% CAGR sized). Long calls stay a candidate (the recommender picks per-trade; no hard pivot).
7. **Strategic real-money decision (USER call, then code). ⟵ STILL OPEN — the one item that
   needs you.** The gate / cohort feeder / real-money path are built around long calls
   (breakeven). The edge is short index put spreads. **This session's evidence sharpens the
   decision and arguably changes it:** the put-spread edge SURVIVED the holdout (P0.1, PF
   4.19→4.41) and the tail is capped by design (P0.2), BUT the portfolio view (P1.3/P1.5)
   shows it makes **~0.3% CAGR / ~0.7%/yr on deployed capital** as a standalone single-index
   strategy, because you only get ~31 trades/3yr with 27 open at once. So the realistic options:
   - **(A) Don't pivot the real-money path yet.** Neither long calls (breakeven) nor the single-
     name spread (real but ~nil capacity) clears the bar for real money. Keep the gate OFF, keep
     researching. (Most defensible given the data.)
   - **(B) Pivot to short index spreads anyway, sized tiny**, accepting low capacity for a real
     (if small) edge — only worthwhile if combined with the capacity work (P3.10 more names +
     P3.8 earnings selling) to build a portfolio of uncorrelated short-premium sleeves.
   - **(C) Broaden first, decide later** — run the fetched-history backtest on the expanded
     basket (META/AMZN/TSLA) + earnings selling, THEN re-evaluate capacity. (Recommended path.)
   Decision is yours; code follows. Nothing here flips the gate — real money stays OFF until you say.

### P3 — more research (the brainstormed ideas not yet built)
8. ~~**Earnings IV-crush SELLING.**~~ **TOOL DONE 2026-06-15, result data-limited.** New
   `src/dolt_earnings_sell.py`: short-strangle backtest — sell an OTM call+put in the front
   expiry ~2 days before earnings, buy back the first data day after (real BID/ASK, 4-leg
   commission, return on credit, realized move reported). 5 unit tests green.
   **CAVEAT:** running it from the local cache yields only **n=1 usable on AAPL** with
   `realized_move=0.0` (the exit-day spot/chain isn't archived). Earnings dates land on
   arbitrary days; the local cache only holds the densely-archived dates the cohort/spread
   backtests use. **A trustworthy sample needs a dedicated earnings-window chain FETCH pass
   first** (slow, rate-limited — don't run alongside other API jobs). The harness is ready;
   the data isn't, yet. (Same free-data wall noted elsewhere in this doc.)
9. ~~**Term-structure / calendar + skew trades.**~~ **FEASIBILITY CONFIRMED 2026-06-15.** Real
    surface verified on a cached SPY snapshot: classic equity put skew (IV 0.64 at low strikes →
    0.18 at high) and a 3-expiry term structure (medIV 0.190→0.162→0.150, i.e. measurable
    backwardation). Skew + term-structure studies ARE buildable on this data. Limit: only ~3
    expiries cached per snapshot day, so calendar-spread granularity is coarse. Not yet built
    (no backtest) — next research item; build on the broadened basket.
10. ~~**Broaden the basket.**~~ **DONE 2026-06-15.** Probed candidates live: **TSLA (~138 rows),
    META (~162), AMZN (~140) all HAVE data** (control SPY ~130). Added **META + AMZN to the
    `tech` segment** in `dolt_research.SEGMENTS` (3→5 names) to thicken the binding-constraint
    sample; TSLA left ungrouped (its own high-beta character). **Broadened-basket verdicts are
    PROVISIONAL** — the recommender/verdict defaults still reflect the old 3-name tech result
    until a backtest with FETCHED META/AMZN histories re-confirms (same fetch-pass dependency
    as P3.8). The universe is now AAPL, SPY, MSFT, NVDA, GOOG, AMD, META, AMZN (+TSLA available).

### P4 — loose ends / cleanups
11. ~~**Wire `dolt_slippage` into the live EV/preflight.**~~ **DONE 2026-06-15.** `backtester.py`
    now builds the real `measure_spread_table()` once (when the Dolt cache exists) and applies a
    delta/DTE-aware `half_spread_fraction()` haircut at the BS-fallback entry/exit instead of the
    flat 7%; falls back to the config value per-bucket-thin or cache-absent. Real-marks trades
    unaffected (their bid/ask already include the spread). The real table proves 7% is wrong BOTH
    ways: **ATM ≈1.4% half-spread (7% overstates), deep-OTM ≈6.5–16% (7% understates).** 9
    backtester tests still green.
12. ~~**Dead `sentiment` scorer.**~~ **RESOLVED 2026-06-15 (disabled + documented).** Weight is
    already 0.0 in config; the primary yahooquery fetch path hardcodes `sentiment_score = 0.0`
    (the root cause of zero variance), and calibration already masks zero-variance features
    (`mask_zero_variance`). So it cannot affect scoring. Left the display plumbing intact (the
    "Sentiment: Neutral" UI tag) rather than a risky multi-file rip-out; revisit only if a real
    sentiment FEED is wired (free data doesn't provide one — textblob+yfinance news is empty/optional).
13. **DEFERRED (optional, low-value).** Real risk-free rate from DoltHub `rates/us_treasury` vs
    hardcoded 0.045. At the DTEs traded here the rfr barely moves option prices; not worth a new
    network-fetch dependency. Documented as a deliberate skip, not an oversight.

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
