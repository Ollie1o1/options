# Discover — the top-down research view

`opt discover` · TUI tab **[6] DISCOVER** (or `d`) · web cockpit tab **🗺️ Discover** ·
`opt card TICKER` for one name. Code: `src/topdown/`.

Discover is a **research tool**. It authorises no new trade: the only ordering in it
with validated evidence behind it is the Outlook rank of the 16 sector/asset ETFs
(`vol_managed_outlook`, VALIDATED). Everything else says what it is and what tested it.

## Compare expirations in web Discover

Open **Discover → Compare expirations**. Choose the ticker, an archived session, and a call or put.
Filter by days until expiration and select a strike quoted on two or more expiration dates. Choose
up to three dates and enter a stock price to see what happens at each contract's own expiration.
**Market and studies** opens the market, sector and evidence views. The TUI and `opt discover --dte`
display the study groups and the source of any example quote.

The comparison uses two-sided quotes from one ticker, archive session and provider. The details show
the quote time when the provider supplies one, when the fetch finished, and the last trade time.
Older captures keep their original timestamp field. The reported stock price, when consistent, is the
value stored with the archive.
The payoff is a cash-equivalent exercise value at expiration, minus the archived ask and the simulated
entry fee. The midpoint outcome is a sensitivity check. The archive doesn't record fills or verify the
contract deliverable, so the model assumes a standard 100-share contract and screens out contract IDs
that look adjusted. Exercise handling and dividends are outside this calculation.
[OCC equity option specifications](https://www.theocc.com/clearance-and-settlement/clearing/equity-options-product-specifications)

Expand **Quote details and price assumptions** to download the versioned JSON. It contains the exact
contract IDs, dates, bid and ask, source, stored timestamps, stock-price scenario, fee, multiplier
assumption and both payoff calculations. **Open saved replay** restores the original inputs and
recalculates the expiration results. Later-session history comes from the local archive.

### Replay timeline and reader controls

- Each contract card leads with maximum loss, the stock price needed to break even at expiry, and
  either the archived earnings date or the cost of the bid–ask gap. A scheduled date appears only
  when the capture saved a future date already present in the local earnings calendar. Older captures
  show earnings history as unavailable. Later calendar refreshes leave prior snapshots intact.
- **What happened before expiration?** compares contracts on the same recorded date through the first
  expiry. The simulated early exit starts from the entry ask and uses the later recorded bid. The
  breakdown shows the midpoint change, the bid–ask cost, and both fees. Dates with missing quotes or
  changed providers stay visible and break the chart line.
- **Archive coverage by date** displays capture attempts and gaps on dates present in the local market
  calendar. A comparison with one captured session can show the expiration scenario and will gain a
  timeline when later quotes are present.
- **Breakout research** shows the held-out aggregate evidence for the EMA windows and lets readers
  inspect the uncertainty and sample behind each result. [Study protocol and results](EMA_BREAKOUT_STUDY.md)

### Delivery plan

| Phase | Work | Acceptance check |
|---|---|---|
| 1. Evidence and quote labels | Show historical win rates beside average net return and average loss. Label example quotes by source and show zero when a contract exceeds the displayed risk cap. | A missing quote remains a study-only row. A delayed provider and an archived example are distinguishable. Over-cap sizing never shows one contract. |
| 2. Expiry comparison | Add an archive-first web view with same-strike comparisons, a DTE range, an expiry price scenario and plain-language break-even and maximum loss. | A call and a put reconcile to their expiration payoff. Empty ranges, missing sessions and unusable quotes have clear states. |
| 3. Reproducible replay | Expose bid, ask, timestamp context and midpoint sensitivity. Export the versioned inputs and results. | Reloading the same archive rows and scenario inputs reproduces both dollar results. The archive is read-only. |
| 4. Coverage and payoff curve | Show usable quote and expiry counts by provider. Plot the modeled payoff across stock prices. | The payoff curve includes the selected scenario and each attainable break-even. Coverage describes the captured archive. |
| 5. Paper research readouts | Complete the registered selector comparison, then design a DTE study that matches exposure and uses a common evaluation horizon. Include unfilled scenarios, fees, wider spreads, drawdown and a simpler exposure benchmark. Freeze rules before reading a new holdout. | Report the paired difference and its uncertainty. Show evidence only for the strategy, universe, and exit rule tested. Keep inconclusive and negative results discoverable. |
| 6. Same-day history | Reuse saved replays, capture dated earnings context, compare later exit quotes and expose archive gaps. Avoid building the market hub for replay and breakout research. | A saved replay restores its original assumptions. The same-day timeline displays missing quotes. Earnings dates come from the current session’s capture. |
| 7. Breakout research | Test short, medium, and long EMA windows against a fixed baseline for different forecast horizons. | Freeze the protocol before evaluating the holdout. Display forecast error, uncertainty, and sample size together. |

Phases 1–4 and 6–7 are implemented. Phase 5 starts with the existing registered forward study and its
locked readout. Preserve the raw capture and provider metadata with each study artifact, including
incomplete captures and corrections.

## The four market and research steps

| Step | Question | What you see | Evidence |
|---|---|---|---|
| ① MARKET | What kind of market is this? | One-line *Read*, SPY trend, real VIX / VIX3M curve, VVIX, SKEW, FRED credit spread & yield curve, cross-asset rotation ratios | context only — "VIX → direction" was registered and NOT_ESTABLISHED; ratios/FRED untested |
| ② SECTORS | Where is the strength? | 16 ETFs in Outlook order; FAVOR = held by the TFSA sleeve this month, AVOID = regime-gated BEARISH call; RS vs SPY, % of the sector's stocks above their 50d/200d | order & FAVOR/AVOID **PROVEN**; RS & breadth CONTEXT |
| ③ IDEAS | Why look here? | Lenses, each with its label, one sourced evidence sentence, and how it is sorted | per lens (below) |
| ④ EVIDENCE CARD | Should I look closer at this one? | Verdict, price structure, 5 closest peers, options IV & ±5% odds, events/revisions/news, **where every number came from (source, as-of, age)** | per section |

Every panel ends with a dim `Source · …` line; every lens row and card section carries an as-of date.

## Evidence labels

| Label | Meaning |
|---|---|
| PROVEN | passed a preregistered out-of-sample test (playbooks.json VALIDATED) |
| PROMISING | passed part of one — read the caveat (LEAD) |
| TESTING | a registered test exists; its readout date has not arrived |
| WATCHLIST | descriptive overlay, not tested yet (CONTEXT playbooks) |
| CONTEXT | plain description; never tested as a signal here |
| AVOID | a measured negative, or an imminent event |

Green/red on numbers only ever means up/down, never buy/sell.

## Lenses

| Lens (tag) | Label | Rows come from | Sorted by |
|---|---|---|---|
| Core ETF rotation (CORE) | PROVEN | `src/outlook/vm_rotation.py` — the exact backtested rule | Outlook rank |
| Uptrend pullbacks (PULL) | PROMISING | Radar `equity_pullback` | Radar order |
| ETF option spreads (SPRD) | PROMISING | Radar `etf_wide_spread`, `etf_bear_call` | Radar order |
| Sector leaders (LEAD) | CONTEXT | top-5 Outlook sectors, above 200d, 12-1m > 0 | 12-1 month return |
| Estimate revisions (REV) | TESTING (≥ 2027-01-15) | `estimates_archive.db` 30d breadth ≥ 0.35 | breadth |
| Insider clusters (INS) | WATCHLIST | Radar `insider_cluster` | Radar order |
| Option skew (opt_skew) | WATCHLIST | Archived calls, puts and modeled gamma exposure | Skew flags, then call-put IV spread |
| EDGAR events (edgar_events) | WATCHLIST | Cached SEC filings | Filing date |
| Your holdings (HOLD) | WATCHLIST | buy zones & covered calls | Radar order |
| Landmines (MINE) | AVOID | post-earnings breakdowns; earnings ≤ 7 days | crash flags, then days |

`opt discover --ledger` lists the ideas already tested here that do **not** work
(analyst actions, news types, PEAD, breakouts, pre-earnings run-ups, …) so they are not re-chased.

### Forward record — how a label can change

After 18:00 ET on trading days with settled equity data and a current Radar board, maintenance records what each lens listed
(`data/lens_forward.db`, keyed on each row's own data date, so a stale cache counts once).
`opt discover --forward` shows the record; outcomes stay **locked until 2027-05-25** and 26
complete lens-weeks, then are read by the rule frozen in
[PREREG_DISCOVER_LENS_FORWARD_20261125.md](PREREG_DISCOVER_LENS_FORWARD_20261125.md)
(one lens-week = one observation, excess vs SPY net of 10 bps, Bonferroni 99.17% CI). A read
*proposes* a label change; a human makes it. `opt discover --record` records by hand.

For a contract-search audit, run `opt lab scan-audit --ticker SPY`. Add
`--out scan.json` to save per-ticker coverage, every displayed row, the alert
screen's decision, and the quote slices sent to the contract selector.
`opt lab scan-replay scan.json` compares generic contract selections on those
same inputs. Calendar rows carry their own quotes and stay outside the replay.
The default uses the local archive. Add
`--live` to request Yahoo data and report listed, eligible and checked expirations.
Yahoo request time and last trade time give no bid/ask update time. Audit files
are paper-research inputs and contain no measured fills.

The next step after a reproducible selection is a paired paper comparison:
`opt lab study run` captures each archived session, builds the generic bull-put
pick and a matched liquidity-first alternative from the same quotes, records
later archived marks for both, and prints the readout. Rules, read date and
evidence threshold: [PREREG_SELECTOR_LIQUIDITY_BASELINE_20261126.md](PREREG_SELECTOR_LIQUIDITY_BASELINE_20261126.md).

## Card verdicts

`TESTED SETUP` (a PROVEN/PROMISING playbook fired, nothing flags it) · `SETUP + WARNINGS` ·
`AVOID FOR NOW` (no setup; landmine or bearish sector) · `NO TESTED SETUP` (context only — and,
when the stock's sector ETF is held by the sleeve, the card says owning the ETF is the tested way
to hold that exposure). The ±5% odds say whether they are **market-implied** (SVI density from an
archived chain) or a **MODEL** (lognormal on IV or realized vol).

Card sections: 1 Verdict · 2 Price structure · 3 Peers · 4 Options & market-implied odds ·
5 Events, revisions & news · **6 Portfolio fit** (same-sector count and 126-session co-movement
with what you hold — own picks, long-term fills, and the core sleeve's target ETFs — plus the
least-correlated sleeve ETFs; descriptive, thresholds are rules of thumb) · 7 Where every number
came from.

## Data & freshness

Prices: `data/equity_ohlcv.db` (375 symbols incl. SPHB/SPLV/CPER/UUP for the ratios). The header
line says how many symbols lag SPY's last bar. `opt refresh` updates prices **and** FRED
(`data/fred_cache.json`); the TUI `u` key and the web "Update prices" button do the same.
FRED is the latest revised history — fine for "where are we now", not for backtests.
