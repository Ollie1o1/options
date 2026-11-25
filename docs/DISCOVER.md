# Discover — the top-down research view

`opt discover` · TUI tab **[6] DISCOVER** (or `d`) · web cockpit tab **🗺️ Discover** ·
`opt card TICKER` for one name. Code: `src/topdown/`.

Discover is a **research tool**. It authorises no new trade: the only ordering in it
with validated evidence behind it is the Outlook rank of the 16 sector/asset ETFs
(`vol_managed_outlook`, VALIDATED). Everything else says what it is and what tested it.

## The four numbered steps (identical on every surface)

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
| Your holdings (HOLD) | WATCHLIST | buy zones & covered calls | Radar order |
| Landmines (MINE) | AVOID | post-earnings breakdowns; earnings ≤ 7 days | crash flags, then days |

`opt discover --ledger` lists the ideas already tested here that do **not** work
(analyst actions, news types, PEAD, breakouts, pre-earnings run-ups, …) so they are not re-chased.

## Card verdicts

`TESTED SETUP` (a PROVEN/PROMISING playbook fired, nothing flags it) · `SETUP + WARNINGS` ·
`AVOID FOR NOW` (no setup; landmine or bearish sector) · `NO TESTED SETUP` (context only — and,
when the stock's sector ETF is held by the sleeve, the card says owning the ETF is the tested way
to hold that exposure). The ±5% odds say whether they are **market-implied** (SVI density from an
archived chain) or a **MODEL** (lognormal on IV or realized vol).

## Data & freshness

Prices: `data/equity_ohlcv.db` (375 symbols incl. SPHB/SPLV/CPER/UUP for the ratios). The header
line says how many symbols lag SPY's last bar. `opt refresh` updates prices **and** FRED
(`data/fred_cache.json`); the TUI `u` key and the web "Update prices" button do the same.
FRED is the latest revised history — fine for "where are we now", not for backtests.
