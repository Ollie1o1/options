# Preregistration — Study H4: Intraday Options Execution Clock & Multi-ETF OPRA Liquidity (`cbbo-1m`)

**Registered:** 2026-11-27 (frozen before any Databento OPRA quotes are downloaded).
**Resolves:** The open H4 ("time of day") hypothesis from `docs/OPTIONSDX_RESULTS_20260811.md` §H4, and measures institutional OPRA NBBO spread/depth for `Test O1` (`60–120 DTE`, `5%`-wide ETF Bull Puts) across non-SPY ETFs.

## Data Design (deterministic, zero peeking, budget-capped)

* **Source:** Databento `OPRA.PILLAR`, schema `cbbo-1m` (consolidated NBBO bid/ask + sizes).
* **Universe (6 liquid ETFs):** `SPY`, `QQQ`, `IWM`, `XLE`, `GLD`, `TLT`.
* **Dates (24 sessions):** First full-session (`16:00 ET` close via exchange calendar `data/intraday/calendar.parquet`) trading Wednesday of each month from `2024-01` through `2025-12` (rolled forward deterministically to the next full trading session if Wednesday is a US market holiday like `2025-01-01` or an early close like `2024-07-03`).
* **Intraday Clocks (5):** `09:45`, `10:30`, `12:00`, `14:00`, `15:45` ET (1-minute window each).
* **Hard Cost Ceiling:** `max_usd = $8.00` total against the `$125` Databento signup credit (dry-run verified before download).

## Frozen Hypotheses

| ID | Question | Metric & Unit of Observation | Pass Bar |
|---|---|---|---|
| **H4-A (Intraday Spread Compression)** | Does round-trip bid-ask friction (`(s_ask-s_bid + l_ask-l_bid) / credit_mid`) on `60–120 DTE` `5%`-wide `~0.25Δ` ETF Bull Put spreads (and `20–50 DTE` near-ATM options) tighten meaningfully after the first 15 minutes (`10:30`, `12:00`, `14:00`, `15:45` vs `09:45` ET)? | Paired day-level mean difference (`fric(clock) - fric(09:45)`) across 24 sampled days (day-clustered bootstrap, 10,000 draws, seed `20261127`, Bonferroni across 4 clocks → `98.75%` CI). | `98.75%` CI upper bound `< 0` (strictly tighter than `09:45`) AND holds in both `2024` and `2025` halves. |
| **H4-B (Best Clock vs Close `15:45`)** | Among `10:30`, `12:00`, `14:00`, `15:45`, is any intraday clock strictly tighter or deeper (`min(bid_size, ask_size)`) than the others? | Paired day-level difference vs `15:45` ET (`98.33%` CI across 3 clocks). | `98.33%` CI excludes 0 in both years. |
| **H4-C (Multi-ETF O1 OPRA Friction Audit — descriptive)** | On institutional `15:45 ET` OPRA NBBO quotes, which non-SPY ETFs (`QQQ, IWM, XLE, GLD, TLT`) actually achieve SPY-grade `O1` friction (`<= 10%` median vs `25%` gate) and deep quoted books (`min_depth >= 10`)? | Per-ETF median friction %, p90 friction %, median `min_depth`, and O1 pass rate over 24 monthly sessions. | Descriptive execution map (updates live scanner guidance; does not claim alpha). |

## Operational Consequence
* If **H4-A / H4-B** identifies a statistically superior execution window (e.g., avoid `09:45`, submit limits `10:30–15:45`), codify it as a live execution timing badge/window in `src/radar/live_options.py` and `docs/GO_LIVE_RUNBOOK.md`.
* Use **H4-C** to rank which ETFs belong in the live `etf_wide_spread` (`O1`) watchlist vs which ones hit a structural OPRA spread wall.
