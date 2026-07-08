# Data & Leverage Roadmap

Captured 2026-06-25. Decisions + reference for improving the screener's data and
for the leverage execution-vehicle work. Free items are built; paid items are
"back pocket".

## 1. Options data quality

**Diagnosis:** backtest data is fine (free DoltHub 7yr EOD chains); the *live*
side is the weak link. Yahoo option quotes are delayed/stale/wide and its IV
field is unusable (hence the "IV corrected … yahoo X% → solved Y%" re-solving).
Better data sharpens fills, IV accuracy, and intraday signals — it does **not**
manufacture edge (scorer IC still ~0.03). Worth doing for execution + the
short-vol side, not to rescue the long-call picks.

### Built (free)

- **FRED → yfinance Treasury fallback** (`src/macro_rates.py`). FRED's
  `fredgraph.csv` endpoint read-times-out from this network even at 15s, so the
  macro panel showed "N/A". When FRED yields nothing, we now fall back to
  yfinance Treasury indices: 10Y (`^TNX`), 3M (`^IRX`), and the **10y-3m slope**
  (the Fed's preferred inversion signal; Yahoo has no clean 2Y). Panel marks
  "(via Yahoo — FRED unreachable)". Adaptive 10x scaling handles both `^TNX`
  quote conventions (43.9 vs 4.39).
- **Finnhub earnings + dividend provider** (`src/earnings_provider.py`).
  Yahoo earnings dates are unreliable and the IV-crush penalty depends on them.
  Finnhub's free tier gives a clean earnings calendar + indicated dividend yield.
  Activates when `FINNHUB_API_KEY` (env) or `config.data_providers.finnhub_api_key`
  is set; otherwise returns None and the existing yfinance path is used.
  `get_next_earnings_date` now prefers it. **Action: drop a free key in to
  activate** (finnhub.io/register).

### Back pocket (paid) — ranked by leverage-per-dollar

| Source | Cost | What it unlocks |
|--------|------|-----------------|
| **Theta Data** (thetadata.net) | ~$30–80/mo | Best cheap option. OPRA real-time **and** historical chains, greeks, NBBO, intraday OI/volume. Kills the IV-solve noise; feeds real intraday UOA. |
| **Tradier** brokerage API | Free w/ funded acct | Real-time NBBO chains + streaming — cheapest path to live quote quality if opening an account anyway. |
| **Polygon Options Starter** | $29/mo | 15-min delayed full historical aggregates/trades/quotes — backtest + near-live tier. |

## 2. Leverage as an execution vehicle

**Reframe (important):** a leveraged perp is NOT a cheaper option. The option
premium is paid upfront and is the max loss (convex, floored). A leveraged
delta-1 position has no floor — low margin = liquidation right behind you — plus
funding. So "options are expensive → use leverage" trades a capped, convex cost
for liquidation risk + funding + no convexity.

**The valid half:** options get expensive when IV is **rich** (today: +26% IV
premium, options RICH). When overpaying for vol, a delta-1 leveraged position
skips the vol tax. So the rule is: **use IV richness as the switch** — rich IV +
directional conviction + a defined stop → leverage; cheap IV or need
defined/unattended risk → option.

### Built (free): risk-matched selector

`src/leverage_selector.py` (display-only, wired into `pick_context.context_lines`
after the Quant read, single-leg directional picks only). For each pick it sizes
leverage so margin-at-risk = the option premium (same dollars at risk), then
reports implied leverage, liquidation price, funding drag, and which vehicle the
IV regime favors.

**Funding is live, not hardcoded.** The carry cost of a leveraged delta-1
position is fundamentally the short rate, so daily funding is derived each scan
from the live risk-free rate (`get_risk_free_rate`, ^IRX, 15-min cache) plus a
configurable venue `carry_spread_annual` (default 5%). The line shows the live
annualised rate (e.g. "live 8.7%/yr"). `funding_rate_daily` is `null` by default
(derive live); set a positive number only to pin a manual override.
`maintenance_margin` and `max_leverage` are venue constants and stay in config.
Config block: `config.leverage_selector`.

Example line:
```
Leverage read: same $642 at risk = 18.9x long, liq @ $255.80 (-5.3%)
  |  funding ~$77 over hold  |  IV RICH -> leverage favored: skip the vol tax;
  risk is the liq wick, not premium
```

### Venues (USDC/USDT from a Phantom wallet)

- **xStocks** (Backed) — tokenized AAPLx/TSLAx/SPYx on **Solana**, native to
  Phantom; spot tokens, leverage via Kamino lending.
- **Gains Network / gTrade** — synthetic leveraged stock perps, USDC, low fee,
  on Base/Arbitrum/Polygon (Phantom's EVM side reaches Base). Most direct.
- **Ostium** (Arbitrum) — RWA perps incl. equity indices/commodities/FX.

Verify current chain support before use; this space moves monthly.

### Back pocket (free, later)

- Funding's financing base is already live (risk-free rate). The remaining
  assumption is the venue `carry_spread_annual` (the platform's markup over the
  base). A true live per-pair funding feed from the **gTrade subgraph** /
  xStocks APIs would replace that spread with the venue's actual quoted funding.

## Crypto leverage candidate signals (2026-07-07)

`python -m src.leverage validate` runs four daily-horizon candidates
(trend_breakout, funding_contrarian, trend_carry, xsect_momentum) through a
net-of-cost walk-forward harness over the `universe.py` set (BTC/ETH/SOL) and
prints a PROMOTE / DEAD / UNDERPOWERED verdict each. Bar: OOS net-positive,
PF>=1.2, survives 1.5x cost, n>=20. Nothing is wired to the live ticket / paper
ledger yet — that is a deliberate follow-up gated on which candidates survive.
First run (700d, 2026-07-07): all four DEAD — no candidate cleared the cost wall.
