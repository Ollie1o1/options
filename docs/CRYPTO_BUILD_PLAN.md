# Crypto Strategist — Build Plan & Progress

**Started:** 2026-05-01
**Last updated:** 2026-05-02

This is the running plan for the crypto-mode addition to the options screener.
Equity mode `[1]` stays as-is; crypto mode `[2]` is the new system.

---

## North-star goal

Build a research-grade tool that lets retail capture the few crypto edges that
still exist (volatility risk premium, term-structure carry, skew dynamics,
funding/basis arbitrage) on **defined-risk options structures** — never with
leveraged perpetuals.

Realistic Year-1 expectation: **−5% to +15%** annualized. Wide variance because
regime matters more than skill at small sample. Target Year-2+ if Year-1 is
profitable: **8–25% annualized.**

Probability of beating naive BTC HODL: **~30–40%** (active strategies sacrifice
upside in raging bulls in exchange for surviving bears).

---

## Architectural decisions (locked)

| # | Decision | Resolution |
|---|---|---|
| 1 | Repo structure | Single repo. Equity files at `src/*.py`, crypto at `src/crypto/*.py`, top-level launcher at `src/launcher.py`. |
| 2 | Database | Separate. Equity in `paper_trades.db`, crypto in `paper_trades_crypto.db`. Same schema; both share `paper_manager.py`. |
| 3 | Config | Shared `config.json`; crypto reads regime-multiplier sub-keys from the existing structure. |
| 4 | Cron | Equity cron unchanged (14:07 ET weekdays). Crypto cron not yet installed — would run hourly UTC since BTC is 24/7. |

---

## Phase 0–5 — DONE (commits `c468fe3`, `6758104`, `d3f31b0`, `644f002`)

### Phase 0 — Architecture sign-off ✅
Decisions above committed. Safety tag `pre-crypto-split-2026-05-01` placed.

### Phase 1 — Skeleton + menu split ✅
- `src/launcher.py` — top-level `[1] STOCKS / [2] CRYPTO / [Q]` menu
- `src/__main__.py` and `run.py` route through the launcher when no flags
- With ANY argv flag (`-ds`, `-sps`, `--enforce-exits`, etc.) the launcher
  dispatches DIRECTLY to `options_screener.main()` — preserves cron and
  every shortcut, zero behavior change for the equity path

### Phase 2 — Crypto data layer ✅
- `src/crypto/data_fetching.py` — Deribit chains, Binance funding, yfinance spot history
- `src/crypto/cache.py` — SQLite WAL cache for HTTP fetches (60s funding,
  5min chains, 1h history TTLs)
- All endpoints public, no auth required

### Phase 3 — Scoring + regime ✅
- `src/crypto/regime.py` — bull / chop / bear from 200d MA + rvol percentile
- `src/crypto/scoring.py` — 7 chain-level signals: `iv_rank`, `vrp`,
  `term_structure`, `skew`, `funding_z`, `basis`, `liquidity`
- `src/crypto/strategy.py` — strategy-aware ranking (Long Call / Long Put /
  Bull Put / Bear Call / Iron Condor) with regime-fit weights, target
  moneyness, target DTE bands. Includes `build_credit_spread_candidates`
  and `build_iron_condor_candidates`.

### Phase 4 — Crypto paper ledger ✅
- Reuses `PaperManager` with parameterized `db_path`
- `paper_trades_crypto.db` created on first paper trade
- Iron condor / spread / long-premium logging round-trips clean

### Phase 5 — UI ✅
- `src/crypto/screener.py` interactive sub-menu (BTC discover, ETH
  discover, funding/basis dashboard, portfolio, calibration, backtest)
- Strategy-bucketed output (per-regime filtering, hides incoherent
  strategies like Long Call in BEAR)
- Numbered + abbreviated log prompt (1, 2, lp, bc, "long put" all work)
- Chain signals row shows the 6 chain-level component scores inline

### Tier 1.1 — Backtester + snapshot accumulator ✅
- `src/crypto/chain_snapshot.py` — auto-saves every live Deribit fetch to
  `data/crypto_snapshots/<date>/<CURRENCY>.parquet`. After 30-60 days of
  daily scans, real chain data accumulates and the backtester switches
  from synthetic to real for those dates.
- `src/crypto/backtester.py` — walk-forward simulator. Synthesizes
  option chains via BS pricing where real snapshots are missing (IV
  assumption: 30d rolling realized vol × 1.10 historical ratio, with
  symmetric smile). Walks each trade forward via TP/SL/expiry exit
  using BS revaluation along the real spot path. Outputs per-strategy
  PF/win-rate, per-component IC, regime breakdown.

**First 1-year BTC backtest result (74 trades):**

| Strategy | N | Win% | Avg P&L | PF |
|---|---|---|---|---|
| **Bear Call** | 28 | 71% | +2.2% | **1.09x** |
| Bull Put | 10 | 60% | −48.8% | 0.38x |
| Long Put | 26 | 23% | −12.1% | 0.66x |
| Long Call | 8 | 12% | −31.2% | 0.29x |
| **ALL** | 74 | 45% | −14.7% | **0.62x** |

**Per-component IC (Spearman):**
- `skew_score` IC = **+0.291** ← strongest signal
- `vrp_score` IC = **−0.231** ← inverse (likely synthetic-IV bias)
- `moneyness_fit` IC = −0.202
- `dte_fit` IC = −0.162
- `term_structure_score` IC = +0.000 ← noise in this window

**Honest takeaway**: only Bear Call has positive EV in synthetic-chain
backtests. Long premium losses are exaggerated by the constant-IV
assumption (real Long Calls profit from IV expansion in real bull moves).
Treat the backtester as a *floor* on long-premium performance and a
*fair test* of credit-spread performance.

---

## Phase 6 — Paper-only validation (in progress, 4-6 weeks clock time)

**Goal:** 100 closed crypto paper trades with PF > 1.0.

**Process:**
- Run scans normally; auto-log surfaces top picks
- Cron-driven exit enforcement (not yet wired for crypto — see TODO)
- Weekly: review calibration_status, watch IC drift on snapshot history
- Goal at 30 trades: per-component IC stable, drop noise components
- Goal at 100 trades: confirm PF > 1.0 across at least one regime change

**TODO before Phase 6 can work end-to-end:**
- [ ] Wire `enforce_exits.sh`-equivalent for `paper_trades_crypto.db`
  (hourly UTC since crypto is 24/7)
- [ ] Calibration snapshot weekly cron for crypto DB
- [ ] Update `crontab -l` with the two new lines

---

## Tier 1.2–1.5 — IN PROGRESS / TODO

These add orthogonal signals that the existing system can't see, and
construct trades the system flags but doesn't execute.

### Tier 1.2 — Cross-exchange funding aggregation 🚧 NEXT
**Why**: Binance funding alone misses divergence — when one venue's funding
is materially different from others, that's an arb signal AND a leading
indicator of perp-spot mean reversion.

**Build**: REST pulls from Binance + Bybit + OKX + dYdX. Aggregate into a
single funding view. Compute cross-exchange divergence (max-min, std,
z-score). New signal `funding_divergence` for scoring. Display in the
funding/basis dashboard with all exchanges side-by-side.

**Estimate**: ~250 LOC, half a day.

### Tier 1.3 — OI tracking + surge detector
**Why**: OI surges with rising prices = leverage building (fade signal). OI
drops with falling prices = liquidations clearing (mean-reversion signal).
Combined with funding extremes, this is real fade-the-crowd alpha.

**Build**: Pull OI by exchange, persist daily into a small SQLite table,
compute z-score over 30-day window. Surface in chain signals row + as a
new scoring component.

**Estimate**: ~200 LOC, half a day.

### Tier 1.4 — Stablecoin supply tracker ✅
**Why**: USDT/USDC supply expansion has historically led BTC by 1-3 days.
A slow signal, fits multi-week DTE option positions.

**Built**: DefiLlama public API (free, no auth). Tracks USDT + USDC
circulating supply with 24h / 7d / 30d deltas, plus 7-day pct-change
z-score against 120-day distribution. Supply-weighted combined z-score
feeds the new `score_stablecoin_flow` component. Surfaced in:
- Chain signals diagnostic line (column "StableFlow")
- Live scan print: signed z-score + 7d pct change + direction
- Funding/basis dashboard: dedicated section with per-coin breakdown
- Backtester chain-quality calc

Magnitude-based scoring (matches funding_z / oi_surge convention) so
the strategy module's regime-fit handles direction. Boosted in bear
(1.20×) and bull (1.10×) regimes.

Live state at ship: combined z = −0.19σ (mild contraction), USDT 7d
−0.11%, USDC 7d −0.61%, 30d USDT +3.06%. No surge signal currently.

### Tier 1.5 — Calendar spread builder
**Why**: The system flags "strong contango" (term_structure score 0.96
today) but doesn't construct the calendar trade. A 7-DTE / 30-DTE ATM
calendar is the natural play to capture term-structure carry.

**Build**: Mirror `build_credit_spread_candidates` for calendars. Show
new strategy bucket "Calendar" when term structure is steep.

**Estimate**: ~150 LOC, an evening.

---

## Tier 2 — moderate ROI, more work

- ETF flow tracker (BITB, IBIT, FBTC daily flows — Farside scrape)
- Liquidation heatmap (Coinglass API)
- Multi-factor regime detector (price MA + vol + funding + dominance)
- Crypto-specific calibration UI mirroring equity's

## Tier 3 — lower priority

- WebSocket streaming (Deribit + Binance) — needed only if intraday
- Risk reversals + butterfly pricing
- On-chain metrics (Glassnode/CryptoQuant) — paid data, $40-200/month
- Cross-asset macro (DXY, yields, SPX) — unstable lags, hard to systematize

## Explicit non-goals

- ❌ Leverage / perp directional bets (negative EV for retail; structural)
- ❌ Altcoin support (illiquid options, manipulated spot)
- ❌ More technical indicators (RSI/MACD/Ichimoku — overfit trap, no edge)
- ❌ Sentiment scrapers (Twitter/Reddit — pure noise after costs)
- ❌ News integration with AI scoring (already in equity, adds nothing)
- ❌ Real-money trading until Phase 6 produces PF > 1.0 across 100+ trades

---

## Phase 7 — Real money, tiny

**Pre-conditions:**
- 100+ closed paper trades with PF > 1.0
- Confirmed positive IC on at least 2 components in real-snapshot backtests
- Stress test of the crypto book under −20% / +20pp IV scenarios
- Cron-driven exits running for ≥4 weeks without missed closes

**Initial sizing:** $500–$1000 of book on Deribit (cash-collateralized,
defined-risk only — credit spreads, iron condors). Naked positions and
directional perps banned.

**Sizing scale-up:** if 3 months of real-money produces PF within 0.2x of
paper-trade PF, scale to $5k. If real PF materially worse than paper,
stop and figure out the slippage gap before scaling.

---

## File map

```
options/
├── src/
│   ├── launcher.py                       # [1]/[2] menu dispatcher
│   ├── options_screener.py               # equity (unchanged)
│   ├── paper_manager.py                  # shared (parameterized db_path)
│   ├── utils.py                          # shared (BS pricing)
│   └── crypto/
│       ├── __init__.py
│       ├── cache.py                      # SQLite HTTP cache
│       ├── data_fetching.py              # Deribit + Binance + yfinance
│       ├── scoring.py                    # 7 chain-level signals
│       ├── strategy.py                   # strategy-aware ranking
│       ├── regime.py                     # bull/chop/bear classifier
│       ├── chain_snapshot.py             # auto-capture daily chains
│       ├── backtester.py                 # walk-forward simulator
│       └── screener.py                   # interactive crypto sub-menu
├── data/
│   └── crypto_snapshots/                 # gitignored — auto-fills
│       └── <YYYY-MM-DD>/<CURRENCY>.parquet
├── paper_trades.db                       # equity (gitignored)
├── paper_trades_crypto.db                # crypto (gitignored)
└── docs/CRYPTO_BUILD_PLAN.md             # this file
```

---

## Run commands

```bash
# Top menu
python3 run.py

# Equity direct (bypasses menu, preserves cron behavior)
python3 run.py -ds --10
python3 run.py -sps --5

# Crypto direct
python3 -m src.crypto.screener

# Backtest from inside crypto menu
python3 run.py → [2] → [6] → BTC → 365 → 7
```
