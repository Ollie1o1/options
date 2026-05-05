# Crypto Strategist — Build Plan & Progress

**Started:** 2026-05-01
**Last updated:** 2026-05-04

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
- [x] Wire `enforce_exits.sh`-equivalent for `paper_trades_crypto.db`
  (hourly since crypto is 24/7) ✅ — `src/crypto/exit_enforcer.py` +
  `scripts/enforce_exits_crypto.sh`. Cron line installed at `:05` every
  hour. Handles long premium / credit spreads / iron condors / calendars
  via Deribit pricing with same TP/SL/time-exit rules as equity.
- [x] Calibration snapshot weekly cron for crypto DB ✅ —
  `scripts/calibrate_snapshot_crypto.sh` (Sun 18:30 ET, after the
  equity calibrate at 18:13). Crypto-aware integrity check: skips the
  spread-PnL floor for `Calendar%` rows (they repurpose `spread_width`
  to encode days-between-expirations). Writes
  `logs/calibration_crypto_<DATE>.txt` and appends per-component IC to
  `logs/calibration_history_crypto.tsv` for drift tracking.
- [x] Update `crontab -l` with the new hourly crypto line + weekly
  crypto calibrate line

- [x] Auto-log driver (`src/crypto/auto_logger.py`) installed with cron
  line `0 */4 * * * auto_log_crypto.sh`. Off-switch defaults to `false`;
  flip `crypto.auto_log_enabled` to `true` in `config.json` to start the
  loop. See `## Auto-entry / auto-log cron — ACTIVE (paper-only)` below.

**Current crontab (5 jobs):**
```
7 14 * * 1-5  enforce_exits.sh                # equity, weekday afternoons
13 18 * * 0   calibrate_snapshot.sh           # equity, Sunday evenings
5 * * * *     enforce_exits_crypto.sh         # crypto, every hour 24/7
30 18 * * 0   calibrate_snapshot_crypto.sh    # crypto, Sunday evenings
0 */4 * * *   auto_log_crypto.sh              # crypto auto-log, every 4h (off-switch in config)
```

---

## Tier 1.2–1.5 — DONE

These add orthogonal signals that the existing system can't see, and
construct trades the system flags but doesn't execute. All four
shipped between 2026-05-02 and 2026-05-04.

### Tier 1.2 — Cross-exchange funding aggregation ✅
**Why**: Binance funding alone misses divergence — when one venue's funding
is materially different from others, that's an arb signal AND a leading
indicator of perp-spot mean reversion.

**Built**: REST pulls from Binance + Bybit + OKX + dYdX in
`data_fetching.fetch_aggregated_funding()`. Returns per-venue rates plus
aggregate stats (mean, std, max-min spread, divergence z). New
`score_funding_divergence` component (weight 1/10) lights up when the
spread between venues is large relative to its rolling distribution.
Surfaced in:
- Chain signals diagnostic row (column "FundDiv")
- Funding/basis dashboard (per-venue rates side-by-side + divergence z)
- Backtester chain-quality computation

### Tier 1.3 — OI tracking + surge detector ✅
**Why**: OI surges with rising prices = leverage building (fade signal). OI
drops with falling prices = liquidations clearing (mean-reversion signal).
Combined with funding extremes, this is real fade-the-crowd alpha.

**Built**: `data_fetching.fetch_oi_snapshot()` pulls OI from all 4 venues,
persisted daily to `crypto_cache.db`. `oi_z_score()` computes a 30-day
rolling z. New `score_oi_surge` component (weight 1/10) feeds the composite
score, with magnitude-based scoring so the strategy module's regime-fit
handles direction. Surfaced in chain-signals diagnostic row (column "OI"),
live scan print, and backtester.

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

### Tier 1.5 — Calendar spread builder ✅
**Why**: The system flagged steep term structure but didn't construct
the calendar trade. Single-strike calendars (long back-month + short
front-month) are theta-positive and capture term-structure carry.

**Built**: New `Calendar Call` and `Calendar Put` strategy definitions
with regime fits (chop=1.00 best, bull=0.65/0.50, bear=0.50/0.65).
`build_calendar_candidates()` pairs ATM strikes from front DTE ≈10
with back DTE ≈35 (must differ by ≥7 days), computes net debit, theta
edge proxy (1/√T_front − 1/√T_back), and applies a term-structure
penalty when back IV is ≥5% richer than front (deep contango makes the
debit too expensive).

UI integration:
- Surfaces as a strategy bucket in `_present_scan` (alongside long
  premium, credit spreads, iron condors).
- Dedicated `_print_calendar_table` showing strike, both expiries,
  both IVs, net debit, and score.
- Numbered + abbreviated log prompt accepts "Calendar Put", "cp",
  prefix matches, etc.
- `_log_calendar()` handler. The schema doesn't yet have a
  `back_expiration` column, so we encode the back expiry into
  strategy_name ("Calendar Put [back 2026-05-29]") and repurpose
  `spread_width` to hold days-between-expiries. Backtest can
  reconstruct from these.

Live state at ship: in current BEAR regime, Calendar Put surfaced as
the 3rd strategy alongside Long Put + Bear Call. Top pick: ATM $78k
put, front 11d / back 25d, IVs 37.0%/38.0%, debit $1,037, score 0.52.
Calendar Call hidden in BEAR (regime fit 0.50 < 0.55 threshold).

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
│       ├── data_fetching.py              # Deribit + 4-venue funding/OI + DefiLlama
│       ├── scoring.py                    # 10 chain-level signals
│       ├── strategy.py                   # strategy-aware ranking + builders
│       ├── regime.py                     # bull/chop/bear classifier
│       ├── chain_snapshot.py             # auto-capture daily chains
│       ├── backtester.py                 # walk-forward simulator
│       ├── exit_enforcer.py              # hourly auto-close (Deribit-priced)
│       └── screener.py                   # interactive crypto sub-menu
├── scripts/
│   ├── enforce_exits.sh                  # equity, daily 14:07 ET
│   ├── enforce_exits_crypto.sh           # crypto, hourly 24/7
│   └── calibrate_snapshot.sh             # equity, weekly Sun 18:13 ET
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

---

## Auto-entry / auto-log cron — ACTIVE (paper-only)

**Status:** built and installed (2026-05-04). Off-switch defaults to
`false` — flip `crypto.auto_log_enabled` to `true` in `config.json` to
start the loop. Driver: `src/crypto/auto_logger.py`. Wrapper:
`scripts/auto_log_crypto.sh`. Cron: `0 */4 * * *` (every 4 hours).

### Why this is now active for paper

The original deferral was framed around real-money risk (Phase 7). For
paper-only sample accumulation it is appropriate to enable now: paper
cannot blow up the account, and the closed-trade volume is itself the
gate. Real-money activation (Phase 7) still requires the original gates
plus an auto-log track record.

The intuition-building tradeoff is real but accepted: the bottleneck for
calibration is closed-trade volume, not screener-decision practice. The
weekly calibration snapshot replaces the daily review.

### Built decisions

| # | Decision | Value |
|---|---|---|
| 1 | Currencies | BTC and ETH |
| 2 | Cadence | Every 4 hours (6 runs/day) |
| 3 | Picks per run | 1 per currency (max 12/day before per-day cap) |
| 4 | Selection rule | Single highest score across all surfaced strategy buckets |
| 5 | Quality floor | `crypto.min_auto_log_score` (default `0.50`) |
| 6 | Per-day cap | 4 trades per currency per UTC day |
| 7 | Concentration cap | Skip if currency already has ≥3 open positions |
| 8 | Regime sanity | Skip if regime classifier returns `None` |
| 9 | Off-switch | `crypto.auto_log_enabled` (default `false`) |
| 10 | Dry-run period | None — writes directly to live ledger |

### Built artifacts

- `src/crypto/auto_logger.py` — driver + safeguards + CLI
- `scripts/auto_log_crypto.sh` — cron wrapper
- `tests/crypto/test_auto_logger.py` — 7 unit tests
- `config.json` — new `crypto.auto_log_enabled` and
  `crypto.min_auto_log_score` keys
- crontab line: `0 */4 * * * auto_log_crypto.sh`
- Auto-logged rows tagged via `weight_profile = 'crypto_auto_v1'` so
  manual logs (`crypto_baseline`) are distinguishable in IC analysis

### Mandatory safeguards (build alongside the auto-log mode)

1. **Quality floor**: skip a strategy bucket if its top score is below
   `config["crypto"]["min_auto_log_score"]` (recommended initial: 0.50).
2. **Stress-test gate**: skip the entire run if the crypto portfolio's
   current −20%/+20pp scenario loss exceeds 100% of nominal book.
3. **Per-day cap**: max N new logs per cron run (recommended: 2 per
   currency = 4 total). Prevents over-logging in calibration windows.
4. **Regime sanity**: skip if the regime classifier returned UNKNOWN
   (insufficient history).
5. **Concentration limit**: skip if the same ticker already has ≥3 open
   positions in `paper_trades_crypto.db`.
6. **Dry-run shadow ledger**: first 30 days of cron auto-log writes to
   `paper_trades_crypto.db.dryrun` instead of the real ledger so you
   can compare auto-log decisions against your manual decisions before
   committing to the live ledger.
7. **Stable score floor**: a 7-day rolling minimum on the score floor —
   don't auto-log on a regime change day where scores haven't settled.

### Off-switch

The first thing the auto-log mode should check (before anything else):
read `config["crypto"]["auto_log_enabled"]` (default `false`). One-line
config change disables the entire auto-log cron without touching
crontab.

### Why this list of safeguards

Each one corresponds to a specific failure mode we've already seen in
the equity ledger:
- **Quality floor** → equity Long Put PF 0.69 was visible because
  scores were average; auto-logging would have made it worse.
- **Stress gate** → equity portfolio hit −183% of book stress in
  late April; user manually backed off `-ss`. Cron wouldn't have.
- **Per-day cap** → ORCL got logged 6× in a single scan once; dedup
  filter exists but a quality-floor breach could still over-log.
- **Concentration** → 23 ICs accumulated in 4 days created the
  −$37k stress scenario.
- **Dry-run** → catches the entire class of "system thought it was
  reasonable but the data was malformed" bugs that have come up
  three times in this build.

Build the safeguards FIRST, then the auto-log mode, then enable the
cron. Reverse order = blow up the ledger.
