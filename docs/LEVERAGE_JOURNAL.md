# Leverage Validation Journal

Append-only record of validation-gate evaluations for the BTC/ETH perp leverage
strategy (`src/leverage/`). Criteria are defined in
`docs/superpowers/specs/2026-06-02-leverage-module-build-design.md`
(§"Validation gate"). **No real money** until ALL five criteria pass and are
recorded here:

1. Walk-forward net-of-cost expectancy > 0 in **every** OOS window.
2. Profitable across ±20% param perturbation on BTC **and** ETH.
3. ≥100 paper trades over ≥4 weeks; bootstrapped 95% CI on per-trade
   expectancy excludes 0.
4. Observed max DD ≤ 1.5× backtest max DD.
5. Kill criterion: CI includes 0 at 100 trades ⇒ rejected, no overrides.

---

## 2026-06-02 — module built (P0–P3), pre-validation

`src/leverage/` implemented: data (Binance fapi + Bybit fallback), pure 6-param
signal engine, conservative event-driven backtest + walk-forward/robustness
harness, sqlite perps paper ledger, risk/liquidation rules, Trade Ticket +
`status` CLI. Real-money path inert behind `--live`.

No gate criteria evaluated yet. Next: run
`python -m src.leverage backtest --symbol BTC --walk-forward` and `--robustness`
on BTC and ETH to evaluate criteria #1 and #2.

---

## 2026-06-03 — gate criteria #1 & #2 evaluated: BOTH strategies FAIL (no edge)

**Verdict: stay in paper/observation. Do NOT deploy. Real money remains off.**

### What was built this session
- `reversion.py` — mean-reversion signal (fade |z|≥z_entry vs 20-bar mean,
  target = the mean, stop = atr_stop_mult·ATR). Motivated by the research below.
- `backtest.py` — made strategy-agnostic (`signal_fn=`), and it now records each
  trade's side + exit reason.
- `analysis.py` — risk/performance report: Sharpe, Sortino, profit factor, max DD,
  win/loss asymmetry, **per-side breakdown**, exit-reason mix.
- `optimize.py` — leak-free train/test holdout grid search over reversion params.
- CLI/menu — `--strategy {reversion,breakout}`, `optimize` subcommand; menu
  defaults to reversion + `[5] OPTIMIZE`, shows risk analysis. Menu network actions
  are crash-guarded; `data.load_history` falls back Binance→Bybit and serves the
  cache offline when both fail. 76 tests green.

### Signal research (cached BTC/ETH 5m, ~550d, IS/OOS 70/30)
- Breakout/momentum (the original signal) is **sub-coinflip** at 5–15min: 47–49% OOS.
- Mean-reversion fade z>2σ is **~56% directional**, stable 54–57% across 5 time-blocks,
  both coins — the "≥51% for now" goal was met at the *signal* level.

### But neither converts to a tradeable edge (gate #1 & #2)
| strategy | walk-forward (#1) | robustness (#2) | full-period |
|---|---|---|---|
| breakout | 0/6 OOS positive | 0/15 (BTC), 1/15 (ETH) | NO EDGE, ~0 gross |
| reversion | 0/6 OOS positive | — | NO EDGE, −100% (ruin) |

**Root cause (the key finding):** reversion's per-side split is short win 82–85%
(+2.0–2.3%) vs long win 8–10% (−2.2–2.4%) — a near-mirror that is **pure
down-drift** (BTC −32%, ETH −48% over the sample), not reversion skill. "Fade
rallies" = "short a bear market"; it inverts in a bull market. The honest optimizer
made this explicit: best train params showed 95% short win ("EDGE") that collapsed
to profit-factor 0.54 / NO EDGE on held-out test. Tuning does not rescue it.

### Gate status
- Criterion #1 (walk-forward all-positive): **FAIL** (0/6, both strategies).
- Criterion #2 (robustness all-positive): **FAIL**.
- Criteria #3–5: not reached (no paper accumulation — would be wasted on a signal
  that fails #1/#2).

### To circle back to
1. The only lead worth exploring: tiny fixed targets (~0.3–0.5%, matching the 15min
   reversion) + much shorter holds + a **drift/regime neutralizer** so the short bias
   isn't doing the work. Cost (~0.11%/round-turn) is the hard hurdle at that scale.
2. Re-run `python -m src.leverage backtest --strategy reversion --walk-forward` and
   `optimize` periodically as more data accrues; gate auto-refreshes the menu banner.
3. Live `signal` currently returns "no setup" on BTC/ETH (nothing stretched).
4. Research harness for the IS/OOS rule battery lives in `/tmp/lev_research.py` +
   `/tmp/lev_confirm.py` (ephemeral — recreate from this journal / memory note
   `project_leverage_signal_research` if needed; analysis+optimize logic is permanent).

### Repo state
All session work (UI wiring, menu/data resilience, reversion strategy, analysis,
optimizer) is **uncommitted** on `main` as of 2026-06-03.

---

## 2026-06-07 — gate re-checked, verdict UNCHANGED: still no edge

**Verdict: real money remains OFF.** Nothing has changed the 2026-06-03 conclusion.
This is a status re-confirmation, not new development.

### What was checked
- `signal BTC` / `signal ETH` (reversion): **no actionable setup** on the latest
  bar for either coin — nothing stretched ≥ z_entry.
- `backtest BTC --strategy reversion --walk-forward`: **0/6 OOS windows positive**
  (gate #1 still FAIL). Per-window expectancy −0.0010 to −0.0014, win 32–48%,
  max DD ~86–92%. Identical shape to 2026-06-03 — no improvement.
- Paper ledger (`paper_trades_leverage.db`): **still empty** (0 open, 0 closed,
  $0.00 realized). Correct — no paper accumulation while #1/#2 fail.
- Leverage test suite: **77 tests green** (`unittest discover tests/leverage`).

### Data freshness — confirmed CURRENT (not stale)
Verified the 5m cache tops up to **2026-06-07 21:35 UTC** for both BTCUSDT and
ETHUSDT (159,903 / 159,899 bars from 2024-11-29). Binance fapi + Bybit both return
HTTP 200 from this shell — the network path works. The walk-forward report's last
window ending **2026-05-29** is *not* stale data: `walk_forward_windows` only emits
**complete 2-month OOS blocks**, so the final ~9 days (2026-05-29 → 06-07) don't yet
fill a window and are correctly excluded. So today's 0/6 FAIL is on a fresh, full
~18-month sample — it strengthens, not merely repeats, the 2026-06-03 verdict.

### Gate status (unchanged)
- #1 walk-forward all-positive: **FAIL** (0/6, both coins, both strategies).
- #2 robustness all-positive: **FAIL** (per 2026-06-03).
- #3–5: not reached.

### NEW TOOL + NEW FINDING — drift neutralizer refutes lead #1

Built `backtest.neutralize_drift(df5, df15)` + a `backtest --neutralize-drift` CLI
flag (5 unit tests, suite now 82 green). It detrends OHLC so cumulative drift → 0
while leaving bar-to-bar oscillation (ATR/bands/z-scores) intact, to test whether a
strategy has edge *independent of the sample's trend* — directly probing the
2026-06-03 "it's just shorting a bear market" root-cause and the "circle back" lead #1
("add a drift/regime neutralizer so the short bias isn't doing the work").

**Result (full-period reversion, raw → drift-neutralized):**

| coin | drift removed | net exp RAW | net exp NEUTRAL | short/long win RAW | short/long win NEUTRAL |
|---|---|---|---|---|---|
| BTC | −26%/yr | −0.2072%/tr | −0.1760%/tr | 85.5% / 8.0% | 85.3% / 8.1% |
| ETH | −41%/yr | −0.2927%/tr | −0.2655%/tr | 81.3% / 9.9% | 81.1% / 10.1% |

**Interpretation — lead #1 is a dead end.** Removing the drift barely moves the needle
(+0.03 pp/trade) and the strategy is still NO EDGE, still −100% ruin, with the per-side
85/8 win asymmetry essentially **unchanged**. Reason: at ≤4h holds the macro drift is
only ~0.01%/trade — negligible vs the ±2.4% per-trade swings and the 0.13% round-turn
cost. So the journal's earlier "the short bias is doing the work" story is, at the
*per-trade* level, incomplete: the long/short asymmetry is a **drift-independent
mechanical artifact** of the fade geometry (target = 20-bar mean, stop = ATR mult,
worst-case stop-before-target fill), not trend capture. A drift/regime neutralizer will
NOT rescue reversion — do not spend effort there.

### NEW FINDING — lead #1 (micro-targets) is also a dead end: cost wall is 22×

Ran BTC reversion full-period with costs+funding zeroed to isolate the **gross** edge:

| | expectancy/trade |
|---|---|
| NET (0.13% round-turn + funding) | **−0.2072%** |
| GROSS (zero cost, zero funding) | **+0.0060%** (long +0.0041%, short +0.0082%) |

So there IS a *real but microscopic* reversion signal (+0.6 bps/trade gross, both sides
positive) — it's just **~22× too small** to clear the 13 bps round-turn cost. This kills
the "tiny fixed targets (~0.3–0.5%)" lead from 2026-06-03: shrinking the target shrinks
the gross move (and thus the +0.6 bps further), while the 13 bps cost is fixed — micro-
targets make the edge:cost ratio **worse**. The only way this trades is to attack *cost*
(maker-only/rebate fills + zero slippage, getting round-turn under ~0.6 bps) AND have the
signal survive at that scale — both implausible. **Verdict: stop optimizing the entry
rule; the binding constraint is execution cost, not signal.**

---
