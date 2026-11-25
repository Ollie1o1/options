# Playbook 1 Result: Uptrend Pullback Mean Reversion (`equity_pullback`)

**Date:** 2026-11-24
**Preregistration:** [PREREG_EQUITY_PULLBACKS_20261124.md](PREREG_EQUITY_PULLBACKS_20261124.md)
**Estimator:** [scripts/backtest_equity_pullbacks.py](../scripts/backtest_equity_pullbacks.py)
**Raw Artifact:** [reports/backtest_equity_pullbacks_20261124.json](../reports/backtest_equity_pullbacks_20261124.json)
**Verdict:** **H-A = `PASS` / H-B = `FAIL` → Tier: `LEAD`**

---

## 1. Frozen Rule Tested

Long shares (`t+1` open/close causal fill, `10 bps` round-trip friction) when:
`Close > SMA200` AND `12-1m mom > 0` AND `RSI(2) < 10` AND `3d ret <= -3%` AND `Close >= $10` AND `20d med $Vol >= $50M`, excluding known earnings inside the next 5 sessions. Exit on first of `Close > SMA(5)` or `10` trading sessions.

---

## 2. Primary & Secondary Holdout Results (Purged Split: Train `<= 2022-12-31`, Holdout `>= 2023-01-16`)

| Universe / Arm | Split | Trades | Wk Clusters | Win Rate | Mean Net Ret [95% Clust CI] | PF [95% Clust CI] | Excess vs SPY [95% CI] | H-A | H-B |
|---|---|---:|---:|---:|---|---|---|---|---|
| **Primary Broad** (`squeeze_prices.db`, 13,890 syms, `t+1`, 10 bps) | Train | 25,448 | 222 | 60.4% | +0.252% [-0.227%, +0.702%] | 1.140 [0.892, 1.461] | -0.001% [-0.271%, +0.256%] | — | — |
| **Primary Broad** | **Holdout** | **25,629** | **185** | **60.3%** | **+0.388% [+0.036%, +0.731%]** | **1.232 [1.019, 1.496]** | **+0.011% [-0.223%, +0.233%]** | **`PASS`** | **`FAIL`** |
| **Primary Broad** | Full (2018–26) | 51,253 | 410 | 60.4% | +0.325% [+0.020%, +0.607%] | 1.188 [1.010, 1.389] | +0.005% [-0.175%, +0.176%] | — | — |
| **Secondary Liquid** (`equity_ohlcv.db`, 62 syms, `t+1`, 10 bps) | **Holdout** | **971** | **173** | **62.2%** | **+0.553% [+0.114%, +0.996%]** | **1.431 [1.076, 1.940]** | **+0.179% [-0.127%, +0.487%]** | **`PASS`** | **`FAIL`** |
| Sensitivity: Broad MOC (10 bps) | Holdout | 26,476 | 185 | 64.7% | +0.436% [+0.111%, +0.754%] | 1.265 [1.060, 1.511] | +0.050% [-0.191%, +0.281%] | `PASS` | `FAIL` |
| Sensitivity: Broad `t+1` (20 bps) | Holdout | 25,629 | 185 | 59.2% | +0.288% [-0.064%, +0.631%] | 1.168 [0.967, 1.419] | -0.089% [-0.323%, +0.133%] | `FAIL` | `FAIL` |

### Realistic 5-Slot Capacity Book (Primary Broad, 20% Equity/Slot)
- **Full (`2018-10-03..2026-08-07`, `n=2,372`):** CAGR **`+19.98%`** vs SPY `+14.94%`, Sharpe `0.93`, MaxDD `-48.79%`.
- **Holdout (`2023-01-24..2026-08-07`, `n=1,063`):** CAGR **`+20.13%`** vs SPY `+22.02%`, Sharpe `0.97`, MaxDD `-26.70%`.
- **Secondary Liquid 5-Slot Full (`n=1,020`):** CAGR `+3.94%`, MaxDD `-20.16%` (cash-heavy because 62 symbols leave slots idle most days).

---

## 3. Annual Regime Profile (Primary Broad, 10 bps)

| Year | Trades | Win Rate | Mean Net Ret | Excess vs SPY | PF |
|---|---:|---:|---:|---:|---:|
| 2018 (Q4) | 1,332 | 48.5% | -1.45% | -0.49% | 0.507 |
| 2019 | 3,530 | 67.3% | +0.58% | -0.00% | 1.476 |
| 2020 | 5,497 | 58.6% | +0.18% | +0.25% | 1.086 |
| 2021 | 10,852 | 64.1% | +0.69% | +0.02% | 1.465 |
| 2022 | 4,237 | 51.3% | -0.51% | -0.21% | 0.784 |
| 2023 | 4,929 | 57.6% | +0.14% | -0.28% | 1.091 |
| 2024 | 7,909 | 65.1% | +0.74% | -0.03% | 1.590 |
| 2025 | 6,983 | 61.0% | +0.41% | +0.14% | 1.218 |
| 2026 | 5,984 | 55.7% | +0.14% | +0.15% | 1.068 |

---

## 4. Interpretation & Tier Decision

1. **Genuine standalone positive expectancy at stock friction (`<=10 bps`), but NOT proven alpha over SPY:** Both primary (survivorship-free 13,890-symbol universe) and secondary (`equity_ohlcv.db`) holdouts clear H-A (`mean_ret` CI > 0 and `PF` CI > 1.0), and `known_clear` earnings holdout trades perform slightly better (`+0.425%` broad / `+0.647%` liquid) than unknown-calendar names. However, matched-window excess over SPY is flat (`+0.011%` broad, `+0.179%` liquid, CIs span zero → **H-B FAIL**), and clustered `t = 1.41` (`1.86` liquid) does not clear the Harvey `t >= 3.0` promotion gate (`alloc = reject`).
2. **Cost sensitivity proves why equity execution matters:** Doubling round-trip friction from `10 bps` to `20 bps` pushes the holdout CI lower below zero (`PF CI [0.967, 1.419]`). This setup only works in **shares ($0 commission, 2–5 bps spread)** — never via 15%-spread short-dated options!
3. **Bear-regime hazard:** Lost money in `2018` and `2022` (broad equity bear markets). On the Radar, `equity_pullback` is tagged **`LEAD`** and automatically receives a `RISK_OFF` / `VIX_BACKWARDATION` warning flag whenever macro posture turns defensive.
