# Playbook 3 Result: Path-Managed Squeeze Momentum (`squeeze_equity`)

**Date:** 2026-11-24
**Preregistration:** [PREREG_SQUEEZE_PATH_EQUITY_20261124.md](PREREG_SQUEEZE_PATH_EQUITY_20261124.md)
**Estimator:** [scripts/backtest_squeeze_equity.py](../scripts/backtest_squeeze_equity.py)
**Raw Artifact:** [reports/backtest_squeeze_equity_20261124.json](../reports/backtest_squeeze_equity_20261124.json)
**Verdict:** **`FAIL` across all 3 arms (`S1`, `S2`, `S3`) and `HOLD42` → Tier: `CLOSED`**

---

## 1. Ex-Meme Moving-Block Results (`40 bps` RT, Bonferroni `98.33%` CI, block length = 4 fortnight clusters)

| Arm | Trades (Ex-Meme) | Clusters | Win Rate | Median | Mean Net Ret [98.33% CI] | PF [98.33% CI] | Holdout 2023–26 Mean [95% CI] | 4-Slot CAGR / MaxDD | Verdict |
|---|---:|---:|---:|---:|---|---|---|---|---|
| `S1` (-12% stop/trail) | 1,853 | 189 | 37.1% | -5.81% | +3.04% [-1.21%, +8.47%] | 1.399 [0.85, 2.20] | +2.54% [-1.68%, +8.48%] | +18.60% / -52.39% | **`FAIL`** |
| `S2` (Primary: -12% / +20% half / -15% trail) | 1,804 | 189 | 38.8% | -6.79% | +2.15% [-1.30%, +6.26%] | 1.263 [0.86, 1.84] | +1.44% [-1.88%, +5.55%] | +13.65% / -43.51% | **`FAIL`** |
| `S3` (-12% stop / +25% TP) | 1,902 | 188 | 39.6% | -9.61% | +1.64% [-1.56%, +5.00%] | 1.179 [0.85, 1.61] | +0.63% [-2.30%, +3.55%] | +3.92% / -45.75% | **`FAIL`** |
| `HOLD42` (Unstopped Control) | 1,508 | 185 | 47.6% | -1.94% | +5.32% [-2.91%, +14.26%] | 1.447 [0.80, 2.47] | +3.53% [-5.29%, +14.47%] | **-1.21%** / -62.44% | **`FAIL`** |

### Sensitivities & Outlier Diagnostics
- **`S2` @ 100 bps RT friction:** Mean `+1.55%` `[-1.90%, +5.66%]`, PF `1.181` → `FAIL`.
- **`S2` @ `$20M` min dollar volume:** Mean `+2.24%` `[-1.45%, +6.83%]`, PF `1.279` → `FAIL`.
- **Drop Top-3 Tickers (`AXSM`, `GME`/`CAR`, `QMMM`):** Cuts `S1` mean return from `+3.04%` to `+1.20%` (`PF 1.157`) and `S2` from `+2.15%` to `+1.19%` (`PF 1.146`).

---

## 2. Key Takeaways

1. **Extreme right-tail variance defeats systematic significance:** Although point estimates are positive (`+1.6%` to `+3.0%` per trade), ~61% of trades lose, median trade is `-5.8%` to `-9.6%`, and even un-Bonferroni'd 95% moving-block CIs span zero across every arm.
2. **Path management DID transform compounding:** Unstopped `HOLD42` suffered 56 delistings (`-100%` write-offs) and a **`-1.21%` compounded CAGR** (`-62.4%` MaxDD) on a 4-slot book despite a `+5.32%` arithmetic mean; `S1` / `S2` turned that into `+18.6%` / `+13.7%` compounded CAGR by truncating left-tail wipeouts.
3. **Production Action:** Marked **`CLOSED`** in `configs/playbooks.json` — excluded from the default daily RADAR feed so it does not waste attention. Remains accessible only under `opt lab` / Squeeze research tools.
