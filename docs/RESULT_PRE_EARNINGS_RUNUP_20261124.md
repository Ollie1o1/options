# Test O2 Result: Pre-Earnings Anticipation Window (`T-10 → T-1`, Closed Before Report)

**Date:** 2026-11-24
**Preregistration:** [PREREG_OPTIONS_DTE_AND_PRE_EARNINGS_20261124.md](PREREG_OPTIONS_DTE_AND_PRE_EARNINGS_20261124.md)
**Estimator:** [scripts/backtest_pre_earnings_runup.py](../scripts/backtest_pre_earnings_runup.py)
**Raw Artifact:** [reports/backtest_pre_earnings_runup_20261124.json](../reports/backtest_pre_earnings_runup_20261124.json)
**Verdict:** **`FAIL` across all 3 claimants (`O2-EQ`, `O2-LC`, `O2-ST`) → Tier: `CLOSED`**

---

## 1. Primary Results (`data/dolt_options.db`, `2020–2026`, Bonferroni `98.33%` Week-Clustered CI)

All 2,431 matched events were strictly closed on or before `D - 1` (zero overnight binary announcement exposure; 2,522 events dropped for missing snaps/cutoffs, never held through reports).

| Claimant | Events | Weeks | Win Rate | Mean Net Ret [98.33% CI] | Excess vs SPY [98.33% CI] | PF [98.33% CI] | Verdict |
|---|---:|---:|---:|---|---|---|---|
| **`O2-EQ`** (Shares, 10 bps RT) | 2,278 | 257 | 56.7% | +0.70% [+0.12%, +1.29%] | **-0.01% [-0.34%, +0.31%]** | 1.414 [1.061, 1.916] | **`FAIL`** |
| **`O2-LC`** (Long ATM Call, Crossed) | 1,437 | 232 | 36.2% | **-7.55% [-13.96%, -1.29%]** | -7.55% | 0.693 [0.499, 0.940] | **`FAIL`** |
| **`O2-ST`** (Long ATM Straddle, Crossed) | 952 | 204 | 17.3% | **-11.37% [-13.16%, -9.56%]** | -11.37% | 0.237 [0.167, 0.319] | **`FAIL`** |

---

## 2. Sensitivities (Mid-Fill Upper Bound, Exact Quotes, Mega-Caps)

| Sensitivity | Events | Weeks | Win Rate | Mean Ret [98.33% CI] | PF | Takeaway |
|---|---:|---:|---:|---|---:|---|
| `O2-LC_mid` | 1,437 | 232 | 50.4% | **+8.98% [+2.00%, +15.81%]** | 1.515 | Market drift + IV lift = +9.0% at mid, destroyed by 16.5% crossing wall |
| `O2-ST_mid` | 952 | 204 | 40.8% | +1.69% [-0.24%, +3.66%] | 1.283 | Pure IV expansion barely beats theta even at mid; crushed at touch (-11.4%) |
| `O2-LC_crossed_exact_only` | 1,353 | 218 | 35.8% | -7.85% [-14.13%, -1.62%] | 0.679 | Strike interpolation has negligible effect |
| `O2-ST_crossed_exact_only` | 881 | 190 | 16.9% | -11.67% [-13.41%, -9.88%] | 0.223 | Consistent with full sample |
| `O2-EQ_megacap` | 252 | 101 | 59.9% | +1.04% [-1.26%, +2.65%] | 1.530 | Still indistinguishable from market drift |
| `O2-ST_crossed_megacap` | 61 | 36 | 22.9% | -3.41% [-9.92%, +4.11%] | 0.625 | Tighter spreads reduce loss from -11.4% to -3.4%, still negative |

---

## 3. Conclusion

1. **No pre-earnings share alpha (`O2-EQ`):** The `+0.70%` raw 2-week gain equals SPY's drift to the basis point (`-0.01%` excess, `CI [-0.34%, +0.31%]`).
2. **Pre-earnings option buying loses in every single year (`2020..2026`) at crossed fills:** Both `O2-LC` (`-7.55%`) and `O2-ST` (`-11.37%`) had negative returns in all 7 calendar years.
3. **Production Action:** Marked **`CLOSED`** in `configs/playbooks.json` — excluded from the default Radar feed.
