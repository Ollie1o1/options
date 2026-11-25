# Playbook 4 Result: Multi-Week PEAD (`pead_breakout`) & Estimate Revisions Status

**Date:** 2026-11-24
**Preregistration:** [PREREG_PEAD_AND_REVISIONS_20261124.md](PREREG_PEAD_AND_REVISIONS_20261124.md)
**Estimators:** [scripts/backtest_pead_equity.py](../scripts/backtest_pead_equity.py) · [scripts/estimate_revisions_test.py](../scripts/estimate_revisions_test.py)
**Raw Artifact:** [reports/backtest_pead_equity_20261124.json](../reports/backtest_pead_equity_20261124.json)
**Verdict:** **Arm A (`pead_breakout`) = `FAIL` (`CLOSED` on Dolt large-caps)** · **Arm B (`estimate_revisions`) = `LOCKED` until `2027-01-15`**

---

## 1. Arm A — Multi-Week PEAD Reaction Drift (`data/dolt_options.db` + `squeeze_prices.db`, `2020–2026`, `10 bps` RT)

| Cohort | Trades | Month Clusters | Win Rate | Mean Net Ret [95% CI] | PF [95% CI] | Excess vs SPY [95% CI] | H1 / H2 Excess | Verdict |
|---|---:|---:|---:|---|---|---|---|---|
| **Primary `pead_pos_20d`** (`+5%`, `>=2x` vol, 20d) | 614 | 74 | 56.0% | +1.10% [-0.68%, +2.78%] | 1.296 [0.86, 2.01] | +0.04% [-0.88%, +0.99%] | +0.42% / -0.26% | **`FAIL`** |
| Secondary `pead_pos_40d` (40d) | 610 | 73 | 52.6% | +1.50% [-0.91%, +3.88%] | 1.277 [0.87, 1.91] | -0.25% [-1.62%, +1.10%] | +0.04% / -0.49% | **`FAIL`** |
| Diagnostic `pead_neg_20d` (`-5%`, `>=2x` vol, 20d) | 571 | 76 | 47.8% | -0.42% [-2.46%, +1.38%] | 0.902 [0.58, 1.43] | **`-1.16% [-2.19%, -0.19%]`** | **-1.39% / -1.00%** | **AVOID FLAG** |
| Control `non_earn_pos_20d` (non-earn `+5%`, `>=2x` vol, 20d) | 939 | 78 | 55.4% | **+3.00% [+0.71%, +5.21%]** | **1.615 [1.12, 2.34]** | +1.41% [-0.43%, +3.16%] | +1.28% / +1.55% | Control only |

### Key Findings from Arm A
1. **Large-cap positive PEAD is effectively priced in (`+0.04%` excess vs SPY):** Consistent with modern institutional coverage of S&P 500/Nasdaq-100 optionable names — no tradeable alpha at 20d or 40d. Marked **`CLOSED`** in `configs/playbooks.json`.
2. **Actionable Risk Flag — Negative Earnings Breakdown Drift (`pead_neg_20d`):** Stocks that gap/sell off `<= -5%` on `>= 2x` volume on earnings lag SPY by **`-1.16%` over the next 20 sessions** (95% clustered CI `[-2.19%, -0.19%]`, negative in both `2020–2023H1` and `2023H2–2026`). Surfaced on Radar & Dossier as a **`POST_EARN_CRASH`** risk flag to warn against buying dips right after an earnings breakdown.
3. **Unpreregistered Control Lead (`non_earn_pos_20d`):** Non-earnings `+5% / 2x vol` breakouts earned `+3.00%` (`PF 1.615`, CI strictly > 1.0, consistent across H1 `+3.02%` and H2 `+2.98%`). Per `RESEARCH_PROCESS.md`, a control arm is **not** promoted automatically; logged for future preregistered testing on a broader/out-of-sample universe.

---

## 2. Arm B — Point-in-Time Analyst Estimate Revisions (`data/estimates_archive.db`)

- **Status:** Strictly date-gated (`UNLOCK_DATE = 2027-01-15`; returns exit code `2` before unlock).
- **Coverage (`--coverage-only` as of 2026-11-24):** 6 snapshots (`2026-10-14` .. `2026-11-23`), **2 of 4** minimum non-overlapping (`>= 21d` gap) snapshots (`2026-10-14` with 91 symbols; `2026-11-09` with 104 symbols). Archive collection continues automatically via `scripts/snapshot_estimates.py`.
