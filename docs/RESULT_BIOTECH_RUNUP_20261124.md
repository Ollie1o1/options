# Playbook 6 Result: Funded Biotech Pre-Catalyst Run-Up (`biotech_runup`)

**Date:** 2026-11-24
**Preregistration:** [PREREG_BIOTECH_RUNUP_AND_INSIDER_20261124.md](PREREG_BIOTECH_RUNUP_AND_INSIDER_20261124.md)
**Estimator:** [scripts/backtest_biotech_runup.py](../scripts/backtest_biotech_runup.py)
**Raw Artifact:** [reports/backtest_biotech_runup_20261124.json](../reports/backtest_biotech_runup_20261124.json)
**Verdict:** **H6-A = `FAIL` / H6-B = `FAIL` → Tier: `CLOSED`**

---

## 1. Point-in-Time Results (`catalyst_pit.db`, `T-45 → T-3`, `50 bps` RT, `2022–2025`)

| Bucket | Trades | Tickers | Quarters | Win Rate | Median | Mean Net Ret [95% Ticker CI] | PF [95% Ticker CI] | Excess vs XBI [95% Ticker CI] | Verdict |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| **Primary `B_FUNDED_T45_T3`** (`FUNDED THROUGH`) | 215 | 94 | 13 | 38.6% | -7.03% | +0.81% [-3.70%, +5.63%] | 1.070 [0.71, 1.55] | -0.32% [-4.38%, +3.89%] | **`FAIL`** |
| Quarter-Clustered Check (`B_FUNDED_T45_T3`) | 215 | 94 | 13 | 38.6% | -7.03% | +0.81% [-7.48%, +10.89%] | 1.070 [0.47, 2.22] | -0.32% [-4.87%, +4.97%] | **`FAIL`** |
| Control `CONTROL_RAISE_BEFORE` | 15 | 7 | 8 | 53.3% | +0.22% | +8.12% (n < floor) | 1.648 | +2.73% | `INSUFFICIENT` |
| Control `UNKNOWN_RUNWAY` | 35 | 20 | 12 | 42.9% | -5.79% | +1.37% (n < floor) | 1.142 | -2.34% | `INSUFFICIENT` |

---

## 2. Interpretation & Tier Decision

1. **ClinicalTrials.gov Primary Completion Dates (`PCD`) do not produce a systematic run-up:** Across 215 point-in-time `FUNDED THROUGH` biotech events (`94` distinct tickers), median return is `-7.03%`, win rate `38.6%`, and excess return vs `XBI` is `-0.32%` (`CI [-4.38%, +3.89%]`). This directly confirms our preregistered prior: `PCD` is an operational registry estimate that frequently slips, unlike exact FDA PDUFA calendar dates.
2. **Production Action:** Marked **`CLOSED`** in `configs/playbooks.json` — excluded from the default daily RADAR feed. The catalyst board (`src/catalyst/`) remains available in `opt lab` purely as a risk-calendar lookup (`avoid holding options across clinical readouts`).
3. **Insider Cluster Buy (`insider_cluster`):** Remains **`CONTEXT`** until the EDGAR Form 4 backtest corpus (`docs/superpowers/specs/2026-09-21-insider-cluster-backtest-design.md`) is completed.
