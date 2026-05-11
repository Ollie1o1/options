# Calibration Journal

A timestamped log of every `--calibrate --apply` event against `config.json`.
Each entry captures the IC snapshot, the shrinkage, the actual recommended
weight changes, and the rationale. Read top-down to see how the model has
evolved over time.

---

## 2026-05-11 — first IC-driven directional calibration

**Command:** `python -m src.backtester --calibrate --apply`
**Backup file:** `config.bak.20260511-111722.json`
**Trigger:** 182 closed paper trades (well past the 25-trade hard floor;
shrinkage λ = 182 / (182 + 60) = 0.752, IC weighted 75% vs current 25%).
**Scope:** `composite_weights` only (single-leg / directional bucket). Spread
weights and iron-condor weights left untouched — they live in separate
sections and are gated by their own 30-closed thresholds (Bear Call needs 10
more, IC needs 24 more).

### IC snapshot (top signals)

| Component | IC | Direction |
|---|---|---|
| vrp | +0.212 | strong positive |
| term_structure | +0.197 | strong positive |
| vega_risk | +0.168 | strong positive |
| gamma_theta | -0.159 | strong negative |
| pop | -0.143 | strong negative (counterintuitive — see notes) |
| spread | +0.142 | positive |
| iv_velocity | +0.130 | positive |
| iv_edge | +0.114 | positive |
| theta | -0.110 | negative |

### Biggest weight shifts (Δ ≥ 0.05)

| Component | Old | New | Δ |
|---|---|---|---|
| pop | 0.1300 | 0.0322 | **-0.098** |
| vrp | 0.0500 | 0.1401 | **+0.090** |
| term_structure | 0.0400 | 0.1287 | **+0.089** |
| vega_risk | 0.0300 | 0.1088 | **+0.079** |
| spread | 0.0100 | 0.0881 | **+0.078** |
| momentum | 0.1000 | 0.0387 | -0.061 |
| liquidity | 0.0800 | 0.0198 | -0.060 |
| ev | 0.0700 | 0.0174 | -0.053 |
| rr | 0.1000 | 0.0491 | -0.051 |

### Interpretation

The book's first IC-driven calibration was a vol-edge correction. The
priors (set by hand-tuned heuristics) over-weighted directional/PoP signals
(pop, momentum, rr, liquidity) and under-weighted volatility-surface
signals (vrp, term_structure, vega_risk, iv_velocity, spread). 182 closed
trades pulled the weights toward vol-edge dominance, which matches the
canonical options-trading principle: trade the surface, not the direction.

The negative IC on `pop` (-0.143) is the most striking finding. High-PoP
trades systematically underperform because the credit is small and tail
risk eats the realized P&L. The calibrator correctly identified this and
slashed `pop` from 0.13 → 0.03.

### Known issues flagged but NOT addressed by this calibration

**Six variance-zero scorers** (gamma_pin, max_pain, oi_change,
option_rvol, pcr, sentiment) — 151 rows but constant scores. The score
functions are returning a sentinel for every trade. They now sit at 0.000
weight (correct given they carry no information), but the underlying
scorer bugs should be investigated separately. Each is potentially a
useful signal once the scorer returns real values.

### Resulting weight table (full, sorted desc)

```
vrp                  0.1401
term_structure       0.1287
vega_risk            0.1088
iv_velocity          0.0907
iv_edge              0.0885
spread               0.0881
iv_rank              0.0662
rr                   0.0491
momentum             0.0387
skew_align           0.0349
pop                  0.0322
trader_pref          0.0293
iv_mispricing        0.0265
liquidity            0.0198
ev                   0.0174
theta                0.0149
catalyst             0.0136
gamma_magnitude      0.0074
gex                  0.0025
max_pain             0.0025
em_realism           0.0000
gamma_theta          0.0000
pcr                  0.0000
oi_change            0.0000
sentiment            0.0000
option_rvol          0.0000
gamma_pin            0.0000
```

Total budget: 0.9999 (≈ 1.000, conserved).

### Validation plan (post-apply)

- **Watch first 2 weeks** of auto-logged trades. New trades will be scored
  with the new weights; track win-rate and avg P&L by quality_score
  quintile. If the top quintile still under-performs vs lower quintiles,
  the recalibration didn't land — revert to backup.
- **Re-snapshot weekly** with `scripts/calibrate_snapshot.sh` to track IC
  stability. If next snapshot shows ICs collapsing toward zero, the
  current readings may be over-fit to the 2026-04 / 2026-05 regime.
- **Defer IC-bucket and spread-bucket calibration** until their
  respective sample counts clear 30. ETA: Bear Call ~2-3 weeks, Iron
  Condor late June 2026.

### Revert procedure if regression

```bash
cp config.bak.20260511-111722.json config.json
# or restore the in-DB calibration_marker file from before this run:
cat paper_trades.db.calibration_marker.json
```

---
