# Long Gamma Scan Mode — Design Spec
**Date:** 2026-04-16
**Scope:** 4 files — `data_fetching.py`, `options_screener.py`, `filters.py`, `dashboard.py`
**Goal:** Add a "Long Gamma" scan mode that surfaces single-stock options setups where volatility is compressed and a large move is imminent — with strategy recommendations (Straddle, Strangle, Bull Call Spread, Bear Put Spread).

---

## Background

The existing screener is a premium-selling tool: it scores options where IV is high relative to history (IV rank high = good). Long gamma plays require the opposite — cheap options (IV rank low) into a volatility expansion or directional breakout. These two scoring philosophies are mutually exclusive, making a dedicated scan mode the right architectural choice.

All raw signals needed already exist in the codebase. This feature is primarily a new scoring path and strategy recommendation layer on top of existing data.

---

## Section 1 — Signal Layer (`data_fetching.py`)

### New signal: `bb_width_pct`
- **What it is:** Bollinger Band width ((BB upper − BB lower) / SMA-20) as a percentile vs its own 252-day rolling history. Range 0–1. Low value = compressed volatility.
- **Why:** The existing `is_squeezing` bool (BB inside Keltner Channel) is binary. `bb_width_pct` is continuous — a value of 0.05 means "tighter than 95% of the past year," which is far more actionable than a yes/no flag.
- **Where computed:** Inside `calculate_momentum_indicators()`, using the already-available `bb_std`, `sma_20`, and close history. Appended as the 10th value in the return tuple (the function currently returns a 9-tuple). All callers must be updated to unpack the new value.
- **Where surfaced:** Added to the `context` dict in `fetch_options_yfinance` return value, same pattern as all other context fields.

### No other changes to `data_fetching.py`
All other needed signals already exist and are returned:
- `is_squeezing` (bool) — BB inside Keltner Channel
- `rsi_14` — RSI 14
- `atr_trend` — ATR 14 vs 20-day MA ratio minus 1
- `adx_14` — ADX-14 trend strength
- `rvol` — relative volume (current / 30-day avg)
- `iv_rank_30`, `iv_percentile_30` — IV rank vs history
- `sma_20`, `sma_50` — moving averages

---

## Section 2 — Scoring (`options_screener.py → calculate_scores`)

### New `long_gamma_score` branch
Activated when `mode == "Long Gamma"`. Weights sourced from `config.json` under `"long_gamma_weights"`.

| Component | Signal | Direction | Default Weight |
|---|---|---|---|
| `iv_cheap` | `1 - iv_rank_score` | Low IV rank = good (inverted from all other modes) | 0.30 |
| `squeeze` | `1 - bb_width_pct` | Tight BB = compressed = primed to expand | 0.25 |
| `rvol` | relative volume score | Unusual volume = catalyst brewing | 0.20 |
| `momentum` | RSI + ADX blend | Confirms directional energy building | 0.15 |
| `liquidity` | existing liquidity score | Still need tradeable options | 0.10 |

### Momentum score computation (for this mode)
`momentum_score = 0.5 * rsi_momentum + 0.5 * adx_score`
- `rsi_momentum`: distance from 50 (either direction), normalized 0–1. RSI of 70 or 30 both score high — we want movement, not direction.
- `adx_score`: ADX clipped to [0, 40], normalized. ADX > 25 = trending = high score.

### `long_gamma_score` as sort key
Replaces `quality_score` as the primary sort column when mode == "Long Gamma". `quality_score` is still computed alongside it so paper trade logging stays consistent.

---

## Section 3 — Strategy Recommendation (`options_screener.py → enrich_and_score`)

### New `recommended_strategy` column
Derived from signals already on the dataframe at enrichment time:

| Condition | Strategy |
|---|---|
| `is_squeezing` + ADX < 20 + RSI 45–55 + iv_rank < 0.15 | **Straddle** (ATM, max gamma, very cheap IV) |
| `is_squeezing` + ADX < 20 + RSI 45–55 + iv_rank 0.15–0.40 | **Strangle** (OTM wings, reduce cost) |
| `is_squeezing` + ADX ≥ 20 + RSI > 55 + price > SMA50 | **Bull Call Spread** |
| `is_squeezing` + ADX ≥ 20 + RSI < 45 + price < SMA50 | **Bear Put Spread** |
| Not squeezing + rvol > 1.5 + ADX > 25 | **Directional Debit Spread** (momentum already moving) |
| Otherwise | **Monitor** (setup not ripe) |

### Strategy logic placement
Computed as a vectorized column in `enrich_and_score` after Greeks and scores are attached. Uses `np.select` with condition arrays for efficiency. The column is available on the results dataframe — paper trade integration is out of scope for this phase (no changes to `paper_manager.py`).

---

## Section 4 — Filters (`filters.py`)

### New `filter_long_gamma` function
Hard gates (not score components — stocks failing these are excluded entirely):

| Filter | Value | Rationale |
|---|---|---|
| IV rank | < 40% | Buying expensive options destroys edge |
| DTE | 20–60 days | < 20 = insufficient time; > 60 = too much carry cost while waiting |
| Volume | ≥ 100 | Illiquid options cannot be bought/sold cleanly |
| Spread pct | < 40% | Reuse existing spread filter |
| Delta target | **Removed** | Straddles/strangles are ATM by definition; delta filtering is irrelevant |

---

## Section 5 — Dashboard (`dashboard.py`)

### Selectbox change
Add `"Long Gamma"` to the `scan_mode` selectbox options list. One line change.

### Priority columns when mode == "Long Gamma"
Results table reordered to surface relevant signals first:
`recommended_strategy`, `long_gamma_score`, `bb_width_pct`, `is_squeezing`, `iv_rank`, `rvol`, `rsi_14`, `adx_14`

Existing columns remain — just repositioned.

---

## Section 6 — Configuration (`config.json`)

New section added:
```json
"long_gamma_weights": {
  "iv_cheap": 0.30,
  "squeeze": 0.25,
  "rvol": 0.20,
  "momentum": 0.15,
  "liquidity": 0.10
}
```

---

## What Does NOT Change

- Paper trading (`paper_manager.py`) — no changes needed; `recommended_strategy` is just a new field
- Stress testing (`stress_test.py`) — no changes
- Existing scan modes — no behavior changes
- `simulation.py`, `utils.py`, `scoring.py` — untouched

---

## Key Constraints

- `calculate_momentum_indicators` currently returns a 9-tuple. Adding `bb_width_pct` requires updating all callers of this function (there may be one or two). Implementation plan must identify these.
- `options_screener.py` is 4001 lines. Edits must be surgical — find the exact `calculate_scores` branch and `enrich_and_score` strategy section, no wholesale rewrites.
- All new config keys must have safe defaults (in case `config.json` is missing the section) to avoid breaking existing scans.
