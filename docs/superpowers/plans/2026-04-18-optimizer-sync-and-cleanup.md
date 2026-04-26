# Optimizer Sync & Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sync backtest_optimizer.py with config.json's 27 weight keys so Week 4 optimization covers all scoring components. Remove dead code. Fix stale cache file.

**Architecture:** The optimizer currently operates on 21 weight keys but config.json has 27. The 6 missing keys (`iv_velocity`, `iv_mispricing`, `term_structure`, `gamma_magnitude`, `vega_risk`, `spread`) collectively account for 21% of the composite score. The optimizer also has hardcoded `CURRENT_WEIGHTS` that diverge from config.json — but it already has `_load_current_from_config()` that reads from config.json at runtime, so the hardcoded dict is just a fallback. The fix: add the 6 keys to `WEIGHT_KEYS`, add score computations where possible, set neutral for the rest, and update the hardcoded fallback weights.

**Tech Stack:** Python, NumPy, SciPy

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/backtest_optimizer.py` | Modify | Add 6 missing weight keys, expand score array, update fallback weights |
| `src/calc_expected_move.py` | Delete | Dead code — 19 lines re-exporting from options_screener, never imported |
| `tests/test_data_fetching.py` | Modify | Add test for clear_chain_cache clearing all caches |

---

### Task 1: Sync WEIGHT_KEYS and CURRENT_WEIGHTS with config.json

**Files:**
- Modify: `src/backtest_optimizer.py:62-86`

- [ ] **Step 1: Update WEIGHT_KEYS to include all 27 keys**

Replace the WEIGHT_KEYS list (line 62-66) with:

```python
WEIGHT_KEYS: List[str] = [
    "pop", "em_realism", "iv_mispricing", "rr", "momentum", "iv_rank",
    "liquidity", "catalyst", "theta", "ev", "trader_pref", "iv_edge",
    "skew_align", "gamma_theta", "pcr", "gex", "oi_change", "sentiment",
    "option_rvol", "vrp", "gamma_pin", "max_pain", "iv_velocity",
    "gamma_magnitude", "vega_risk", "term_structure", "spread",
]
```

Order matches config.json for readability.

- [ ] **Step 2: Update CURRENT_WEIGHTS fallback to match config.json**

Replace CURRENT_WEIGHTS (line 69-76) with actual values from config.json:

```python
CURRENT_WEIGHTS: Dict[str, float] = {
    "pop": 0.13, "em_realism": 0.0, "iv_mispricing": 0.05, "rr": 0.10,
    "momentum": 0.10, "iv_rank": 0.08, "liquidity": 0.08, "catalyst": 0.0,
    "theta": 0.06, "ev": 0.07, "trader_pref": 0.0, "iv_edge": 0.08,
    "skew_align": 0.02, "gamma_theta": 0.0, "pcr": 0.0, "gex": 0.01,
    "oi_change": 0.0, "sentiment": 0.0, "option_rvol": 0.0, "vrp": 0.05,
    "gamma_pin": 0.0, "max_pain": 0.01, "iv_velocity": 0.05,
    "gamma_magnitude": 0.03, "vega_risk": 0.03, "term_structure": 0.04,
    "spread": 0.01,
}
```

- [ ] **Step 3: Update RESEARCH_WEIGHTS to include all 27 keys**

Replace RESEARCH_WEIGHTS (line 79-86). Set the 6 new keys to sensible research-backed starting values:

```python
RESEARCH_WEIGHTS: Dict[str, float] = {
    "pop": 0.22, "em_realism": 0.04, "iv_mispricing": 0.04, "rr": 0.11,
    "momentum": 0.05, "iv_rank": 0.18, "liquidity": 0.09, "catalyst": 0.01,
    "theta": 0.07, "ev": 0.05, "trader_pref": 0.02, "iv_edge": 0.08,
    "skew_align": 0.03, "gamma_theta": 0.01, "pcr": 0.01, "gex": 0.01,
    "oi_change": 0.01, "sentiment": 0.01, "option_rvol": 0.02, "vrp": 0.06,
    "gamma_pin": 0.01, "max_pain": 0.01, "iv_velocity": 0.04,
    "gamma_magnitude": 0.02, "vega_risk": 0.02, "term_structure": 0.03,
    "spread": 0.01,
}
```

- [ ] **Step 4: Commit**

```bash
git add src/backtest_optimizer.py
git commit -m "fix(optimizer): sync WEIGHT_KEYS with config.json (21 → 27 keys)"
```

---

### Task 2: Expand compute_component_scores to 27 elements

**Files:**
- Modify: `src/backtest_optimizer.py:179-258`

The function currently returns a 21-element array. Expand to 27 by adding the 6 new score computations. Of the 6 new keys:

- `iv_velocity`: can approximate from HV trend (HV ratio derivative)
- `iv_mispricing`: set neutral (requires live IV surface, not available in backtest)
- `term_structure`: can approximate from HV slope across windows
- `gamma_magnitude`: compute from BS gamma
- `vega_risk`: compute from BS vega relative to premium
- `spread`: set neutral (requires live bid-ask, not available in backtest)

- [ ] **Step 1: Add score computations and expand the array**

After the existing score computations (around line 232), add:

```python
    # -- New factors (added to match config.json composite_weights) --------

    # IV velocity: rate of change in vol — proxy via HV ratio deviation from 1.0
    iv_velocity_score = float(np.clip(abs(hv_ratio - 1.0) * 2.0, 0, 1))

    # IV mispricing: requires live surface data, neutral in backtest
    iv_mispricing_score = _NEUTRAL

    # Term structure: proxy — longer DTE with elevated vol = steeper term structure
    term_structure_score = float(np.clip((hv_ratio - 0.9) * T * 8.0, 0, 1))

    # Gamma magnitude: BS gamma normalized by underlying price
    import math as _m
    if T > 0 and sigma > 0 and S > 0:
        sq = sigma * _m.sqrt(T)
        d1 = (_m.log(S / K) + (r + 0.5 * sigma**2) * T) / sq
        gamma = _m.exp(-0.5 * d1**2) / (_m.sqrt(2 * _m.pi) * S * sq)
        gamma_magnitude_score = float(np.clip(gamma * S * 100, 0, 1))
    else:
        gamma_magnitude_score = _NEUTRAL

    # Vega risk: vega as fraction of premium (high vega/premium = risky for sellers)
    if T > 0 and sigma > 0 and S > 0:
        vega = S * _m.exp(-0.5 * d1**2) / _m.sqrt(2 * _m.pi) * _m.sqrt(T)
        vega_risk_score = float(np.clip(1.0 - vega / max(prem * 10, 0.01), 0, 1))
    else:
        vega_risk_score = _NEUTRAL

    # Spread: requires live bid-ask, neutral in backtest
    spread_score = _NEUTRAL
```

Then replace the scores array (line 235-257) to include all 27:

```python
    scores = np.array([
        pop_score,              # pop
        em_realism_score,       # em_realism
        iv_mispricing_score,    # iv_mispricing
        rr_score,               # rr
        momentum_score,         # momentum
        iv_rank_score,          # iv_rank
        liquidity_score,        # liquidity
        _NEUTRAL,               # catalyst
        theta_score,            # theta
        ev_score,               # ev
        _NEUTRAL,               # trader_pref
        iv_edge_score,          # iv_edge
        skew_align_score,       # skew_align
        _NEUTRAL,               # gamma_theta
        _NEUTRAL,               # pcr
        _NEUTRAL,               # gex
        _NEUTRAL,               # oi_change
        _NEUTRAL,               # sentiment
        _NEUTRAL,               # option_rvol
        vrp_score,              # vrp
        _NEUTRAL,               # gamma_pin
        _NEUTRAL,               # max_pain
        iv_velocity_score,      # iv_velocity
        gamma_magnitude_score,  # gamma_magnitude
        vega_risk_score,        # vega_risk
        term_structure_score,   # term_structure
        spread_score,           # spread
    ], dtype=np.float64)
```

- [ ] **Step 2: Update the docstring**

Change `"""Return a length-21 array` to `"""Return a length-27 array`.

- [ ] **Step 3: Update BacktestResult docstring**

Change the comment `# (N, 21)` at line 416 to `# (N, 27)`.

- [ ] **Step 4: Verify optimizer runs**

Run: `./venv/bin/python -c "from src.backtest_optimizer import compute_component_scores, WEIGHT_KEYS; import numpy as np; s = compute_component_scores(500, 480, 45/365, 0.05, 0.25, 0.6, 1.1, 0.45, 0.7); print(f'Keys: {len(WEIGHT_KEYS)}, Scores: {len(s)}, All finite: {np.all(np.isfinite(s))}')"`

Expected: `Keys: 27, Scores: 27, All finite: True`

- [ ] **Step 5: Commit**

```bash
git add src/backtest_optimizer.py
git commit -m "feat(optimizer): expand score computation to 27 components"
```

---

### Task 3: Delete dead code file

**Files:**
- Delete: `src/calc_expected_move.py`

- [ ] **Step 1: Verify nothing imports it**

Run: `grep -rn "calc_expected_move" src/ tests/ *.py`
Expected: only hits in calc_expected_move.py itself

- [ ] **Step 2: Delete the file**

```bash
rm src/calc_expected_move.py
```

- [ ] **Step 3: Commit**

```bash
git add -A src/calc_expected_move.py
git commit -m "chore: remove dead calc_expected_move.py (unused re-export shim)"
```

---

### Task 4: Add test for complete cache clearing

**Files:**
- Modify: `tests/test_data_fetching.py`

- [ ] **Step 1: Add test**

```python
def test_clear_chain_cache_clears_all_caches():
    """clear_chain_cache should clear sentiment and seasonality caches too."""
    from src.data_fetching import (
        clear_chain_cache, _CHAIN_CACHE, _NEWS_CACHE, _INFO_CACHE,
        _SENTIMENT_CACHE, _SEASONALITY_CACHE,
    )
    _CHAIN_CACHE["X"] = {}
    _NEWS_CACHE["X"] = []
    _INFO_CACHE["X"] = {}
    _SENTIMENT_CACHE["X:sentiment"] = 0.5
    _SEASONALITY_CACHE["X:seasonality"] = 0.6
    clear_chain_cache()
    assert len(_CHAIN_CACHE) == 0
    assert len(_NEWS_CACHE) == 0
    assert len(_INFO_CACHE) == 0
    assert len(_SENTIMENT_CACHE) == 0
    assert len(_SEASONALITY_CACHE) == 0
```

- [ ] **Step 2: Run all tests**

Run: `./venv/bin/python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_data_fetching.py
git commit -m "test: verify clear_chain_cache clears all in-memory caches"
```

---

### Task 5: Run full verification

- [ ] **Step 1: Run full test suite**
- [ ] **Step 2: Verify optimizer smoke test**
- [ ] **Step 3: Verify screener import**
