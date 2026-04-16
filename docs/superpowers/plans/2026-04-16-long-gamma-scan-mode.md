# Long Gamma Scan Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Long Gamma" scan mode to the options screener that surfaces single-stock setups where volatility is compressed and a breakout is imminent, with per-row strategy recommendations (Straddle, Strangle, Bull Call Spread, Bear Put Spread).

**Architecture:** Add `bb_width_pct` (continuous squeeze signal) to `calculate_momentum_indicators`, add `filter_long_gamma` to `filters.py`, add a `long_gamma_score` scoring path and `recommended_strategy` column to `options_screener.py`, wire up the new mode in `_score_fetched_data`, and expose it in the dashboard + config.

**Tech Stack:** Python, pandas, numpy, Streamlit. No new dependencies.

---

## File Map

| File | Change |
|---|---|
| `src/data_fetching.py` | Add `bb_width_pct` to `calculate_momentum_indicators` return + both context dicts |
| `src/filters.py` | Add `filter_long_gamma` function |
| `src/options_screener.py` | Add `long_gamma_score` in `calculate_scores`, `recommended_strategy` in `enrich_and_score`, Long Gamma branch in `_score_fetched_data` |
| `src/dashboard.py` | Add "Long Gamma" to selectbox, priority columns for this mode |
| `config.json` | Add `long_gamma_weights` section |
| `tests/test_long_gamma.py` | New test file |

---

## Task 1: Add `bb_width_pct` to `calculate_momentum_indicators`

**Files:**
- Modify: `src/data_fetching.py`

### Step 1: Find the squeeze block in `calculate_momentum_indicators`

Open `src/data_fetching.py`. The function `calculate_momentum_indicators` is at line ~1163. The squeeze block near the end of the function looks like:

```python
        # Squeeze
        bb_std = close.rolling(window=20).std()
        bb_upper = sma_20 + (bb_std.iloc[-1] * 2)
        bb_lower = sma_20 - (bb_std.iloc[-1] * 2)
        kc_atr = atr.iloc[-1]
        kc_upper = sma_20 + (kc_atr * 1.5)
        kc_lower = sma_20 - (kc_atr * 1.5)
        is_squeezing = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        return ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14
```

- [ ] **Step 2: Replace that block with the version that adds `bb_width_pct`**

```python
        # Squeeze
        bb_std = close.rolling(window=20).std()
        bb_upper = sma_20 + (bb_std.iloc[-1] * 2)
        bb_lower = sma_20 - (bb_std.iloc[-1] * 2)
        kc_atr = atr.iloc[-1]
        kc_upper = sma_20 + (kc_atr * 1.5)
        kc_lower = sma_20 - (kc_atr * 1.5)
        is_squeezing = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        # BB Width percentile: (BB upper - BB lower) / SMA-20, ranked vs history
        # Low value = volatility compressed = primed to expand
        bb_width_pct = None
        try:
            bb_width_series = (close.rolling(window=20).std() * 4.0) / close.rolling(window=20).mean().replace(0, np.nan)
            bb_width_series = bb_width_series.dropna()
            if len(bb_width_series) >= 20:
                current_width = bb_width_series.iloc[-1]
                bb_width_pct = float((bb_width_series < current_width).mean())
        except Exception:
            pass

        return ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14, bb_width_pct
```

- [ ] **Step 3: Update the error return at the top of the function**

Find:
```python
        if hist.empty or len(hist) < 21:
            return None, None, None, None, None, None, False, None, None
```

Replace with:
```python
        if hist.empty or len(hist) < 21:
            return None, None, None, None, None, None, False, None, None, None
```

- [ ] **Step 4: Update the except return at the bottom of the function**

Find:
```python
    except Exception:
        return None, None, None, None, None, None, False, None, None
```

Replace with:
```python
    except Exception:
        return None, None, None, None, None, None, False, None, None, None
```

- [ ] **Step 5: Update the type hint on the function signature**

Find:
```python
def calculate_momentum_indicators(hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], bool, Optional[float], Optional[float]]:
```

Replace with:
```python
def calculate_momentum_indicators(hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], bool, Optional[float], Optional[float], Optional[float]]:
```

- [ ] **Step 6: Update the yfinance caller (line ~1706) to unpack `bb_width_pct`**

Find:
```python
    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14 = \
        calculate_momentum_indicators(hist)
```

Replace with:
```python
    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14, bb_width_pct = \
        calculate_momentum_indicators(hist)
```

- [ ] **Step 7: Update the yahooquery caller (line ~319) to unpack `bb_width_pct`**

Find (in `fetch_options_yahooquery`):
```python
    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14 = \
        calculate_momentum_indicators(hist)
```

Replace with:
```python
    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14, bb_width_pct = \
        calculate_momentum_indicators(hist)
```

- [ ] **Step 8: Add `bb_width_pct` to the yfinance context dict return**

In `fetch_options_yfinance`, find the context dict (at line ~1948). It starts with:
```python
        "context": {
            "hv": hv_30d,
```

Add `"bb_width_pct": bb_width_pct,` immediately after `"hv": hv_30d,`:
```python
        "context": {
            "hv": hv_30d,
            "bb_width_pct": bb_width_pct,
```

- [ ] **Step 9: Add `bb_width_pct` to the yahooquery context dict return**

In `fetch_options_yahooquery`, find the context dict (at line ~477). It starts with:
```python
            "context": {
                "hv": hv_30d,
```

Add `"bb_width_pct": bb_width_pct,` immediately after `"hv": hv_30d,`:
```python
            "context": {
                "hv": hv_30d,
                "bb_width_pct": bb_width_pct,
```

- [ ] **Step 10: Commit**

```bash
git add src/data_fetching.py
git commit -m "feat(data): add bb_width_pct squeeze intensity signal to calculate_momentum_indicators"
```

---

## Task 2: Add `filter_long_gamma` to `filters.py`

**Files:**
- Modify: `src/filters.py`
- Create: `tests/test_long_gamma.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_long_gamma.py`:

```python
"""Tests for Long Gamma scan mode components."""
import pytest
import pandas as pd
import numpy as np


class TestBbWidthPct:
    """Tests for the bb_width_pct signal in calculate_momentum_indicators."""

    def _make_hist(self, n=252, compressed=False):
        """Create a fake price history. compressed=True makes recent vol low."""
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        if compressed:
            # Last 20 days: very low daily moves
            prices[-20:] = prices[-21] + np.cumsum(np.random.normal(0, 0.001, 20))
        idx = pd.date_range(end="2026-04-16", periods=n, freq="B")
        return pd.DataFrame({"Close": prices, "High": prices * 1.005, "Low": prices * 0.995, "Volume": 1_000_000}, index=idx)

    def test_returns_ten_values(self):
        from src.data_fetching import calculate_momentum_indicators
        hist = self._make_hist()
        result = calculate_momentum_indicators(hist)
        assert len(result) == 10, f"Expected 10-tuple, got {len(result)}"

    def test_bb_width_pct_is_float_between_0_and_1(self):
        from src.data_fetching import calculate_momentum_indicators
        hist = self._make_hist()
        *_, bb_width_pct = calculate_momentum_indicators(hist)
        assert bb_width_pct is not None
        assert 0.0 <= bb_width_pct <= 1.0

    def test_compressed_history_has_lower_bb_width_pct(self):
        from src.data_fetching import calculate_momentum_indicators
        normal_hist = self._make_hist(compressed=False)
        compressed_hist = self._make_hist(compressed=True)
        *_, normal_pct = calculate_momentum_indicators(normal_hist)
        *_, compressed_pct = calculate_momentum_indicators(compressed_hist)
        assert compressed_pct is not None and normal_pct is not None
        assert compressed_pct < normal_pct, \
            f"Compressed ({compressed_pct:.3f}) should be < normal ({normal_pct:.3f})"

    def test_empty_history_returns_none_for_bb_width_pct(self):
        from src.data_fetching import calculate_momentum_indicators
        result = calculate_momentum_indicators(pd.DataFrame())
        assert result[-1] is None  # bb_width_pct should be None


class TestFilterLongGamma:
    """Tests for filter_long_gamma."""

    def _make_df(self):
        """Create a minimal options DataFrame for testing."""
        return pd.DataFrame({
            "symbol": ["AAPL"] * 5,
            "type": ["call"] * 5,
            "strike": [150.0, 155.0, 160.0, 165.0, 170.0],
            "iv_percentile_30": [0.10, 0.35, 0.45, 0.20, 0.60],  # 0.45 and 0.60 fail IV gate
            "T_years": [30/365, 45/365, 10/365, 50/365, 35/365],  # 10 days fails DTE gate
            "volume": [200, 500, 100, 50, 150],  # 50 fails volume gate
            "spread_pct": [0.10, 0.20, 0.30, 0.50, 0.15],  # 0.50 fails spread gate
        })

    def test_filters_high_iv_rank(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        # iv_percentile_30 >= 0.40 should be excluded (indices 2 and 4)
        assert all(result["iv_percentile_30"] < 0.40)

    def test_filters_short_dte(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        dte = result["T_years"] * 365.0
        assert all(dte >= 20)
        assert all(dte <= 60)

    def test_filters_low_volume(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        assert all(result["volume"] >= 100)

    def test_filters_wide_spread(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        assert all(result["spread_pct"] <= 0.40)

    def test_empty_df_returns_empty(self):
        from src.filters import filter_long_gamma
        result = filter_long_gamma(pd.DataFrame())
        assert result.empty
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:\Users\Oliver\desktop\options
python -m pytest tests/test_long_gamma.py -v 2>&1 | head -40
```

Expected: FAIL — `ImportError: cannot import name 'filter_long_gamma' from 'src.filters'`

- [ ] **Step 3: Add `filter_long_gamma` to `src/filters.py`**

Append to the end of `src/filters.py`:

```python

def filter_long_gamma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Long Gamma mode hard gates. Passes only options that are:
    - Cheap (IV rank < 40%)
    - Long-dated enough for a move to develop (DTE 20–60)
    - Liquid enough to buy (volume >= 100)
    - Tight spreads (spread_pct < 40%)
    Delta filtering is intentionally omitted: straddles and strangles span
    the full delta range by design.
    """
    if df.empty:
        return df

    # IV rank gate: only cheap options
    iv_pct_col = next(
        (c for c in ["iv_percentile_30", "iv_percentile"] if c in df.columns), None
    )
    if iv_pct_col:
        df = df[df[iv_pct_col].fillna(1.0) < 0.40].copy()

    # DTE gate: 20–60 calendar days
    if "T_years" in df.columns:
        dte = df["T_years"] * 365.0
        df = df[(dte >= 20) & (dte <= 60)].copy()

    # Minimum volume: illiquid options cannot be bought cleanly
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0).astype(float) >= 100].copy()

    # Spread filter (reuse existing threshold)
    if "spread_pct" in df.columns:
        df = df[df["spread_pct"].fillna(1.0) <= 0.40].copy()

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_long_gamma.py::TestFilterLongGamma -v
```

Expected: All 5 `TestFilterLongGamma` tests PASS

- [ ] **Step 5: Run `bb_width_pct` tests (from Task 1)**

```bash
python -m pytest tests/test_long_gamma.py::TestBbWidthPct -v
```

Expected: All 4 `TestBbWidthPct` tests PASS (these pass because Task 1 was done first)

- [ ] **Step 6: Commit**

```bash
git add src/filters.py tests/test_long_gamma.py
git commit -m "feat(filters): add filter_long_gamma for Long Gamma scan mode"
```

---

## Task 3: Add `long_gamma_score` to `calculate_scores`

**Files:**
- Modify: `src/options_screener.py`

- [ ] **Step 1: Locate the end of `calculate_scores`**

In `src/options_screener.py`, find the "Save components" block near line ~1598:

```python
    # Save components
    df["ev_score"] = ev_score
    df["spread_pct"] = df["spread_pct"].replace([float("inf"), -float("inf")], pd.NA)
    df["liquidity_score"], df["delta_quality"], df["iv_quality"] = liquidity, delta_quality, iv_quality
    df["spread_score"], df["theta_score"], df["momentum_score"] = spread_score, theta_score, momentum_score
    df["iv_rank_score"], df["catalyst_score"] = iv_rank_score, catalyst_score
    df["iv_advantage_score"] = iv_edge_score
    df["pop_score"] = pop_score
    df["rr_score"]  = rr_score
    df["trader_pref_score"] = trader_pref_score
    df["gamma_theta_score"] = gamma_theta_score
    df["pcr_score"]         = pcr_score
    df["gex_score"]         = gex_score
    df["oi_change_score"]   = oi_change_score
    df["sentiment_score_norm"] = sentiment_score_component

    return df
```

- [ ] **Step 2: Insert `long_gamma_score` block before `return df`**

Replace:

```python
    df["sentiment_score_norm"] = sentiment_score_component

    return df
```

With:

```python
    df["sentiment_score_norm"] = sentiment_score_component

    # Long Gamma mode: compute dedicated score with inverted IV rank
    # Done after all component scores are saved so we can reference them
    if mode == "Long Gamma":
        lg_w = config.get("long_gamma_weights", {
            "iv_cheap": 0.30, "squeeze": 0.25, "rvol": 0.20, "momentum": 0.15, "liquidity": 0.10
        })
        # Invert IV rank: low IV = good for buying
        _iv_cheap = (1.0 - pd.to_numeric(df.get("iv_rank_score", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)).clip(0, 1)
        # Invert BB width: tight bands = compressed = high score
        _bb_raw = pd.to_numeric(df.get("bb_width_pct", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)
        _squeeze = (1.0 - _bb_raw).clip(0, 1)
        # Relative volume: rank-normalized (high rvol = something brewing)
        _rvol_raw = pd.to_numeric(df.get("rvol", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
        _n = len(_rvol_raw)
        if _n > 1:
            _rvol_n = (_rvol_raw.rank(method="average", na_option="keep") - 1.0) / (_n - 1.0)
        else:
            _rvol_n = pd.Series(0.5, index=df.index)
        _mom = pd.to_numeric(df.get("momentum_score", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)
        _liq = pd.to_numeric(df.get("liquidity_score", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)

        _lg_sum = sum(lg_w.values()) or 1.0
        df["long_gamma_score"] = (
            lg_w.get("iv_cheap", 0.30) * _iv_cheap
            + lg_w.get("squeeze", 0.25) * _squeeze
            + lg_w.get("rvol", 0.20) * _rvol_n
            + lg_w.get("momentum", 0.15) * _mom
            + lg_w.get("liquidity", 0.10) * _liq
        ) / _lg_sum
        # Override quality_score so existing sort logic works unchanged
        df["quality_score"] = df["long_gamma_score"]
    else:
        df["long_gamma_score"] = pd.NA

    return df
```

- [ ] **Step 3: Verify syntax by importing the module**

```bash
python -c "from src.options_screener import calculate_scores; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/options_screener.py
git commit -m "feat(scoring): add long_gamma_score path to calculate_scores"
```

---

## Task 4: Add `recommended_strategy` column and bypass delta filter

**Files:**
- Modify: `src/options_screener.py`

- [ ] **Step 1: Locate the final filters block in `enrich_and_score`**

Find the block immediately after `df = calculate_scores(...)`:

```python
    df = calculate_scores(df, config, vix_regime_weights, trader_profile, mode, min_dte, max_dte, sector_etf=_sector_etf)

    # Final Filters
    if mode == "Premium Selling":
        d_min = fc.get("premium_selling_delta_min", 0.15)
        d_max = fc.get("premium_selling_delta_max", 0.40)
    else:
        d_min = fc.get("delta_min", 0.15)
        d_max = fc.get("delta_max", 0.35)
    df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    if mode != "Premium Selling":
        df = df[df["rr_ratio"] >= 0.25].copy()
```

- [ ] **Step 2: Replace that block with Long Gamma-aware version**

```python
    df = calculate_scores(df, config, vix_regime_weights, trader_profile, mode, min_dte, max_dte, sector_etf=_sector_etf)

    # Strategy recommendation for Long Gamma mode
    if mode == "Long Gamma":
        _is_sq = df.get("is_squeezing", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        _adx = pd.to_numeric(df.get("adx_14", pd.Series(20.0, index=df.index)), errors="coerce").fillna(20.0)
        _rsi = pd.to_numeric(df.get("rsi_14", pd.Series(50.0, index=df.index)), errors="coerce").fillna(50.0)
        _iv_pct = pd.to_numeric(df.get("iv_percentile_30", pd.Series(0.3, index=df.index)), errors="coerce").fillna(0.3)
        _rvol = pd.to_numeric(df.get("rvol", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
        _underlying = pd.to_numeric(df.get("underlying", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        _sma50 = pd.to_numeric(df.get("sma_50", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        _neutral_rsi = (_rsi >= 45) & (_rsi <= 55)
        _conditions = [
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct < 0.15),
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct >= 0.15),
            _is_sq & (_adx >= 20) & (_rsi > 55) & (_underlying > _sma50),
            _is_sq & (_adx >= 20) & (_rsi < 45) & (_underlying < _sma50),
            (~_is_sq) & (_rvol > 1.5) & (_adx > 25),
        ]
        _choices = ["Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", "Directional Debit Spread"]
        df["recommended_strategy"] = np.select(_conditions, _choices, default="Monitor")
    else:
        df["recommended_strategy"] = ""

    # Final Filters
    if mode == "Long Gamma":
        # No delta target for straddles/strangles — use full delta range
        pass
    elif mode == "Premium Selling":
        d_min = fc.get("premium_selling_delta_min", 0.15)
        d_max = fc.get("premium_selling_delta_max", 0.40)
        df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    else:
        d_min = fc.get("delta_min", 0.15)
        d_max = fc.get("delta_max", 0.35)
        df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    if mode not in ("Premium Selling", "Long Gamma"):
        df = df[df["rr_ratio"] >= 0.25].copy()
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "from src.options_screener import enrich_and_score; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Add tests for `recommended_strategy` logic**

Add to `tests/test_long_gamma.py`:

```python

class TestRecommendedStrategy:
    """Tests for the recommended_strategy column logic."""

    def _make_row(self, is_squeezing, adx, rsi, iv_pct, rvol, underlying, sma_50):
        return pd.DataFrame([{
            "is_squeezing": is_squeezing,
            "adx_14": adx,
            "rsi_14": rsi,
            "iv_percentile_30": iv_pct,
            "rvol": rvol,
            "underlying": underlying,
            "sma_50": sma_50,
        }])

    def _apply_strategy(self, df):
        """Replicate the np.select logic from enrich_and_score."""
        _is_sq = df.get("is_squeezing", pd.Series(False)).fillna(False).astype(bool)
        _adx = pd.to_numeric(df.get("adx_14", pd.Series(20.0)), errors="coerce").fillna(20.0)
        _rsi = pd.to_numeric(df.get("rsi_14", pd.Series(50.0)), errors="coerce").fillna(50.0)
        _iv_pct = pd.to_numeric(df.get("iv_percentile_30", pd.Series(0.3)), errors="coerce").fillna(0.3)
        _rvol = pd.to_numeric(df.get("rvol", pd.Series(1.0)), errors="coerce").fillna(1.0)
        _underlying = pd.to_numeric(df.get("underlying", pd.Series(0.0)), errors="coerce").fillna(0.0)
        _sma50 = pd.to_numeric(df.get("sma_50", pd.Series(0.0)), errors="coerce").fillna(0.0)
        _neutral_rsi = (_rsi >= 45) & (_rsi <= 55)
        _conditions = [
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct < 0.15),
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct >= 0.15),
            _is_sq & (_adx >= 20) & (_rsi > 55) & (_underlying > _sma50),
            _is_sq & (_adx >= 20) & (_rsi < 45) & (_underlying < _sma50),
            (~_is_sq) & (_rvol > 1.5) & (_adx > 25),
        ]
        _choices = ["Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", "Directional Debit Spread"]
        return np.select(_conditions, _choices, default="Monitor")

    def test_straddle_when_squeeze_no_direction_very_cheap(self):
        df = self._make_row(True, 15, 50, 0.10, 1.0, 150, 145)
        result = self._apply_strategy(df)
        assert result[0] == "Straddle"

    def test_strangle_when_squeeze_no_direction_moderately_cheap(self):
        df = self._make_row(True, 15, 50, 0.25, 1.0, 150, 145)
        result = self._apply_strategy(df)
        assert result[0] == "Strangle"

    def test_bull_call_spread_when_bullish_breakout(self):
        df = self._make_row(True, 25, 60, 0.25, 1.0, 155, 150)
        result = self._apply_strategy(df)
        assert result[0] == "Bull Call Spread"

    def test_bear_put_spread_when_bearish_breakout(self):
        df = self._make_row(True, 25, 40, 0.25, 1.0, 145, 150)
        result = self._apply_strategy(df)
        assert result[0] == "Bear Put Spread"

    def test_directional_debit_spread_when_momentum_no_squeeze(self):
        df = self._make_row(False, 28, 62, 0.25, 2.0, 155, 150)
        result = self._apply_strategy(df)
        assert result[0] == "Directional Debit Spread"

    def test_monitor_when_no_conditions_met(self):
        df = self._make_row(False, 15, 50, 0.35, 0.8, 150, 145)
        result = self._apply_strategy(df)
        assert result[0] == "Monitor"
```

- [ ] **Step 5: Run strategy tests**

```bash
python -m pytest tests/test_long_gamma.py::TestRecommendedStrategy -v
```

Expected: All 6 `TestRecommendedStrategy` tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/options_screener.py tests/test_long_gamma.py
git commit -m "feat(screener): add recommended_strategy column and bypass delta filter for Long Gamma mode"
```

---

## Task 5: Wire Long Gamma mode into `_score_fetched_data` and pass `bb_width_pct` to df

**Files:**
- Modify: `src/options_screener.py`

### Step 1: Find where `bb_width_pct` from context needs to reach the df

The context fields from `fetch_options_yfinance` are extracted in `_score_fetched_data` and passed to `enrich_and_score`. But `bb_width_pct` must reach the df BEFORE `enrich_and_score` for `calculate_scores` to use it. Looking at the flow, `enrich_and_score` receives the raw options chain df — context fields like `rvol`, `rsi_14`, `is_squeezing` etc. are already attached to the df during `fetch_options_yfinance` (they get added to all rows of the chain df there).

Check that `bb_width_pct` also gets written onto the options chain df rows in `fetch_options_yfinance`. Search for where `is_squeezing` is attached to df in that function.

- [ ] **Step 2: Confirm `is_squeezing` attachment pattern in `fetch_options_yfinance`**

```bash
grep -n "df\[.is_squeezing.\]\|df\[.rsi_14.\]\|df\[.adx_14.\]\|df\[.rvol.\]" src/data_fetching.py | head -20
```

This will show you the lines where these signals are written to the chain df. Note the exact pattern.

- [ ] **Step 3: Add `bb_width_pct` to the chain df in `fetch_options_yfinance`**

Find the block where these signals are assigned (search for `df["is_squeezing"] =`). Add the following line immediately after it:

```python
    df["bb_width_pct"] = bb_width_pct
```

- [ ] **Step 4: Also add to chain df in `fetch_options_yahooquery`**

Same pattern — find `df["is_squeezing"] =` in `fetch_options_yahooquery` and add:

```python
    df["bb_width_pct"] = bb_width_pct
```

- [ ] **Step 5: Wire Long Gamma mode into `_score_fetched_data`**

In `_score_fetched_data`, find the mode dispatch block:

```python
        if mode == "Credit Spreads":
            spreads = find_credit_spreads(df_scored)
            if not spreads.empty:
                result["credit_spreads"].append(spreads)
                result["success"] = True
        elif mode == "Iron Condor":
            condors = find_iron_condors(df_scored)
            if not condors.empty:
                result["iron_condors"] = condors
                result["success"] = True
        elif mode == "Premium Selling":
            puts = df_scored[df_scored["type"] == "put"].copy()
            if not puts.empty:
                result["picks"].append(puts)
                result["success"] = True
        else:
            result["picks"].append(df_scored)
            result["success"] = True
```

Replace with:

```python
        if mode == "Credit Spreads":
            spreads = find_credit_spreads(df_scored)
            if not spreads.empty:
                result["credit_spreads"].append(spreads)
                result["success"] = True
        elif mode == "Iron Condor":
            condors = find_iron_condors(df_scored)
            if not condors.empty:
                result["iron_condors"] = condors
                result["success"] = True
        elif mode == "Premium Selling":
            puts = df_scored[df_scored["type"] == "put"].copy()
            if not puts.empty:
                result["picks"].append(puts)
                result["success"] = True
        elif mode == "Long Gamma":
            from .filters import filter_long_gamma
            lg_filtered = filter_long_gamma(df_scored)
            if not lg_filtered.empty:
                result["picks"].append(lg_filtered)
                result["success"] = True
        else:
            result["picks"].append(df_scored)
            result["success"] = True
```

- [ ] **Step 6: Verify syntax**

```bash
python -c "from src.options_screener import run_scan; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/data_fetching.py src/options_screener.py
git commit -m "feat(screener): wire Long Gamma mode into _score_fetched_data and attach bb_width_pct to chain df"
```

---

## Task 6: Dashboard, config, and full test run

**Files:**
- Modify: `src/dashboard.py`
- Modify: `config.json`

- [ ] **Step 1: Add `long_gamma_weights` to `config.json`**

Open `config.json`. Find any top-level key (e.g., `"premium_selling_weights"`). Add a new sibling key after it:

```json
"long_gamma_weights": {
    "iv_cheap": 0.30,
    "squeeze": 0.25,
    "rvol": 0.20,
    "momentum": 0.15,
    "liquidity": 0.10
},
```

- [ ] **Step 2: Add "Long Gamma" to dashboard selectbox**

In `src/dashboard.py`, find:

```python
        scan_mode = st.selectbox(
            "Strategy",
            ["Discovery scan", "Single-stock", "Budget scan", "Premium Selling"],
            index=0
        )
```

Replace with:

```python
        scan_mode = st.selectbox(
            "Strategy",
            ["Discovery scan", "Single-stock", "Budget scan", "Premium Selling", "Long Gamma"],
            index=0
        )
```

- [ ] **Step 3: Add Long Gamma priority column display**

In `src/dashboard.py`, find the section where the results dataframe is displayed (search for `column_config`). Just before the dataframe display, find where column ordering is determined. Add a new block for Long Gamma mode.

Search for a pattern like:
```python
if scan_mode != "Single-stock":
```
or the area around the `st.dataframe` call. Add this block before the `st.dataframe` call:

```python
# Reorder columns to surface Long Gamma signals first
if scan_mode == "Long Gamma":
    lg_priority = [
        "recommended_strategy", "long_gamma_score", "symbol", "type", "strike",
        "expiration", "bb_width_pct", "is_squeezing", "iv_rank", "rvol",
        "rsi_14", "adx_14", "premium", "volume",
    ]
    existing_priority = [c for c in lg_priority if c in results_df.columns]
    remaining = [c for c in results_df.columns if c not in existing_priority]
    results_df = results_df[existing_priority + remaining]
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_long_gamma.py tests/test_math.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Smoke test — import the full pipeline**

```bash
python -c "
from src.options_screener import run_scan
from src.filters import filter_long_gamma
from src.data_fetching import calculate_momentum_indicators
import pandas as pd
import numpy as np

# Verify calculate_momentum_indicators returns 10 values
hist = pd.DataFrame({'Close': np.random.randn(252).cumsum() + 100,
                     'High': np.random.randn(252).cumsum() + 102,
                     'Low': np.random.randn(252).cumsum() + 98,
                     'Volume': [1000000]*252})
result = calculate_momentum_indicators(hist)
assert len(result) == 10, f'Expected 10, got {len(result)}'

# Verify filter_long_gamma runs without error
df = pd.DataFrame({'iv_percentile_30': [0.2, 0.5], 'T_years': [40/365, 5/365],
                   'volume': [200, 50], 'spread_pct': [0.1, 0.6]})
out = filter_long_gamma(df)
assert len(out) == 0  # both rows fail different gates

print('Smoke test PASSED')
"
```

Expected output: `Smoke test PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/dashboard.py config.json
git commit -m "feat(dashboard): add Long Gamma scan mode to UI and config"
```

- [ ] **Step 7: Final commit — all tests green**

```bash
python -m pytest tests/ -v
git add -A
git status  # verify nothing unexpected staged
```

---

## Self-Review Notes

### Spec Coverage Check
| Spec Section | Task |
|---|---|
| S1: `bb_width_pct` signal | Task 1 |
| S2: `long_gamma_score` weights | Task 3 |
| S3: `recommended_strategy` column | Task 4 |
| S4: `filter_long_gamma` | Task 2 |
| S5: Dashboard selectbox + column order | Task 6 |
| S6: `config.json` `long_gamma_weights` | Task 6 |
| `bb_width_pct` on chain df (needed by calculate_scores) | Task 5 |
| Long Gamma branch in `_score_fetched_data` | Task 5 |

All spec requirements covered.

### Key Constraints Addressed
- `calculate_momentum_indicators` 9-tuple → 10-tuple: both callers updated in Task 1 (steps 6–7)
- `bb_width_pct` must be on the chain df, not just in context: Task 5 step 3–4
- Delta filter bypass for Long Gamma: Task 4 step 2
- Safe config defaults (in case `long_gamma_weights` missing from config.json): Task 3 uses `.get()` with inline defaults
