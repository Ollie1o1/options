"""Reliability backtest — decides how much each signal is allowed to count.

A signal only earns weight in the verdict if its sign has historically agreed
with forward returns. For each price-derived signal we recompute its value at
every day across a liquid basket, pair it with the forward N-day return, and
measure two things:

  hit_rate — fraction of non-flat days where sign(signal) == sign(forward)
  ic       — rank (Spearman) correlation of signal value vs forward return

A signal that beats a small null margin earns weight proportional to its IC and
is tagged `reliable` (or `ok n=…` if the sample is thin). Anything that doesn't
is given **zero weight** and tagged `weak — context only`, so the verdict never
leans on noise. News/analyst signals are not derivable from a close series, so
they default to context-only unless wired to a historical source later.

Pure math (`signal_skill`, `compute_reliability`) is unit-tested offline; the
fetch + cache wrapper (`load_or_compute_reliability`) hits the network.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

# Signals that can be reconstructed historically from a close series alone.
_PRICE_SIGNALS = ("trend", "momentum", "bounce", "rsi", "support")

# Signals shown but not backtestable from closes → context-only by default.
_CONTEXT_ONLY = ("news", "analyst")

_DEFAULT_BASKET = (
    "SPY", "QQQ", "IWM", "SMH", "NVDA", "AMD",
    "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
)

_CACHE_PATH = "data/intel_reliability.json"
_CACHE_TTL_S = 7 * 24 * 3600  # one week
_FORWARD_DAYS = 10
_MIN_N = 200          # below this, a signal is "thin" at best
_IC_NULL = 0.02       # IC must clear this to earn weight


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, dependency-free. None if degenerate."""
    n = len(xs)
    if n < 3:
        return None

    def _rank(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def signal_skill(signal_values: Sequence[float],
                 forward_returns: Sequence[float]) -> Dict[str, Any]:
    """Hit-rate + IC of a signal series against paired forward returns."""
    pairs = [(s, f) for s, f in zip(signal_values, forward_returns)
             if s is not None and f is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "hit_rate": None, "ic": None}
    hits = total = 0
    for s, f in pairs:
        if s == 0:
            continue  # flat signal makes no directional claim
        total += 1
        if (s > 0) == (f > 0):
            hits += 1
    hit_rate = (hits / total) if total else None
    ic = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
    return {"n": n, "directional_n": total, "hit_rate": hit_rate, "ic": ic}


def _signal_value_at(closes: List[float], t: int, name: str) -> Optional[float]:
    """Reconstruct one price-signal's value using data up to and including day t."""
    from src.intel import signals as S
    from src.levels import bounce_stats

    hist = closes[: t + 1]
    if len(hist) < 30:
        return None
    price = hist[-1]
    ma50 = sum(hist[-50:]) / min(len(hist), 50)
    ma200 = sum(hist[-200:]) / 200 if len(hist) >= 200 else None
    ret_5d = price / hist[-6] - 1.0 if len(hist) >= 6 else None
    ret_20d = price / hist[-21] - 1.0 if len(hist) >= 21 else None

    if name == "trend":
        return S.trend_signal(price, ma50, ma200).value
    if name == "momentum":
        return S.momentum_signal(ret_5d, ret_20d).value
    if name == "rsi":
        from src.levels import rsi as _rsi
        return S.rsi_signal(_rsi(hist)).value
    if name == "support":
        from src.levels import support_resistance_levels
        lv = support_resistance_levels(hist)
        sup = lv["supports"][0]["pct"] if lv["supports"] else None
        return S.support_proximity_signal(sup).value
    if name == "bounce":
        # Bounce only makes a claim after a real drop; otherwise flat.
        if ret_5d is None or ret_5d > -0.02:
            return 0.0
        b = bounce_stats(hist, horizons=(_FORWARD_DAYS,))
        st = b["by_horizon"].get(_FORWARD_DAYS, {})
        return S.bounce_signal(st.get("bounce_rate"), st.get("n")).value
    return None


def compute_reliability(price_histories: Dict[str, Sequence[float]],
                        forward_days: int = _FORWARD_DAYS,
                        step: int = 3) -> Dict[str, Any]:
    """Backtest every price signal across the basket; assign weights + tags.

    `step` subsamples days (every 3rd) to keep the O(days × signals) reconstruction
    affordable without materially changing the pooled statistics.
    """
    pooled: Dict[str, Dict[str, List[float]]] = {
        nm: {"sig": [], "fwd": []} for nm in _PRICE_SIGNALS
    }
    for _sym, closes in price_histories.items():
        vals = [float(x) for x in closes if x is not None]
        if len(vals) < 250:
            continue
        last_t = len(vals) - forward_days - 1
        for t in range(30, last_t, step):
            fwd = vals[t + forward_days] / vals[t] - 1.0
            for nm in _PRICE_SIGNALS:
                sv = _signal_value_at(vals, t, nm)
                if sv is not None:
                    pooled[nm]["sig"].append(sv)
                    pooled[nm]["fwd"].append(fwd)

    out: Dict[str, Any] = {"_meta": {"forward_days": forward_days,
                                     "computed_at": time.time()}}
    for nm in _PRICE_SIGNALS:
        skill = signal_skill(pooled[nm]["sig"], pooled[nm]["fwd"])
        out[nm] = _grade(skill)
    for nm in _CONTEXT_ONLY:
        out[nm] = {"weight": 0.0, "tag": "weak — context only",
                   "ic": None, "hit_rate": None, "n": 0}
    return out


def _grade(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Turn raw skill into a weight + tag."""
    ic = skill.get("ic")
    n = skill.get("directional_n") or 0
    base = {"ic": round(ic, 4) if ic is not None else None,
            "hit_rate": round(skill["hit_rate"], 4) if skill.get("hit_rate") is not None else None,
            "n": skill.get("n", 0)}
    if ic is None or n < 30 or ic <= _IC_NULL:
        base.update({"weight": 0.0, "tag": "weak — context only"})
    elif n >= _MIN_N and ic >= 0.05:
        base.update({"weight": round(min(1.0, ic / 0.10), 3), "tag": "reliable"})
    else:
        base.update({"weight": round(min(0.5, ic / 0.10), 3), "tag": f"ok n={n}"})
    return base


# ── Fetch + cache wrapper ──────────────────────────────────────────────────────

def fetch_basket_histories(symbols: Sequence[str] = _DEFAULT_BASKET,
                           period: str = "4y") -> Dict[str, List[float]]:
    """Fetch ~4y closes for the basket via the resilient regime fetcher."""
    from src.regime_dashboard import _safe_hist
    out: Dict[str, List[float]] = {}
    for sym in symbols:
        s = _safe_hist(sym, period)
        if s is not None and len(s) > 0:
            out[sym] = s.tolist()
    return out


def load_or_compute_reliability(
    cache_path: str = _CACHE_PATH,
    force: bool = False,
    fetch: Optional[Callable[[], Dict[str, List[float]]]] = None,
) -> Dict[str, Any]:
    """Return reliability map, using the on-disk cache when fresh.

    Falls back to neutral (all context-only, zero weight) if data is
    unavailable — the briefing still runs, it just won't claim an edge.
    """
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            age = time.time() - cached.get("_meta", {}).get("computed_at", 0)
            if age < _CACHE_TTL_S:
                return cached
        except (OSError, ValueError):
            pass

    histories = (fetch or fetch_basket_histories)()
    if not histories:
        return _neutral_reliability()
    rel = compute_reliability(histories)
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(rel, f, indent=2)
    except OSError:
        pass
    return rel


def _neutral_reliability() -> Dict[str, Any]:
    out = {"_meta": {"forward_days": _FORWARD_DAYS, "computed_at": 0,
                     "note": "no data — neutral fallback"}}
    for nm in _PRICE_SIGNALS + _CONTEXT_ONLY:
        out[nm] = {"weight": 0.0, "tag": "weak — context only",
                   "ic": None, "hit_rate": None, "n": 0}
    return out
