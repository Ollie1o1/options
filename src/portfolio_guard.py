"""Portfolio / correlation guard for a set of screener picks (display-only).

The session's central lesson: short-premium sleeves that look like separate
trades are often the SAME bet (short the market / short vol) — 27 "positions"
that all blow up in one vol spike are one position. The per-contract view can't
see this; this guard can. Given the top picks, it aggregates net Greeks
(direction-aware by mode) and flags when the basket is really one concentrated
bet — directionally (net delta), in vol (net vega), or by underlying.

Pure + failure-safe; the screener renders it after the picks. Greeks on the rows
are per-share BS Greeks for a LONG contract; mode flips the sign for premium
selling. Reported in position units (×100 shares/contract, 1 contract each).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def compute_exposure(picks: List[Dict[str, Any]], mode: str = "Discovery",
                     contract_multiplier: int = 100) -> Dict[str, Any]:
    """Net + gross position Greeks across picks (1 contract each). Premium-selling
    mode flips the sign (you are SHORT the contracts)."""
    sign = -1.0 if mode == "Premium Selling" else 1.0
    net = {"delta": 0.0, "vega": 0.0, "theta": 0.0, "gamma": 0.0}
    gross = {"delta": 0.0, "vega": 0.0}
    by_symbol: Dict[str, int] = {}
    n = 0
    for r in picks:
        d = _f(r.get("delta"))
        if d is None:
            continue
        n += 1
        m = contract_multiplier
        net["delta"] += sign * d * m
        net["vega"] += sign * (_f(r.get("vega")) or 0.0) * m
        net["theta"] += sign * (_f(r.get("theta")) or 0.0) * m
        net["gamma"] += sign * (_f(r.get("gamma")) or 0.0) * m
        gross["delta"] += abs(d * m)
        gross["vega"] += abs((_f(r.get("vega")) or 0.0) * m)
        sym = str(r.get("symbol") or "?").upper()
        by_symbol[sym] = by_symbol.get(sym, 0) + 1
    return {"n": n, "mode": mode, "net": net, "gross": gross, "by_symbol": by_symbol}


def guard_warnings(exposure: Dict[str, Any], conc: float = 0.70,
                   symbol_conc: float = 0.60) -> List[str]:
    """Concentration flags. ``conc`` = share of gross exposure that is net (1.0 =
    all one direction). ``symbol_conc`` = share of picks on a single underlying."""
    out: List[str] = []
    n = exposure.get("n", 0)
    if n < 2:
        return out
    net, gross = exposure["net"], exposure["gross"]
    # Directional concentration: net delta dominates gross -> one directional bet
    if gross["delta"] > 0 and abs(net["delta"]) / gross["delta"] >= conc:
        side = "long" if net["delta"] > 0 else "short"
        out.append(f"Directional concentration: net delta {net['delta']:+,.0f} "
                   f"({abs(net['delta'])/gross['delta']:.0%} of gross) — these picks are "
                   f"one {side} bet on the underlying, not {n} independent trades")
    # Vol concentration: net vega dominates -> all co-move in a vol shock
    if gross["vega"] > 0 and abs(net["vega"]) / gross["vega"] >= conc:
        side = "long" if net["vega"] > 0 else "short"
        out.append(f"Vol concentration: net vega {net['vega']:+,.0f} "
                   f"({abs(net['vega'])/gross['vega']:.0%} of gross) — concentrated {side}-vol; "
                   f"these will co-move in a volatility spike (the same bet many times)")
    # Underlying concentration
    by = exposure.get("by_symbol", {})
    if by:
        top_sym, top_n = max(by.items(), key=lambda kv: kv[1])
        if top_n / n >= symbol_conc and top_n >= 2:
            out.append(f"Underlying concentration: {top_n}/{n} picks are {top_sym} — "
                       f"single-name risk")
    return out


def portfolio_guard(picks: List[Dict[str, Any]], mode: str = "Discovery") -> Dict[str, Any]:
    """Exposure + warnings for a basket of picks."""
    exp = compute_exposure(picks, mode=mode)
    exp["warnings"] = guard_warnings(exp)
    return exp


def format_guard_lines(picks: List[Dict[str, Any]], mode: str = "Discovery",
                       corr_pairs: Optional[List] = None) -> List[str]:
    """Plain-text panel lines (display-only). Empty list if < 2 picks or no data.

    ``corr_pairs`` is an optional list of ``(t1, t2, corr)`` highly-correlated
    pairs; when present they render in this same panel so concentration risk
    (Greeks + price correlation) lives in one place.
    """
    g = portfolio_guard(picks, mode=mode)
    if g["n"] < 2:
        return []
    net = g["net"]
    lines = [f"Portfolio guard ({g['n']} picks, {mode}):  "
             f"net Δ {net['delta']:+,.0f}  |  net vega {net['vega']:+,.0f}  |  "
             f"net theta {net['theta']:+,.0f}"]
    if g["warnings"]:
        for w in g["warnings"]:
            lines.append(f"  ⚠ {w}")
    else:
        lines.append("  ✓ reasonably diversified across direction / vol / underlyings")
    if corr_pairs:
        for t1, t2, c in sorted(corr_pairs, key=lambda p: -p[2])[:5]:
            lines.append(f"  ⚠ {t1}/{t2} corr {c:.2f} — same bet twice; "
                         f"don't size as independent")
        if len(corr_pairs) > 5:
            lines.append(f"  … and {len(corr_pairs) - 5} more pairs ≥0.80")
    return lines
