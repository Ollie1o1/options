"""Per-segment LONG / SHORT / STAND DOWN verdict for the live screener (P2.6).

The recommender (`dolt_research.recommend`) is the source of truth, but it runs
three real-marks backtests over DoltHub per call — far too slow / rate-limited to
run inside the day-to-day screener. So we cache its verdict per SEGMENT and look
it up by symbol at display time. The live tool shows what the backtest learned
without hitting the network.

DISPLAY-ONLY. Like every other signal overlay, this does not size, gate, or
place anything until the real-money gate fires. It reflects the per-trade
direction of edge; it does NOT claim portfolio capacity (the index put spread
has an edge per trade but ~0.3% CAGR sized — see DOLT_NEXT_STEPS P1.3), so the
label carries a short caveat where capacity is thin.

Refresh the cache (slow, hits DoltHub):
    python -m src.dolt_verdict --build
Inspect:
    python -m src.dolt_verdict --show SPY
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.dolt_research import SEGMENTS

CACHE_PATH = "data/dolt_verdicts.json"

# Built-in defaults = the validated findings (DOLT_RESEARCH / DOLT_NEXT_STEPS),
# used when the cache file is absent so the live display works offline. Each:
# best strategy, human label, and a one-line caveat.
DEFAULT_VERDICTS: Dict[str, Dict[str, Any]] = {
    "index": {"best": "put_spread",
              "label": "SHORT — sell put credit spreads (defined risk)",
              "caveat": "edge real per-trade (PF~4) but tiny portfolio capacity (~0.3% CAGR sized)"},
    "semi":  {"best": "short_put",
              "label": "SHORT — sell naked puts",
              "caveat": "PF~1.17; spread version loses here; assignment risk (naked)"},
    "tech":  {"best": None,
              "label": "STAND DOWN — no edge on real marks",
              "caveat": "AAPL/MSFT/GOOG: nothing clears PF threshold"},
}


def _segment_of(symbol: str) -> Optional[str]:
    s = symbol.upper()
    for seg, syms in SEGMENTS.items():
        if s in syms:
            return seg
    return None


def _load_cache(cache_path: str = CACHE_PATH) -> Dict[str, Any]:
    try:
        with open(cache_path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def verdict_for(symbol: str, cache_path: str = CACHE_PATH) -> Optional[Dict[str, Any]]:
    """Return {segment, best, label, caveat, source} for a symbol's segment, or
    None if the symbol isn't in a known DoltHub segment. Prefers the cached
    recommender output; falls back to the built-in validated defaults."""
    seg = _segment_of(symbol)
    if seg is None:
        return None
    cache = _load_cache(cache_path)
    if seg in cache:
        v = dict(cache[seg])
        v.setdefault("source", "cache")
    else:
        v = dict(DEFAULT_VERDICTS[seg])
        v["source"] = "default"
    v["segment"] = seg
    return v


def verdict_line(symbol: str, cache_path: str = CACHE_PATH) -> Optional[str]:
    """One-line, plain-text verdict for the screener (display-only). None if the
    symbol has no DoltHub segment (don't show a verdict we can't back)."""
    v = verdict_for(symbol, cache_path)
    if v is None:
        return None
    src = "" if v.get("source") == "cache" else " [default]"
    line = f"Dolt verdict ({v['segment']}): {v['label']}"
    if v.get("caveat"):
        line += f"  — {v['caveat']}"
    return line + src


def build_cache(start="2022-01-01", end="2024-12-31", db_path=None,
                cache_path: str = CACHE_PATH) -> Dict[str, Any]:
    """Run the recommender per segment and write the cache. Slow (hits DoltHub)."""
    from src.dolt_research import recommend
    out: Dict[str, Any] = {}
    for seg, syms in SEGMENTS.items():
        rec = recommend(syms, start, end, db_path=db_path)
        out[seg] = {"best": rec["best"], "label": rec["action"],
                    "candidates": rec["candidates"],
                    "caveat": DEFAULT_VERDICTS.get(seg, {}).get("caveat", "")}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Per-segment Dolt verdict for the live screener")
    ap.add_argument("--build", action="store_true", help="Refresh the cache from the recommender (slow)")
    ap.add_argument("--show", metavar="SYMBOL", help="Show the verdict for a symbol")
    args = ap.parse_args()
    cfg = {}
    try:
        cfg = json.load(open("config.json")).get("dolt_options", {})
    except Exception:
        pass
    db = cfg.get("cache_path")
    if args.build:
        out = build_cache(db_path=db)
        print(f"Wrote {CACHE_PATH}:")
        for seg, v in out.items():
            print(f"  {seg:6} -> {v['label']}")
        return
    if args.show:
        line = verdict_line(args.show)
        print(line or f"{args.show}: no DoltHub segment (no verdict)")
        return
    for seg in SEGMENTS:
        sym = SEGMENTS[seg][0]
        print(f"{seg:6} ({sym}): {verdict_line(sym)}")


if __name__ == "__main__":
    _cli()
