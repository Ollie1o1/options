"""Real per-contract slippage measured from cached DoltHub option chains.

Replaces the flat 7%/side spread assumption used by the backtester and the live
execution path with an empirical relative-spread table bucketed by moneyness
(|delta|) and DTE. The live system's EV/preflight math is optimistic in exactly
the way the BS-vs-real backtest exposed — this gives it real numbers.

CLI:
    python -m src.dolt_slippage --table          # print the spread table
    python -m src.dolt_slippage --half-spread 0.30 30   # lookup for delta/dte
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CACHE = "data/dolt_options.db"

# Bucket edges
_DELTA_BANDS: List[Tuple[float, float, str]] = [
    (0.00, 0.15, "deep_otm"),
    (0.15, 0.35, "otm"),
    (0.35, 0.55, "atm"),
    (0.55, 0.80, "itm"),
    (0.80, 1.01, "deep_itm"),
]
_DTE_BANDS: List[Tuple[int, int, str]] = [
    (0, 14, "0-14"),
    (14, 35, "14-35"),
    (35, 70, "35-70"),
    (70, 100000, "70+"),
]


def _delta_band(delta) -> Optional[str]:
    if delta is None:
        return None
    a = abs(delta)
    for lo, hi, name in _DELTA_BANDS:
        if lo <= a < hi:
            return name
    return None


def _dte_band(dte: int) -> Optional[str]:
    for lo, hi, name in _DTE_BANDS:
        if lo <= dte < hi:
            return name
    return None


def measure_spread_table(db_path: str = DEFAULT_CACHE,
                         min_mid: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Aggregate relative spread (ask-bid)/mid from the cache, bucketed by
    (delta_band, dte_band). Returns {"delta|dte": {n, median_rel_spread, ...}}."""
    buckets: Dict[str, List[float]] = {}
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT date, expiration, delta, bid, ask, mid FROM dolt_chain "
            "WHERE bid IS NOT NULL AND ask IS NOT NULL AND mid IS NOT NULL "
            "AND mid >= ? AND ask >= bid", (min_mid,))
        for date, expiration, delta, bid, ask, mid in cur:
            db = _delta_band(delta)
            if db is None or not date or not expiration:
                continue
            try:
                dte = (_dt.date.fromisoformat(expiration) - _dt.date.fromisoformat(date)).days
            except (TypeError, ValueError):
                continue
            tb = _dte_band(dte)
            if tb is None or dte < 0:
                continue
            rel = (ask - bid) / mid
            if rel < 0 or rel > 5:   # guard against bad rows
                continue
            buckets.setdefault(f"{db}|{tb}", []).append(rel)
    out: Dict[str, Dict[str, Any]] = {}
    for key, vals in buckets.items():
        vals.sort()
        out[key] = {
            "n": len(vals),
            "median_rel_spread": round(median(vals), 4),
            "p25": round(vals[len(vals) // 4], 4),
            "p75": round(vals[(3 * len(vals)) // 4], 4),
        }
    return out


# Conservative fallback half-spread per side when a bucket has no data.
_FALLBACK_HALF_SPREAD = 0.07


def half_spread_fraction(delta, dte: int, table: Optional[Dict[str, Dict[str, Any]]] = None,
                         db_path: str = DEFAULT_CACHE) -> float:
    """Estimated cost-vs-mid per side (= half the relative spread) for a contract
    of the given delta/DTE. Falls back to 7% if the bucket has no data."""
    if table is None:
        table = measure_spread_table(db_path)
    db = _delta_band(delta)
    tb = _dte_band(int(dte)) if dte is not None else None
    if db and tb:
        cell = table.get(f"{db}|{tb}")
        if cell and cell["n"] >= 20:
            return round(cell["median_rel_spread"] / 2.0, 4)
    return _FALLBACK_HALF_SPREAD


def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Real option slippage from cached DoltHub chains")
    ap.add_argument("--db", default=DEFAULT_CACHE)
    ap.add_argument("--table", action="store_true", help="Print the spread table")
    ap.add_argument("--half-spread", nargs=2, metavar=("DELTA", "DTE"),
                    help="Look up the half-spread per side for a delta and DTE")
    args = ap.parse_args()

    table = measure_spread_table(db_path=args.db)
    if args.half_spread:
        d, t = float(args.half_spread[0]), int(args.half_spread[1])
        hs = half_spread_fraction(d, t, table=table)
        print(f"half-spread per side (delta={d}, dte={t}): {hs:.1%}  "
              f"(flat assumption is 7.0%)")
        return
    # default: print the table
    if not table:
        print("No cached chains yet — run a dolt backtest/validation first.")
        return
    print("Real relative spread (ask-bid)/mid by moneyness x DTE  [median | n]")
    print(f"  {'bucket':22} {'median':>8} {'p25':>7} {'p75':>7} {'n':>7}")
    for key in sorted(table):
        c = table[key]
        print(f"  {key:22} {c['median_rel_spread']:>8.1%} {c['p25']:>7.1%} "
              f"{c['p75']:>7.1%} {c['n']:>7}")
    print(json.dumps({"note": "half-spread per side = median/2; flat assumption was 7%"}))


if __name__ == "__main__":
    _cli()
