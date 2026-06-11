"""Cross-source verification: Yahoo vs CBOE, per contract, on demand.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.cross_check AAPL

Joins the two delayed chains on (type, strike, expiration) and reports how
often they agree on mid price and implied volatility. This is the free version
of the "second data source" trust step the roadmap deferred to a paid Polygon
plan: CBOE's chain is unauthenticated and carries exchange-computed IV and
Greeks, so systematic disagreement points at Yahoo data problems (or vice
versa) before any money is at risk.

Tolerances (same spirit as data_quality's internal IV check):
  IV:  relative |y−c|/c ≤ 15%
  mid: relative ≤ 10% OR absolute ≤ $0.05 (cheap options: relative is noise)
"""
from __future__ import annotations

from typing import Any, Dict, List

IV_REL_TOL = 0.15
MID_REL_TOL = 0.10
MID_ABS_TOL = 0.05


def _mid(row: Dict[str, Any]):
    bid, ask = row.get("bid"), row.get("ask")
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if bid <= 0 and ask <= 0:
        return None
    return (bid + ask) / 2.0


def _key(row: Dict[str, Any]):
    try:
        return (str(row["type"]).lower(), round(float(row["strike"]), 4),
                str(row["expiration"])[:10])
    except (KeyError, TypeError, ValueError):
        return None


def compare(yahoo_rows: List[Dict[str, Any]],
            cboe_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: join on (type, strike, expiration) and measure agreement."""
    cboe_by_key = {}
    for r in cboe_rows:
        k = _key(r)
        if k:
            cboe_by_key[k] = r

    matched = yahoo_only = 0
    iv_compared = iv_agree = 0
    mid_compared = mid_agree = 0
    disagreements: List[Dict[str, Any]] = []

    for y in yahoo_rows:
        k = _key(y)
        if not k:
            continue
        c = cboe_by_key.get(k)
        if c is None:
            yahoo_only += 1
            continue
        matched += 1
        why = []

        y_iv, c_iv = y.get("iv"), c.get("iv")
        if y_iv and c_iv:
            iv_compared += 1
            if abs(float(y_iv) - float(c_iv)) / float(c_iv) <= IV_REL_TOL:
                iv_agree += 1
            else:
                why.append(f"iv yahoo={float(y_iv):.2f} cboe={float(c_iv):.2f}")

        y_mid, c_mid = _mid(y), _mid(c)
        if y_mid is not None and c_mid is not None and c_mid > 0:
            mid_compared += 1
            if (abs(y_mid - c_mid) / c_mid <= MID_REL_TOL
                    or abs(y_mid - c_mid) <= MID_ABS_TOL):
                mid_agree += 1
            else:
                why.append(f"mid yahoo={y_mid:.2f} cboe={c_mid:.2f}")

        if why:
            disagreements.append({"key": k, "why": ", ".join(why)})

    return {
        "matched": matched,
        "yahoo_only": yahoo_only,
        "iv_compared": iv_compared,
        "iv_agree": iv_agree,
        "mid_compared": mid_compared,
        "mid_agree": mid_agree,
        "disagreements": disagreements,
    }


def _yahoo_rows(symbol: str) -> List[Dict[str, Any]]:
    """Fetch Yahoo's chain via the existing pipeline, normalized for compare()."""
    from src.data_fetching import fetch_options_yfinance
    result = fetch_options_yfinance(symbol, max_expiries=6)
    df = (result or {}).get("df")
    if df is None or getattr(df, "empty", True):
        return []
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "type": r.get("type"),
            "strike": r.get("strike"),
            "expiration": str(r.get("expiration"))[:10],
            "bid": r.get("bid"),
            "ask": r.get("ask"),
            "iv": r.get("impliedVolatility"),
        })
    return rows


def main() -> None:
    import argparse
    from src import cboe_client

    ap = argparse.ArgumentParser(description="Yahoo vs CBOE per-contract cross-check")
    ap.add_argument("symbol")
    ap.add_argument("--show", type=int, default=10, help="max disagreements to print")
    args = ap.parse_args()
    sym = args.symbol.upper()

    print(f"Fetching CBOE chain for {sym} …")
    cboe = cboe_client.fetch_chain(sym)
    print(f"  {len(cboe)} CBOE contracts")
    print(f"Fetching Yahoo chain for {sym} …")
    yahoo = _yahoo_rows(sym)
    print(f"  {len(yahoo)} Yahoo contracts (post-filter pipeline)")

    r = compare(yahoo, cboe)
    print(f"\nCross-source agreement — {sym}")
    print(f"  matched contracts: {r['matched']}  (yahoo-only: {r['yahoo_only']})")
    if r["iv_compared"]:
        print(f"  IV  agree: {r['iv_agree']}/{r['iv_compared']} "
              f"({r['iv_agree']/r['iv_compared']:.0%}) within ±{IV_REL_TOL:.0%}")
    if r["mid_compared"]:
        print(f"  mid agree: {r['mid_agree']}/{r['mid_compared']} "
              f"({r['mid_agree']/r['mid_compared']:.0%}) within ±{MID_REL_TOL:.0%}/$"
              f"{MID_ABS_TOL:.02f}")
    if r["disagreements"]:
        print(f"\n  worst disagreements (showing ≤{args.show}):")
        for d in r["disagreements"][:args.show]:
            t, k, e = d["key"]
            print(f"    {t} {k} {e}: {d['why']}")
    if not r["matched"]:
        print("  (no overlap — check the symbol or market hours)")


if __name__ == "__main__":
    main()
