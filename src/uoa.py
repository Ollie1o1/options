"""Unusual options activity from the chain archive's day-over-day deltas.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.uoa [--date D] [--top N]

Informed traders leak into option volume and fresh open interest before stock
moves (Pan & Poteshman 2006). Paid services sell this as "flow"; the daily
CBOE snapshots in ``data/chain_archive.db`` give the EOD version free: for
each contract, how much open interest appeared since the previous snapshot
and how today's volume compares to the existing OI base.

What counts as unusual (per contract):
  - OI jump: ΔOI ≥ 500 contracts AND ≥ +30% vs the prior OI base, or
  - volume spike: volume ≥ 3× prior OI (heavy positioning before OI updates).

Per symbol, flows aggregate into premium-weighted call vs put OI added and a
``net_call_share`` ∈ [0,1] (1 = all new money in calls). This is an
information overlay — it does NOT touch quality_score or any weights while
the Phase-2 gate is gathering. Needs ≥2 days of snapshots; the archive grows
one day per weekday automatically.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from src.chain_archive import DEFAULT_DB

OI_JUMP_MIN = 500          # contracts
OI_JUMP_PCT = 0.30         # vs prior OI base
VOL_VS_OI_MULT = 3.0       # volume >= 3x prior OI


def _prev_snap_date(conn: sqlite3.Connection, symbol: str, date: str) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(snap_date) FROM chain_snapshots WHERE symbol=? AND snap_date<?",
        (symbol, date)).fetchone()
    return row[0] if row and row[0] else None


def oi_deltas(conn: sqlite3.Connection, symbol: str, date: str) -> List[Dict[str, Any]]:
    """Per-contract day-over-day deltas for `symbol` at `date` vs the previous
    available snapshot. New contracts count their full OI as added. Returns []
    when there is no earlier snapshot to diff against."""
    prev = _prev_snap_date(conn, symbol, date)
    if not prev:
        return []
    sql = """
        SELECT t.contract, t.type, t.strike, t.expiration,
               t.open_interest, COALESCE(p.open_interest, 0),
               t.volume, t.bid, t.ask, t.spot, t.snap_date
        FROM chain_snapshots t
        LEFT JOIN chain_snapshots p
               ON p.contract = t.contract AND p.snap_date = ?
        WHERE t.symbol = ? AND t.snap_date = ?
    """
    out = []
    for (contract, type_, strike, expiration, oi, prev_oi, volume,
         bid, ask, spot, snap) in conn.execute(sql, (prev, symbol, date)):
        oi, prev_oi = float(oi or 0), float(prev_oi or 0)
        mid = ((bid or 0) + (ask or 0)) / 2.0
        try:
            from datetime import datetime
            dte = (datetime.strptime(expiration[:10], "%Y-%m-%d")
                   - datetime.strptime(snap[:10], "%Y-%m-%d")).days
        except (TypeError, ValueError):
            dte = None
        is_otm = bool(spot) and (
            (type_ == "call" and strike > spot) or
            (type_ == "put" and strike < spot))
        out.append({
            "contract": contract, "type": type_, "strike": strike,
            "expiration": expiration, "d_oi": oi - prev_oi, "prev_oi": prev_oi,
            "volume": float(volume or 0), "mid": mid, "is_otm": is_otm,
            "dte": dte,
        })
    return out


def symbol_flow(deltas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-contract deltas into a symbol-level flow read. Pure."""
    call_added = put_added = 0.0
    call_prem = put_prem = 0.0
    unusual: List[Dict[str, Any]] = []
    for d in deltas:
        added = max(0.0, d["d_oi"])
        prem = added * d.get("mid", 0) * 100
        if d["type"] == "call":
            call_added += added; call_prem += prem
        else:
            put_added += added; put_prem += prem
        jump = (d["d_oi"] >= OI_JUMP_MIN
                and d["d_oi"] >= OI_JUMP_PCT * max(d["prev_oi"], 1.0))
        spike = d["volume"] >= VOL_VS_OI_MULT * max(d["prev_oi"], 100.0)
        if jump or spike:
            unusual.append({**d, "why": "oi_jump" if jump else "volume_spike"})
    total_prem = call_prem + put_prem
    return {
        "call_oi_added": call_added,
        "put_oi_added": put_added,
        "call_premium_added": call_prem,
        "put_premium_added": put_prem,
        "net_call_share": (call_prem / total_prem) if total_prem > 0 else 0.5,
        "unusual": sorted(unusual, key=lambda u: -u["d_oi"]),
    }


def uoa_report(db_path: str = DEFAULT_DB, date: Optional[str] = None,
               symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Ranked unusual-activity report across symbols for `date` (default:
    latest snapshot day)."""
    with sqlite3.connect(db_path) as conn:
        days = [r[0] for r in conn.execute(
            "SELECT DISTINCT snap_date FROM chain_snapshots ORDER BY snap_date")]
        if not date:
            date = days[-1] if days else None
        if symbols is None:
            symbols = [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM chain_snapshots ORDER BY symbol")]
        rows = []
        if date and len(days) >= 2:
            for sym in symbols:
                deltas = oi_deltas(conn, sym, date)
                if not deltas:
                    continue
                flow = symbol_flow(deltas)
                score = (len(flow["unusual"])
                         + abs(flow["net_call_share"] - 0.5) * 2)
                rows.append({"symbol": sym, "score": score, **flow})
        rows.sort(key=lambda r: -r["score"])
    return {"date": date, "days_available": len(days), "rows": rows}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Unusual options activity (archive OI deltas)")
    ap.add_argument("--date", help="snapshot date YYYY-MM-DD (default: latest)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    r = uoa_report(db_path=args.db, date=args.date)
    if r["days_available"] < 2:
        print(f"Chain archive has {r['days_available']} day(s) of snapshots — "
              f"OI deltas need ≥2. The archive grows automatically every "
              f"weekday; check back tomorrow.")
        return
    print(f"Unusual options activity — {r['date']} vs previous snapshot\n")
    for row in r["rows"][:args.top]:
        ncs = row["net_call_share"]
        lean = "CALL-heavy" if ncs > 0.65 else "PUT-heavy" if ncs < 0.35 else "balanced"
        print(f"  {row['symbol']:6s} new-money: {lean} ({ncs:.0%} calls) | "
              f"call OI +{row['call_oi_added']:,.0f} / put OI +{row['put_oi_added']:,.0f} "
              f"| unusual contracts: {len(row['unusual'])}")
        for u in row["unusual"][:3]:
            print(f"          {u['type']} {u['strike']} {u['expiration']} "
                  f"ΔOI +{u['d_oi']:,.0f} (prev {u['prev_oi']:,.0f}), "
                  f"vol {u['volume']:,.0f} [{u['why']}]")
    if not r["rows"]:
        print("  nothing unusual today.")
    print("\nOverlay only — not wired into scoring while the gate is gathering.")


if __name__ == "__main__":
    main()
