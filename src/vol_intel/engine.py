"""Vol-intelligence orchestrator + CLI. Joins ATM-IV movers (chain archive)
with trailing realized vol / VRP (equity_ohlcv cache) over the intersection
universe and renders the report. Display-only."""
from __future__ import annotations
import argparse
from typing import List, Optional, Tuple

from src.chain_archive import DEFAULT_DB as CHAIN_DB
from src.breakout.data import DEFAULT_DB as OHLCV_DB
from src.vol_intel import atm_iv as _A
from src.vol_intel import vrp as _V
from src.vol_intel import report as _R


def build_rows(chain_db: str = CHAIN_DB, ohlcv_db: str = OHLCV_DB,
               snap_date: Optional[str] = None) -> Tuple[List[dict], List[dict]]:
    movers = _A.iv_move(chain_db, snap_date)
    vrp_rows: List[dict] = []
    for m in movers:
        sym = m["symbol"]
        pct = _V.rv_percentile(sym, ohlcv_db)
        m["rv_pctile"] = pct
        rv = _V.realized_vol_for(sym, ohlcv_db)
        if rv is not None and m.get("iv") is not None:
            row = _V.vrp_row(sym, m["iv"], rv)
            row["rv_pctile"] = pct
            vrp_rows.append(row)
    return movers, vrp_rows


def main(argv=None):
    p = argparse.ArgumentParser(description="IV / volatility-intelligence report")
    p.add_argument("--date", default=None)
    p.add_argument("--chain-db", default=CHAIN_DB)
    p.add_argument("--ohlcv-db", default=OHLCV_DB)
    args = p.parse_args(argv)
    movers, vrp_rows = build_rows(args.chain_db, args.ohlcv_db, args.date)
    if not movers:
        print("  no chain-archive data — run the screener on a weekday to grow "
              "data/chain_archive.db (needs >=2 snapshots)")
        return
    n_cov = len({r["symbol"] for r in vrp_rows})
    print(_R.render_report(movers, vrp_rows, n_cov))


if __name__ == "__main__":
    main()
