"""Taxable ETF Calendar / Diagonal Spread Term-Structure Scanner (`src/radar/calendar_spreads.py`).

Scans `data/chain_archive.db` for liquid ETFs with inverted or steep front-vs-back IV
term structure (`front_iv - back_iv >= min_iv_diff_vp`) suitable for selling rich 20–40 DTE
options against cheaper 75–120 DTE back-month options.

CRA Guard: Multi-leg calendar/diagonal spreads are strictly fenced `[TAXABLE_ONLY]`
and never recommended inside a Canadian TFSA.
Evidence label: WATCHLIST (research / taxable term-structure screen; preregister OOS before sizing).
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.paths import repo_path
from src.structure.chain_data import load_archived_chain_slice

CALENDAR_ETF_UNIVERSE = (
    "SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLV", "XLP", "SMH", "GLD", "TLT",
)


@dataclass(frozen=True)
class CalendarCandidate:
    ticker: str
    spot: float
    opt_type: str               # "call" | "put"
    strike: float
    front_exp: str
    front_dte: int
    back_exp: str
    back_dte: int
    front_iv_pct: float
    back_iv_pct: float
    iv_diff_vp: float           # front_iv - back_iv in vol points (positive = rich front vol)
    net_debit_mid: float
    net_debit_cross: float
    friction_pct: float
    term_regime: str            # "FRONT_RICH_INVERTED" | "FLAT" | "CONTANGO"
    evidence_label: str = "WATCHLIST"
    accounts: str = "TAXABLE_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def scan_etf_calendar_spreads(
    *,
    symbols: Sequence[str] = CALENDAR_ETF_UNIVERSE,
    chain_db: str = "data/chain_archive.db",
    min_iv_diff_vp: float = 0.5,
    max_friction_pct: float = 0.25,
) -> List[CalendarCandidate]:
    """Scan archived chains for liquid ATM ETF calendar spreads where front IV > back IV."""
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return []

    # Latest spots from chain_archive.db
    spots: Dict[str, float] = {}
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "chain_snapshots" not in tables:
            return []
        for tk in symbols:
            row = con.execute(
                "SELECT spot FROM chain_snapshots WHERE symbol = ? AND spot > 0 ORDER BY snap_date DESC LIMIT 1",
                (tk.upper(),),
            ).fetchone()
            if row and row[0]:
                spots[tk.upper()] = float(row[0])
    finally:
        con.close()

    out: List[CalendarCandidate] = []
    for tk in symbols:
        sym = tk.upper()
        sp = spots.get(sym)
        if not sp or sp <= 0:
            continue
        front = load_archived_chain_slice(
            sym, target_dte=30, min_dte=14, max_dte=48, spot=sp, chain_db=chain_db, build_rnd=False
        )
        back = load_archived_chain_slice(
            sym, target_dte=90, min_dte=60, max_dte=130, spot=sp, chain_db=chain_db, build_rnd=False
        )
        if front is None or back is None or front.expiration == back.expiration:
            continue
        if front.atm_iv is None or back.atm_iv is None:
            continue
        iv_diff = round((front.atm_iv - back.atm_iv) * 100.0, 2)
        if iv_diff < min_iv_diff_vp:
            continue

        # Match shared ATM call (or put) strike across both expirations
        for opt_type in ("call", "put"):
            fc = front.match_contract(opt_type, sp, max_rel_dist=0.03)
            if fc is None:
                continue
            bc = back.match_contract(opt_type, fc.strike, max_rel_dist=0.01)
            if bc is None or abs(bc.strike - fc.strike) > 1e-6:
                continue
            debit_mid = round(bc.mid - fc.mid, 2)
            debit_cross = round(bc.ask - fc.bid, 2)
            if debit_mid <= 0.05 or debit_cross <= 0.05:
                continue
            rt_spread = (bc.ask - bc.bid) + (fc.ask - fc.bid)
            fric = rt_spread / max(debit_mid, 1e-6)
            if fric > max_friction_pct:
                continue
            regime = "FRONT_RICH_INVERTED" if iv_diff >= 1.5 else "FLAT"
            out.append(
                CalendarCandidate(
                    ticker=sym,
                    spot=round(sp, 2),
                    opt_type=opt_type,
                    strike=fc.strike,
                    front_exp=front.expiration,
                    front_dte=front.dte,
                    back_exp=back.expiration,
                    back_dte=back.dte,
                    front_iv_pct=round(front.atm_iv * 100.0, 1),
                    back_iv_pct=round(back.atm_iv * 100.0, 1),
                    iv_diff_vp=iv_diff,
                    net_debit_mid=debit_mid,
                    net_debit_cross=debit_cross,
                    friction_pct=round(fric, 4),
                    term_regime=regime,
                )
            )
            break

    out.sort(key=lambda c: (-c.iv_diff_vp, c.friction_pct, c.ticker))
    return out
