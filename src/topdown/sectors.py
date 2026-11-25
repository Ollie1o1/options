"""Sector & asset board: the validated Outlook rank, plus descriptive breadth and relative strength.

Ordering is the Outlook composite rank from `src.outlook.vm_rotation` — the one
ordering here with validated evidence behind it (`vol_managed_outlook`,
VALIDATED; `docs/OUTLOOK_FINDINGS.md`: rank IC +0.05..+0.08 at 42-63d on these
16 instruments). Everything else on a row is description:

  * relative strength vs SPY over 21d and 63d;
  * breadth — the share of that sector's archived constituents above their own
    50d / 200d averages (GICS membership via `src.radar.scanners.get_sector_map`;
    XLK breadth includes semiconductors, as the XLK fund does; SMH breadth is the
    semiconductor subset). Constituents whose last bar is more than
    `stale_days` behind the board date are excluded and counted.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.outlook.vm_rotation import RotationTarget, align_closes, latest_complete_index, target_weights
from src.radar.scanners import OUTLOOK_ETFS, get_sector_map

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]

SECTOR_NAMES: Dict[str, str] = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLI": "Industrials", "XLP": "Consumer Staples", "XLY": "Consumer Discretionary",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
    "SMH": "Semiconductors", "GLD": "Gold", "TLT": "Long Treasuries", "IWM": "US Small Caps",
    "QQQ": "Nasdaq-100",
}

SEMICONDUCTORS = frozenset({
    "NVDA", "AVGO", "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "TXN", "MRVL",
    "ON", "ADI", "NXPI", "MCHP", "ARM", "ASML", "TSM", "RMBS",
})

_SECTOR_ETFS = frozenset(SECTOR_NAMES) - {"GLD", "TLT", "IWM", "QQQ"}


def _fund_set() -> frozenset:
    from src.equity_playbooks.universe import CORE_ETFS

    return frozenset(CORE_ETFS) | frozenset(SECTOR_NAMES) | frozenset(OUTLOOK_ETFS) | {"ARKK", "SPY", "SPHB", "SPLV", "CPER", "UUP"}


_FUNDS = _fund_set()


def _is_fund(sym: str) -> bool:
    return sym in _FUNDS


def sector_members(etf: str, ohlcv: OhlcvMap) -> List[str]:
    """Archived single-name constituents of a sector ETF (GICS membership, funds excluded)."""
    etf = etf.upper()
    if etf not in _SECTOR_ETFS:
        return []
    smap = get_sector_map()
    out: List[str] = []
    for sym in ohlcv:
        if _is_fund(sym):
            continue
        sec = smap.get(sym)
        if etf == "SMH":
            if sym in SEMICONDUCTORS or sec == "SMH":
                out.append(sym)
        elif etf == "XLK":
            if sec in ("XLK", "SMH") or sym in SEMICONDUCTORS:
                out.append(sym)
        elif sec == etf:
            out.append(sym)
    return sorted(out)


def _ret(c: np.ndarray, n: int) -> Optional[float]:
    if len(c) <= n or c[-1 - n] <= 0:
        return None
    return float(c[-1] / c[-1 - n] - 1.0)


def _rv20(c: np.ndarray) -> Optional[float]:
    if len(c) < 22:
        return None
    lr = np.diff(np.log(c[-21:]))
    return float(np.std(lr, ddof=1) * math.sqrt(252.0))


def _pct(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(x * 100.0, nd)


@dataclass
class SectorRow:
    etf: str
    name: str
    rank: Optional[int]
    score: Optional[float]
    direction: str
    conviction: Optional[int]
    posture: str                      # FAVOR (held by the sleeve) | AVOID (regime-gated BEARISH) | NEUTRAL
    weight: float                     # VALIDATED sleeve weight (0 if not held)
    last: Optional[float] = None
    ret_21d_pct: Optional[float] = None
    ret_63d_pct: Optional[float] = None
    rs_21d_pct: Optional[float] = None
    rs_63d_pct: Optional[float] = None
    vs_200d_pct: Optional[float] = None
    rv20_pct: Optional[float] = None
    n_members: int = 0
    n_stale: int = 0
    pct_above_50d: Optional[float] = None
    pct_above_200d: Optional[float] = None
    drivers: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SectorBoard:
    as_of: Optional[str]
    rows: List[SectorRow] = field(default_factory=list)
    sleeve_scale: Optional[float] = None
    sleeve_cash: Optional[float] = None
    basket_rv20: Optional[float] = None
    excluded: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    target: Optional[RotationTarget] = None

    def row(self, etf: str) -> Optional[SectorRow]:
        return next((r for r in self.rows if r.etf == etf.upper()), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of, "rows": [r.to_dict() for r in self.rows],
            "sleeve_scale": self.sleeve_scale, "sleeve_cash": self.sleeve_cash,
            "basket_rv20": self.basket_rv20, "excluded": dict(self.excluded), "notes": list(self.notes),
        }


def _breadth(members: Sequence[str], ohlcv: OhlcvMap, board_date: str, stale_days: int) -> Tuple[int, int, Optional[float], Optional[float]]:
    cutoff = (dt.date.fromisoformat(board_date[:10]) - dt.timedelta(days=stale_days)).isoformat()
    a50 = a200 = n50 = n200 = stale = 0
    for sym in members:
        ds, cs, _ = ohlcv[sym]
        if not ds or str(ds[-1]) < cutoff:
            stale += 1
            continue
        c: np.ndarray = np.asarray(cs, dtype=float)
        if len(c) >= 50:
            n50 += 1
            a50 += int(c[-1] > float(np.mean(c[-50:])))
        if len(c) >= 200:
            n200 += 1
            a200 += int(c[-1] > float(np.mean(c[-200:])))
    return (
        len(members) - stale,
        stale,
        round(100.0 * a50 / n50, 0) if n50 else None,
        round(100.0 * a200 / n200, 0) if n200 else None,
    )


def build_sector_board(ohlcv: OhlcvMap, *, stale_days: int = 5) -> SectorBoard:
    """16-row sector/asset board ordered by the VALIDATED Outlook rank."""
    if "SPY" not in ohlcv:
        return SectorBoard(as_of=None, notes=["SPY missing from the price archive."])
    dates, cols = align_closes(ohlcv, OUTLOOK_ETFS)
    t = latest_complete_index(cols, OUTLOOK_ETFS)
    tgt = target_weights(dates, cols, t) if t is not None else None
    if t is None or tgt is None:
        return SectorBoard(as_of=None, notes=["Not enough aligned history to rank the Outlook universe."])

    spy: np.ndarray = cols["SPY"][: t + 1]
    spy21, spy63 = _ret(spy, 21), _ret(spy, 63)
    board = SectorBoard(
        as_of=tgt.as_of, sleeve_scale=round(tgt.scale, 3), sleeve_cash=round(tgt.cash, 4),
        basket_rv20=None if tgt.basket_rv20 is None else round(tgt.basket_rv20, 4),
        excluded=dict(tgt.excluded), target=tgt,
    )
    ranked = {str(r["ticker"]): (i, r) for i, r in enumerate(tgt.ranked, 1)}
    order = sorted(OUTLOOK_ETFS, key=lambda e: ranked[e][0] if e in ranked else 99)
    for etf in order:
        rk = ranked.get(etf)
        r: Dict[str, Any] = rk[1] if rk else {}
        if etf in tgt.weights:
            posture = "FAVOR"
        elif r.get("direction") == "BEARISH":
            posture = "AVOID"
        else:
            posture = "NEUTRAL"
        row = SectorRow(
            etf=etf, name=SECTOR_NAMES.get(etf, etf),
            rank=rk[0] if rk else None,
            score=None if not r else round(float(r["score"]), 3),
            direction=str(r.get("direction", "—")),
            conviction=None if not r else int(r["conviction"]),
            posture=posture, weight=round(tgt.weights.get(etf, 0.0), 4),
            drivers=str(r.get("drivers", "")),
        )
        if etf in cols:
            c = cols[etf][: t + 1]
            c = c[np.isfinite(c)]
            if len(c) > 1:
                row.last = round(float(c[-1]), 2)
                r21, r63 = _ret(c, 21), _ret(c, 63)
                row.ret_21d_pct, row.ret_63d_pct = _pct(r21), _pct(r63)
                if r21 is not None and spy21 is not None:
                    row.rs_21d_pct = _pct(r21 - spy21)
                if r63 is not None and spy63 is not None:
                    row.rs_63d_pct = _pct(r63 - spy63)
                if len(c) >= 200:
                    row.vs_200d_pct = _pct(float(c[-1]) / float(np.mean(c[-200:])) - 1.0)
                row.rv20_pct = _pct(_rv20(c), 1)
        members = sector_members(etf, ohlcv)
        if members:
            row.n_members, row.n_stale, row.pct_above_50d, row.pct_above_200d = _breadth(
                members, ohlcv, tgt.as_of, stale_days
            )
        board.rows.append(row)
    board.notes.append(
        "Order = VALIDATED Outlook rank (vol_managed_outlook). Breadth/RS columns are descriptive, untested as signals."
    )
    return board


__all__ = ["SECTOR_NAMES", "SEMICONDUCTORS", "SectorRow", "SectorBoard", "sector_members", "build_sector_board"]
