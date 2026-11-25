"""Peer comparison strip: how a ticker stacks up against its closest same-sector names.

Peers are chosen objectively — the same-sector archived names with the highest
correlation of daily log returns over the last 126 common sessions (topped up
from outside the sector, and labelled so, when the sector is too thin).
Every column is descriptive; none of them is a tested signal.

IV comes from the newest `data/chain_archive.db` snapshot for that symbol
(30-ish DTE, strike nearest spot, call/put IV averaged) and carries its own
snapshot date, because the archive covers ~40 symbols and can lag prices.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.paths import repo_path
from src.radar.scanners import get_sector_etf
from .sectors import SECTOR_NAMES, _is_fund, sector_members

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]

CORR_WINDOW = 126
MIN_COMMON = 100


@dataclass
class PeerRow:
    ticker: str
    is_subject: bool
    in_sector: bool
    corr_126d: Optional[float]
    beta_126d: Optional[float]
    ret_21d_pct: Optional[float]
    ret_63d_pct: Optional[float]
    ret_12_1_pct: Optional[float]
    rv20_pct: Optional[float]
    vs_200d_pct: Optional[float]
    from_52w_high_pct: Optional[float]
    atm_iv_pct: Optional[float] = None
    iv_rv: Optional[float] = None
    iv_as_of: Optional[str] = None
    last_bar: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PeerStrip:
    subject: str
    sector_etf: Optional[str]
    sector_name: Optional[str]
    method: str
    rows: List[PeerRow] = field(default_factory=list)
    # subject's rank among the strip for each metric, e.g. {"ret_63d_pct": "2/6"}
    standing: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject, "sector_etf": self.sector_etf, "sector_name": self.sector_name,
            "method": self.method, "rows": [r.to_dict() for r in self.rows],
            "standing": dict(self.standing), "note": self.note,
        }


def _logret_by_date(ohlcv: OhlcvMap, sym: str) -> Dict[str, float]:
    ds, cs, _ = ohlcv[sym]
    c: np.ndarray = np.asarray(cs, dtype=float)
    out: Dict[str, float] = {}
    for i in range(1, len(c)):
        if c[i] > 0 and c[i - 1] > 0:
            out[str(ds[i])] = float(math.log(c[i] / c[i - 1]))
    return out


def _corr_beta(a: Dict[str, float], b: Dict[str, float]) -> Tuple[Optional[float], Optional[float], int]:
    common = sorted(set(a) & set(b))[-CORR_WINDOW:]
    if len(common) < MIN_COMMON:
        return None, None, len(common)
    x: np.ndarray = np.asarray([a[d] for d in common], dtype=float)
    y: np.ndarray = np.asarray([b[d] for d in common], dtype=float)
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx == 0 or sy == 0:
        return None, None, len(common)
    corr = float(np.corrcoef(x, y)[0, 1])
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(y, ddof=1))
    return corr, beta, len(common)


def _metrics(ohlcv: OhlcvMap, sym: str) -> Dict[str, Optional[float]]:
    _ds, cs, _ = ohlcv[sym]
    c: np.ndarray = np.asarray(cs, dtype=float)
    out: Dict[str, Optional[float]] = {k: None for k in (
        "ret_21d_pct", "ret_63d_pct", "ret_12_1_pct", "rv20_pct", "vs_200d_pct", "from_52w_high_pct")}
    if len(c) < 2:
        return out
    if len(c) > 21:
        out["ret_21d_pct"] = round((c[-1] / c[-22] - 1.0) * 100.0, 2)
        lr = np.diff(np.log(c[-21:]))
        out["rv20_pct"] = round(float(np.std(lr, ddof=1) * math.sqrt(252.0)) * 100.0, 1)
    if len(c) > 63:
        out["ret_63d_pct"] = round((c[-1] / c[-64] - 1.0) * 100.0, 2)
    if len(c) > 252:
        out["ret_12_1_pct"] = round((c[-22] / c[-253] - 1.0) * 100.0, 2)
    if len(c) >= 200:
        out["vs_200d_pct"] = round((c[-1] / float(np.mean(c[-200:])) - 1.0) * 100.0, 2)
    out["from_52w_high_pct"] = round((c[-1] / float(np.max(c[-252:])) - 1.0) * 100.0, 2)
    return out


def archived_atm_iv(sym: str, spot: Optional[float], chain_db: str = "data/chain_archive.db",
                    target_dte: int = 30, min_dte: int = 14, max_dte: int = 65) -> Tuple[Optional[float], Optional[str]]:
    """(ATM IV as a fraction, snapshot date) from the newest archived chain, or (None, None)."""
    full = repo_path(chain_db)
    if spot is None or spot <= 0 or not os.path.exists(full):
        return None, None
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT MAX(snap_date) FROM chain_snapshots WHERE symbol = ?", (sym,)).fetchone()
            if not row or not row[0]:
                return None, None
            snap = str(row[0])
            rows = con.execute(
                "SELECT type, strike, expiration, iv FROM chain_snapshots "
                "WHERE symbol = ? AND snap_date = ? AND iv IS NOT NULL AND iv > 0",
                (sym, snap),
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None, None
    d0 = dt.date.fromisoformat(snap[:10])
    by_exp: Dict[str, List[Tuple[str, float, float]]] = {}
    for typ, k, exp, iv in rows:
        try:
            dte = (dt.date.fromisoformat(str(exp)[:10]) - d0).days
        except ValueError:
            continue
        if min_dte <= dte <= max_dte and k is not None:
            by_exp.setdefault(str(exp)[:10], []).append((str(typ or ""), float(k), float(iv)))
    if not by_exp:
        return None, snap
    exp = min(by_exp, key=lambda e: abs((dt.date.fromisoformat(e) - d0).days - target_dte))
    legs = by_exp[exp]
    k_atm = min({k for _, k, _ in legs}, key=lambda k: abs(k - float(spot)))
    ivs = [iv for _, k, iv in legs if k == k_atm and 0.01 < iv < 5.0]
    return (float(np.mean(ivs)) if ivs else None), snap


def peer_strip(ticker: str, ohlcv: OhlcvMap, *, k: int = 5, chain_db: str = "data/chain_archive.db",
               with_iv: bool = True) -> PeerStrip:
    sym = ticker.upper().strip()
    sec = get_sector_etf(sym)
    if sym in SECTOR_NAMES:  # a sector ETF: compare against the other sector/asset ETFs
        sec = None
    strip = PeerStrip(subject=sym, sector_etf=sec, sector_name=SECTOR_NAMES.get(sec or "", None), method="")
    if sym not in ohlcv:
        strip.note = f"{sym} is not in the price archive."
        return strip
    base = _logret_by_date(ohlcv, sym)
    if sym in SECTOR_NAMES or _is_fund(sym):
        pool = [s for s in SECTOR_NAMES if s != sym and s in ohlcv]
        strip.method = f"top-{k} by 126d return correlation among the 16 sector/asset ETFs"
        in_sector = set(pool)
    else:
        members = [s for s in (sector_members(sec, ohlcv) if sec else []) if s != sym]
        in_sector = set(members)
        pool = members
        if len(members) < k:
            pool = members + [s for s in ohlcv if s != sym and s not in in_sector and not _is_fund(s)]
        strip.method = (
            f"top-{k} by 126d return correlation within {sec} ({len(members)} archived names)"
            if sec else f"top-{k} by 126d return correlation across all archived stocks (no sector on file)"
        )
    scored: List[Tuple[float, str, Optional[float]]] = []
    for p in pool:
        corr, beta, _n = _corr_beta(_logret_by_date(ohlcv, p), base)
        if corr is not None:
            scored.append((corr, p, beta))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen = scored[:k]
    if any(p not in in_sector for _, p, _ in chosen):
        strip.method += "; topped up from outside the sector (marked ·)"

    spy = _logret_by_date(ohlcv, "SPY") if "SPY" in ohlcv else {}

    def _row(s: str, corr: Optional[float], is_subject: bool) -> PeerRow:
        m = _metrics(ohlcv, s)
        _c, beta_spy, _n = _corr_beta(_logret_by_date(ohlcv, s), spy) if spy else (None, None, 0)
        last = float(ohlcv[s][1][-1]) if len(ohlcv[s][1]) else None
        iv, iv_d = archived_atm_iv(s, last, chain_db) if with_iv else (None, None)
        rv = m.get("rv20_pct")
        return PeerRow(
            ticker=s, is_subject=is_subject, in_sector=is_subject or s in in_sector,
            corr_126d=None if corr is None else round(corr, 2),
            beta_126d=None if beta_spy is None else round(beta_spy, 2),
            atm_iv_pct=None if iv is None else round(iv * 100.0, 1),
            iv_rv=None if (iv is None or not rv) else round(iv * 100.0 / float(rv), 2),
            iv_as_of=iv_d, last_bar=str(ohlcv[s][0][-1]) if ohlcv[s][0] else None,
            ret_21d_pct=m["ret_21d_pct"], ret_63d_pct=m["ret_63d_pct"], ret_12_1_pct=m["ret_12_1_pct"],
            rv20_pct=m["rv20_pct"], vs_200d_pct=m["vs_200d_pct"], from_52w_high_pct=m["from_52w_high_pct"],
        )

    strip.rows.append(_row(sym, None, True))
    for corr, p, _beta in chosen:
        strip.rows.append(_row(p, corr, False))

    for key in ("ret_21d_pct", "ret_63d_pct", "ret_12_1_pct", "rv20_pct", "vs_200d_pct", "iv_rv"):
        vals = [(getattr(r, key), r.ticker) for r in strip.rows if getattr(r, key) is not None]
        mine = getattr(strip.rows[0], key)
        if mine is None or len(vals) < 2:
            continue
        order = sorted(vals, key=lambda v: -float(v[0]))
        pos = 1 + [t for _, t in order].index(sym)
        strip.standing[key] = f"{pos}/{len(order)}"
    if not chosen:
        strip.note = "No peer with >=100 common sessions in the archive."
    return strip


__all__ = ["PeerRow", "PeerStrip", "peer_strip", "archived_atm_iv"]
