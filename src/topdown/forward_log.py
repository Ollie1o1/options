"""Forward record of the Discover lenses, and the preregistered read of it.

Preregistration: docs/PREREG_DISCOVER_LENS_FORWARD_20261125.md (frozen before the first row).

`record_results` stores what each lens listed, keyed on (lens, ticker, the row's own data
date) with INSERT OR IGNORE — a cached input re-recorded on later days counts once, on its own
date. `forward_report` refuses to compute any outcome before READ_NOT_BEFORE and MIN_WEEKS
complete lens-weeks; until then it reports counts only, so nobody can peek and re-tune.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DB = os.path.join("data", "lens_forward.db")
DB_PATH_ENV = "OPTIONS_LENS_FORWARD_DB"   # the test runners point this at a temp file
PREREG = "docs/PREREG_DISCOVER_LENS_FORWARD_20261125.md"
READ_NOT_BEFORE = "2027-05-25"
MIN_WEEKS = 26
HORIZONS: Dict[str, int] = {
    "core_rotation": 21, "pullbacks": 10, "leaders": 21,
    "revisions": 21, "insiders": 21, "landmines": 21,
}
COST = 0.001            # 10 bps round trip, subtracted from each excess return
SEED = 20261125
N_BOOT = 10_000
BONFERRONI_LEVEL = 1.0 - 0.05 / len(HORIZONS)   # 99.17%

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lens_rows (
    lens_id TEXT NOT NULL, ticker TEXT NOT NULL, row_as_of TEXT NOT NULL,
    label TEXT, sector_etf TEXT, metric_label TEXT, metric REAL, flags TEXT,
    playbook_id TEXT, why TEXT, record_date TEXT NOT NULL,
    PRIMARY KEY (lens_id, ticker, row_as_of)
);
CREATE TABLE IF NOT EXISTS lens_runs (
    record_date TEXT NOT NULL, lens_id TEXT NOT NULL, label TEXT, n_rows INTEGER,
    board_as_of TEXT, note TEXT,
    PRIMARY KEY (record_date, lens_id)
);
"""


def _path(db_path: Optional[str]) -> str:
    """Resolve the DB: explicit path > $OPTIONS_LENS_FORWARD_DB > data/lens_forward.db."""
    from src.paths import repo_path

    p = db_path or os.environ.get(DB_PATH_ENV) or DEFAULT_DB
    return p if os.path.isabs(p) else repo_path(p)


def ensure_db(db_path: Optional[str] = None) -> str:
    full = _path(db_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with closing(sqlite3.connect(full)) as con:
        con.executescript(_SCHEMA)
    return full


def record_results(results: Sequence[Any], *, db_path: Optional[str] = None, record_date: Optional[str] = None,
                   board_as_of: Optional[str] = None) -> Dict[str, int]:
    """Store every lens's rows (and the run itself, empty or not). Returns counts."""
    full = ensure_db(db_path)
    rd = record_date or dt.date.today().isoformat()
    new = seen = 0
    con = sqlite3.connect(full)
    try:
        for res in results:
            lz = res.lens
            con.execute("INSERT OR REPLACE INTO lens_runs VALUES (?,?,?,?,?,?)",
                        (rd, lz.id, lz.label, len(res.rows), board_as_of, res.note or ""))
            for row in res.rows:
                as_of = str(row.as_of or board_as_of or rd)[:10]
                cur = con.execute(
                    "INSERT OR IGNORE INTO lens_rows VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (lz.id, str(row.ticker).upper(), as_of, row.evidence_label or lz.label,
                     row.sector_etf, row.metric_label, row.metric, json.dumps(list(row.flags or ())),
                     row.playbook_id, row.why, rd),
                )
                seen += 1
                new += int(cur.rowcount or 0)
        con.commit()
    finally:
        con.close()
    return {"lenses": len(results), "rows_seen": seen, "rows_new": new}


def record_hub(hub: Any, *, db_path: Optional[str] = None, record_date: Optional[str] = None) -> Dict[str, int]:
    return record_results(hub.lenses, db_path=db_path, record_date=record_date,
                          board_as_of=getattr(hub.board, "as_of", None))


def due(state: Optional[dict], today: str, hhmm: int) -> bool:
    """Once per supported trading day, after settled daily prices are eligible."""
    if hhmm < 1800:
        return False
    from src.data_quality import holiday_calendar_covers, is_trading_day

    if not is_trading_day(today) or not holiday_calendar_covers(int(today[:4])):
        return False
    return (state or {}).get("last_lens_snapshot") != today


def run_snapshot(*, db_path: Optional[str] = None, record_date: Optional[str] = None,
                 expected_session: Optional[str] = None, **hub_kwargs: Any) -> Dict[str, Any]:
    """Record the hub only once the expected session is available for scheduled runs."""
    from src.topdown.hub import build_hub

    hub = build_hub(**hub_kwargs)
    if expected_session:
        if hub.board.as_of != expected_session or any(
            not series[0] or series[0][-1] != expected_session for series in hub.ohlcv.values()
        ):
            return {"complete": False, "reason": "equity archive has not settled for the expected session"}
        if hub.snapshot is None or hub.snapshot.ohlcv_as_of != expected_session or hub.snapshot.degraded_scanners:
            from src.radar.engine import run_radar

            snapshot = run_radar(
                ohlcv_db=hub_kwargs.get("ohlcv_db", "data/equity_ohlcv.db"),
                chain_db=hub_kwargs.get("chain_db", "data/chain_archive.db"),
                cache_path=None,
            )
            if snapshot.ohlcv_as_of != expected_session or snapshot.degraded_scanners:
                return {"complete": False, "reason": "Radar could not build a complete current snapshot"}
            hub = build_hub(**{**hub_kwargs, "ohlcv": hub.ohlcv, "snapshot": snapshot})
    return {**record_hub(hub, db_path=db_path, record_date=record_date), "complete": True}


# ── reading ──────────────────────────────────────────────────────────────────

def _iso_week(day: str) -> str:
    y, w, _ = dt.date.fromisoformat(day[:10]).isocalendar()
    return f"{y}-W{w:02d}"


def lens_weeks(db_path: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Tuple[str, Optional[str]]]]]:
    """Group by data week. Entry cannot precede the first recorded observation."""
    full = _path(db_path)
    out: Dict[str, Dict[str, Dict[str, Tuple[str, Optional[str]]]]] = {}
    if not os.path.exists(full):
        return out
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT lens_id, ticker, row_as_of, sector_etf, record_date FROM lens_rows "
                           "ORDER BY lens_id, row_as_of, ticker").fetchall()
    finally:
        con.close()
    for lens_id, tk, as_of, sec, recorded in rows:
        wk = out.setdefault(lens_id, {}).setdefault(_iso_week(as_of), {})
        entry_anchor = max(as_of, recorded)
        if tk not in wk or entry_anchor < wk[tk][0]:
            wk[tk] = (entry_anchor, sec)
    return out


@dataclass
class LensForward:
    lens_id: str
    label: str
    horizon: int
    weeks_recorded: int
    rows: int
    first: Optional[str]
    last: Optional[str]
    locked: bool
    lock_reason: str = ""
    weeks_complete: int = 0
    mean_excess: Optional[float] = None
    ci95: Optional[Tuple[float, float]] = None
    ci_bonf: Optional[Tuple[float, float]] = None
    mean_excess_vs_sector: Optional[float] = None
    proposal: str = ""
    weekly: List[Tuple[str, float]] = field(default_factory=list)


def _excess_for_week(hits: Dict[str, Tuple[str, Optional[str]]], H: int, ohlcv: Dict[str, Any],
                     cal: List[str], spy: Dict[str, float]) -> Tuple[Optional[float], Optional[float], bool]:
    """(mean excess vs SPY, mean excess vs sector ETF, complete?) for one lens-week."""
    import bisect

    ex: List[float] = []
    ex_sec: List[float] = []
    complete = True
    for tk, (entry_anchor, sec) in hits.items():
        i = bisect.bisect_right(cal, entry_anchor[:10])
        j = i + H
        if j >= len(cal):
            complete = False
            continue
        d0, d1 = cal[i], cal[j]
        series = ohlcv.get(tk)
        if series is None:
            complete = False
            continue
        px = dict(zip(series[0], series[1]))
        p0, p1 = px.get(d0), px.get(d1)
        if not p0 or not p1 or not math.isfinite(p0) or not math.isfinite(p1) or p0 <= 0 or p1 <= 0:
            complete = False
            continue
        r = math.log(p1 / p0)
        ex.append(r - math.log(spy[d1] / spy[d0]) - COST)
        if sec and sec in ohlcv and sec != tk:
            sp = dict(zip(ohlcv[sec][0], ohlcv[sec][1]))
            s0, s1 = sp.get(d0), sp.get(d1)
            if s0 and s1 and s0 > 0 and s1 > 0:
                ex_sec.append(r - math.log(s1 / s0) - COST)
    return (float(np.mean(ex)) if ex else None, float(np.mean(ex_sec)) if ex_sec else None, complete)


def _block_boot_ci(x: np.ndarray, block: int, levels: Sequence[float]) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(SEED)
    n = len(x)
    block = max(1, min(block, n))
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOT, -1)[:, :n]
    means = x[idx].mean(axis=1)
    out = []
    for lv in levels:
        a = (1.0 - lv) / 2.0
        out.append((float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))))
    return out


def _proposal(label: str, ci_bonf: Tuple[float, float], mean: float) -> str:
    lo, hi = ci_bonf
    if hi < 0:
        return "confirmed AVOID" if label == "AVOID" else ("drift flag — forward CI below zero" if label == "PROVEN"
                                                          else "propose AVOID")
    if label == "PROVEN":
        return "monitor (no drift flag)"
    if lo > 0 and label in ("CONTEXT", "WATCHLIST", "TESTING"):
        return "propose PROMISING"
    if label == "PROMISING" and mean <= 0:
        return "flag for review (point estimate <= 0)"
    return "no change"


def forward_report(*, db_path: Optional[str] = None, ohlcv: Optional[Dict[str, Any]] = None,
                   today: Optional[str] = None, ohlcv_db: str = "data/equity_ohlcv.db") -> List[LensForward]:
    """Per evaluated lens: counts always; outcomes only once the read is unlocked."""
    today = today or dt.date.today().isoformat()
    weeks = lens_weeks(db_path)
    from src.topdown.lenses import LENS_BY_ID

    full = _path(db_path)
    counts: Dict[str, Tuple[int, Optional[str], Optional[str]]] = {}
    if os.path.exists(full):
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            for lid, n, a, b in con.execute(
                    "SELECT lens_id, COUNT(*), MIN(row_as_of), MAX(row_as_of) FROM lens_rows GROUP BY lens_id"):
                counts[lid] = (int(n), a, b)
        finally:
            con.close()

    date_locked = today < READ_NOT_BEFORE
    need_prices = not date_locked and any(len(weeks.get(lid, {})) >= MIN_WEEKS for lid in HORIZONS)
    if need_prices and ohlcv is None:
        from src.radar.scanners import load_local_ohlcv_safe

        ohlcv = load_local_ohlcv_safe(ohlcv_db)
    out: List[LensForward] = []
    for lid, H in HORIZONS.items():
        lz = LENS_BY_ID.get(lid)
        label = lz.label if lz else "?"
        wk = weeks.get(lid, {})
        n_rows, first, last = counts.get(lid, (0, None, None))
        lf = LensForward(lens_id=lid, label=label, horizon=H, weeks_recorded=len(wk), rows=n_rows,
                         first=first, last=last, locked=True)
        if date_locked:
            lf.lock_reason = f"locked until {READ_NOT_BEFORE} (preregistered)"
            out.append(lf)
            continue
        if len(wk) < MIN_WEEKS or not ohlcv or "SPY" not in ohlcv:
            lf.lock_reason = f"{len(wk)}/{MIN_WEEKS} lens-weeks recorded"
            out.append(lf)
            continue
        cal = list(ohlcv["SPY"][0])
        spy_px = {d: float(c) for d, c in zip(ohlcv["SPY"][0], ohlcv["SPY"][1])}
        weekly: List[Tuple[str, float]] = []
        weekly_sec: List[float] = []
        for week in sorted(wk):
            m, ms, complete = _excess_for_week(wk[week], H, ohlcv, cal, spy_px)
            if not complete or m is None:
                continue
            weekly.append((week, m))
            if ms is not None:
                weekly_sec.append(ms)
        lf.weeks_complete = len(weekly)
        if len(weekly) < MIN_WEEKS:
            lf.lock_reason = f"{len(weekly)}/{MIN_WEEKS} lens-weeks with complete {H}-session outcomes"
            out.append(lf)
            continue
        x = np.array([v for _, v in weekly], dtype=float)
        cis = _block_boot_ci(x, int(math.ceil(H / 5)) + 1, (0.95, BONFERRONI_LEVEL))
        ci95, ci_b = cis[0], cis[1]
        mean = float(x.mean())
        lf.locked = False
        lf.mean_excess = mean
        lf.ci95, lf.ci_bonf = ci95, ci_b
        lf.mean_excess_vs_sector = float(np.mean(weekly_sec)) if weekly_sec else None
        lf.weekly = weekly
        lf.proposal = _proposal(label, ci_b, mean)
        out.append(lf)
    return out


def status_line(rep: Sequence[LensForward]) -> str:
    rec = sum(1 for r in rep if r.rows)
    wk = max((r.weeks_recorded for r in rep), default=0)
    unlocked = sum(1 for r in rep if not r.locked)
    return (f"Discover lens record: {rec}/{len(rep)} lenses recorded, up to {wk} weeks · "
            + (f"{unlocked} readable" if unlocked else f"read unlocks {READ_NOT_BEFORE} + {MIN_WEEKS} weeks"))


__all__ = [
    "DEFAULT_DB", "PREREG", "READ_NOT_BEFORE", "MIN_WEEKS", "HORIZONS", "COST", "LensForward",
    "ensure_db", "record_results", "record_hub", "due", "run_snapshot", "lens_weeks",
    "forward_report", "status_line",
]
