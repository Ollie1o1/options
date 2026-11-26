"""Historical companion to the selector study, on DoltHub end-of-day chains (2019-2026).

Rules: docs/PREREG_SELECTOR_BASELINE_DOLT_HISTORY_20261126.md.

Why a companion and not a replica: DoltHub keeps about 27 strikes per expiration and
re-chooses them every day (measured 2026-11-26: on SPY 2023-05-19 puts, 2 to 12 of 27
strikes survived from 03-15 to later dates), and many days are missing. A contract bought
on day 0 usually has no later quote, so the book's daily exit rules cannot be replayed
without substituting nearby strikes, an error that would differ between the arms. Held to
expiration, each arm settles at intrinsic value and needs no later quote at all.

Data path, all cached and resumable:
  1. `fetch`    one chain per symbol per weekly entry day (Wednesday; Thursday, then Tuesday
                when the day is missing upstream), through `src.dolt_options`' cache.
  2. `convert`  each cached chain -> `chain_snapshots` rows in a study-local archive. Dolt has no
                underlying price, so spot is the intercept of call mid - put mid regressed on
                strike for the nearest expiration (C - P = D(F - K)), within 8% of the money.
                Dolt has no open interest; it is stored as NULL.
  3. `settle`   the underlying's close at each entry's expiration, unadjusted from the local
                adjusted-close archive by the ratio (parity spot / adjusted close) observed on the
                nearest fetched day. If that ratio moved more than 5% since entry (a split
                during the hold), the entry's own ratio is used, which keeps pre-split strikes on
                their own basis.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import repo_path

SYMBOLS = ("SPY", "DIA", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
           "AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA")
START, END = "2019-02-11", "2026-06-12"
SPLIT_RATIO_JUMP = 0.05
PARITY_BAND = 0.08


def weekly_days(start: str = START, end: str = END, weekday: int = 2) -> List[str]:
    """The given weekday (0=Mon) of every ISO week in range that is a trading day."""
    from src.data_quality import is_trading_day
    d = dt.date.fromisoformat(start)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    out = []
    while d <= dt.date.fromisoformat(end):
        if is_trading_day(d):
            out.append(d.isoformat())
        d += dt.timedelta(days=7)
    return out


def _fetched(cache_db: str) -> Dict[Tuple[str, str], int]:
    from src import dolt_options as do
    do._ensure_cache(cache_db)
    with sqlite3.connect(cache_db) as con:
        return {(s, d): int(n) for s, d, n in con.execute("SELECT symbol, date, n_rows FROM dolt_fetched")}


DAILY_FROM = "2020-01-06"   # before this week DoltHub is weekly, stamped on Saturday (measured)


def _candidates(wednesday: str) -> List[str]:
    """Entry-day candidates for one week, in order of preference."""
    d = dt.date.fromisoformat(wednesday)
    sat = (d + dt.timedelta(days=3)).isoformat()
    if wednesday < DAILY_FROM:
        return [sat]
    return [wednesday, (d + dt.timedelta(days=1)).isoformat(), (d - dt.timedelta(days=1)).isoformat(), sat]


def fetch(symbols: Sequence[str] = SYMBOLS, *, start: str = START, end: str = END,
          cache_db: Optional[str] = None, workers: int = 6, verbose: bool = True) -> Dict[str, Any]:
    """Fetch one chain per symbol per week: Wed, else Thu, Tue, Sat (Sat only before 2020)."""
    from src import dolt_options as do
    cache = cache_db or repo_path(do.DEFAULT_CACHE)
    weeks = {w: _candidates(w) for w in _all_wednesdays(start, end)}
    summary: Dict[str, Any] = {}
    for rank in range(4):
        have = _fetched(cache)
        hit = {(s, _week(d)) for (s, d), n in have.items() if n > 0}
        todo = [(s, c[rank]) for w, c in weeks.items() if rank < len(c) for s in symbols
                if (s, _week(w)) not in hit]
        if not todo:
            break
        res: Dict[str, Any] = {"pairs": len(todo), "fetched": 0, "failed": 0, "rows": 0, "rate_limited": False}
        for sym in sorted({s for s, _ in todo}):   # per symbol: the backfill fetches a full grid
            days = sorted({d for s, d in todo if s == sym})
            r = do.backfill_parallel([sym], days, db_path=cache, workers=workers, verbose=False)
            for k in ("fetched", "failed", "rows"):
                res[k] += int(r.get(k) or 0)
            if r.get("rate_limited"):
                res["rate_limited"] = True
                break
        summary[f"pass{rank}"] = res
        if verbose:
            print(rank, res, flush=True)
        if res["rate_limited"]:
            break
    return summary


def _all_wednesdays(start: str, end: str) -> List[str]:
    d = dt.date.fromisoformat(start)
    d += dt.timedelta(days=(2 - d.weekday()) % 7)
    out = []
    while d <= dt.date.fromisoformat(end):
        out.append(d.isoformat())
        d += dt.timedelta(days=7)
    return out


def _week(day: str) -> str:
    y, w, _ = dt.date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def _chunks(xs: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield list(xs[i:i + n])


def parity_spot(chain: Sequence[Dict[str, Any]], day: str) -> Optional[float]:
    """Spot implied by put-call parity on the nearest expiration with >= 3 usable strike pairs."""
    exps = sorted({str(c["expiration"]) for c in chain if c.get("expiration") and str(c["expiration"]) > day})
    for exp in exps:
        calls = {c["strike"]: c for c in chain if c["expiration"] == exp and c["type"] == "call"}
        puts = {c["strike"]: c for c in chain if c["expiration"] == exp and c["type"] == "put"}
        pts = []
        for k in sorted(set(calls) & set(puts)):
            c, p = calls[k], puts[k]
            if None in (c.get("bid"), c.get("ask"), p.get("bid"), p.get("ask")):
                continue
            if c["bid"] <= 0 or p["bid"] <= 0 or c["ask"] < c["bid"] or p["ask"] < p["bid"]:
                continue
            pts.append((float(k), 0.5 * (c["bid"] + c["ask"]) - 0.5 * (p["bid"] + p["ask"])))
        if len(pts) < 3:
            continue
        rough = min(pts, key=lambda kv: abs(kv[1]))[0]
        near = [(k, v) for k, v in pts if abs(k / rough - 1) <= PARITY_BAND]
        if len(near) < 3:
            continue
        ks, vs = np.array([k for k, _ in near]), np.array([v for _, v in near])
        slope, icpt = np.polyfit(ks, vs, 1)
        if not (-1.2 < slope < -0.8) or icpt <= 0:
            continue
        return float(-icpt / slope)  # D*F / D = forward; ~spot over a few weeks
    return None


def convert(*, cache_db: Optional[str] = None, out_dir: str, symbols: Sequence[str] = SYMBOLS,
            ohlcv_db: str = "data/equity_ohlcv.db") -> Dict[str, Any]:
    """Write `chain_archive.db` (study-local) and `spot_ratio.json` from the Dolt cache."""
    from src import chain_archive
    from src import dolt_options as do

    cache = cache_db or repo_path(do.DEFAULT_CACHE)
    os.makedirs(out_dir, exist_ok=True)
    chain_db = os.path.join(out_dir, "chain_archive.db")
    if os.path.exists(chain_db):
        os.remove(chain_db)
    chain_archive.ensure_db(chain_db)
    adj = _adj_closes(ohlcv_db, symbols)
    fetched = _fetched(cache)
    ratios: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
    counts = {"chains": 0, "no_rows": 0, "no_spot": 0, "rows": 0}
    with sqlite3.connect(cache) as src, sqlite3.connect(chain_db) as dst:
        for (sym, day), n in sorted(fetched.items()):
            if sym not in symbols:
                continue
            if n == 0:
                counts["no_rows"] += 1
                chain_archive._record_attempt(dst, day, sym, "no_rows", "DoltHub has no chain for this day")
                continue
            cur = src.execute("SELECT expiration, strike, type, bid, ask, iv, delta, gamma, theta, vega, rho "
                              "FROM dolt_chain WHERE symbol=? AND date=?", (sym, day))
            cols = [d[0] for d in cur.description]
            chain = [dict(zip(cols, r)) for r in cur.fetchall()]
            chain = [c for c in chain if c.get("strike") and c.get("expiration") and c.get("type") in ("call", "put")]
            spot = parity_spot(chain, day)
            if spot is None:
                counts["no_spot"] += 1
            else:
                base = _adj_on_or_before(adj.get(sym, {}), day)   # Saturday chains carry Friday's close
                if base:
                    ratios[sym][day] = spot / base
            rows = []
            for c in chain:
                code = f"{sym}{c['expiration']}{c['type'][0].upper()}{c['strike']:g}"
                rows.append((sym, day, code, c["type"], c["strike"], c["expiration"], c.get("bid"), c.get("ask"),
                             None, None, c.get("iv"), c.get("delta"), c.get("gamma"), c.get("theta"),
                             c.get("vega"), c.get("rho"), None, None, None, spot, None, "dolthub_eod"))
            dst.executemany(f"INSERT OR IGNORE INTO chain_snapshots VALUES ({','.join('?' * 22)})", rows)
            chain_archive._record_attempt(dst, day, sym, "archived")
            counts["chains"] += 1
            counts["rows"] += len(rows)
    with open(os.path.join(out_dir, "spot_ratio.json"), "w", encoding="utf-8") as fh:
        json.dump(ratios, fh, sort_keys=True)
    return {"chain_db": chain_db, **counts}


def _adj_closes(ohlcv_db: str, symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
    full = repo_path(ohlcv_db)
    if not os.path.exists(full):
        return out
    q = ",".join("?" * len(symbols))
    with sqlite3.connect(f"file:{full}?mode=ro", uri=True) as con:
        for t, d, c in con.execute(f"SELECT ticker, date, close FROM ohlcv WHERE ticker IN ({q}) AND date>=?",
                                   [*symbols, "2019-01-01"]):
            if c:
                out[t][d] = float(c)
    return out


def _nearest_ratio(ratios: Dict[str, float], day: str, max_days: int = 7) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[int, str, float]] = None
    d0 = dt.date.fromisoformat(day)
    for d, r in ratios.items():
        gap = abs((dt.date.fromisoformat(d) - d0).days)
        if gap <= max_days and (best is None or (gap, d) < (best[0], best[1])):
            best = (gap, d, r)
    return (best[1], best[2]) if best else None


def settle(conn: sqlite3.Connection, *, out_dir: str, ohlcv_db: str = "data/equity_ohlcv.db",
           today: Optional[str] = None) -> Dict[str, int]:
    """Write one `settled` mark per entry whose expiration has passed."""
    with open(os.path.join(out_dir, "spot_ratio.json"), encoding="utf-8") as fh:
        ratios: Dict[str, Dict[str, float]] = json.load(fh)
    today = today or dt.date.today().isoformat()
    entries = [dict(r) for r in conn.execute("SELECT entry_id, symbol, session, expiration FROM entries")]
    adj = _adj_closes(ohlcv_db, sorted({e["symbol"] for e in entries}))
    counts = {"settled": 0, "split_basis": 0, "no_close": 0, "open": 0}
    for e in entries:
        sym, exp = e["symbol"], e["expiration"]
        if exp > today:
            counts["open"] += 1
            continue
        close_adj = _adj_on_or_before(adj.get(sym, {}), exp)
        r_exp = _nearest_ratio(ratios.get(sym, {}), exp)
        r_ent = _nearest_ratio(ratios.get(sym, {}), e["session"], max_days=0)
        close: Optional[float] = None
        src = None
        if close_adj is not None and r_exp is not None and r_ent is not None:
            if abs(r_exp[1] / r_ent[1] - 1) > SPLIT_RATIO_JUMP:
                close, src = close_adj * r_ent[1], f"adj_close x entry ratio ({r_ent[0]}); split during hold"
                counts["split_basis"] += 1
            else:
                close, src = close_adj * r_exp[1], f"adj_close x ratio on {r_exp[0]}"
        if close is None:
            counts["no_close"] += 1
        else:
            counts["settled"] += 1
        conn.execute(
            "INSERT INTO marks (entry_id, mark_session, status, underlying_close, source) VALUES (?,?,?,?,?) "
            "ON CONFLICT(entry_id, mark_session) DO UPDATE SET underlying_close=excluded.underlying_close, "
            "source=excluded.source, status=excluded.status",
            (e["entry_id"], exp, "settled", close, src))
    conn.commit()
    return counts


def _adj_on_or_before(series: Dict[str, float], day: str, max_days: int = 4) -> Optional[float]:
    d = dt.date.fromisoformat(day)
    for k in range(max_days + 1):
        v = series.get((d - dt.timedelta(days=k)).isoformat())
        if v:
            return v
    return None


def ratio_check(out_dir: str) -> Dict[str, Any]:
    """How stable the parity/adjusted ratio is week to week (dividend steps are small, splits big)."""
    with open(os.path.join(out_dir, "spot_ratio.json"), encoding="utf-8") as fh:
        ratios = json.load(fh)
    out = {}
    for s, series in ratios.items():
        days = sorted(series)
        steps = [abs(series[b] / series[a] - 1) for a, b in zip(days, days[1:])]
        if steps:
            out[s] = {"days": len(days), "median_step": statistics.median(steps), "max_step": max(steps),
                      "jumps_over_5pct": sum(x > SPLIT_RATIO_JUMP for x in steps)}
    return out


def settings(base: Dict[str, Any]) -> Dict[str, Any]:
    """The frozen historical-companion settings, derived from the forward study's."""
    st = json.loads(json.dumps(base))
    st["study_id"] = "selector_vs_liquidity_bull_put_dolt_hold_v1"
    st["symbols"] = list(SYMBOLS)
    st["source"] = "dolthub_eod"
    st["capture_window_et"] = None
    st["baseline"]["min_open_interest"] = 0
    st["exits"] = {**st["exits"], "mode": "hold_to_expiry"}
    st["readout"] = {**st["readout"], "read_after": "2026-11-26", "block_weeks": 8}
    return st


def run(*, out_dir: str, study_db: Optional[str] = None, cache_db: Optional[str] = None,
        ohlcv_db: str = "data/equity_ohlcv.db", today: Optional[str] = None,
        reconvert: bool = True) -> Dict[str, Any]:
    """convert -> capture every fetched (day, symbol) -> settle -> readout."""
    from src.selector_study import capture, readout
    from src.selector_study.db import connect, default_settings

    conv = convert(cache_db=cache_db, out_dir=out_dir, ohlcv_db=ohlcv_db) if reconvert else {
        "chain_db": os.path.join(out_dir, "chain_archive.db")}
    chain_db = conv["chain_db"]
    conn = connect(study_db or os.path.join(out_dir, "selector_study_dolt.db"),
                   settings=settings(default_settings()))
    try:
        with sqlite3.connect(chain_db) as con:
            by_day: Dict[str, List[str]] = {}
            for d, s in con.execute("SELECT snap_date, symbol FROM archive_attempts ORDER BY snap_date, symbol"):
                by_day.setdefault(str(d), []).append(str(s))
        caps = [capture.capture_session(conn, d, chain_db=chain_db, symbols=syms) for d, syms in by_day.items()]
        settled = settle(conn, out_dir=out_dir, ohlcv_db=ohlcv_db, today=today)
        report = readout.build(conn, today=today)
    finally:
        conn.close()
    return {"convert": conv, "sessions": len(caps), "settle": settled,
            "ratio_check": ratio_check(out_dir), "readout": report}
