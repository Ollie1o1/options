"""Single-ticker 1-screen Dossier builder (`opt check TICKER` & TUI Dossier pane)."""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np

from src.equity_playbooks.indicators import mom_12_1_series, sma, trailing_rv, wilder_rsi
from src.levels import support_resistance_levels
from src.paths import repo_path
from .engine import load_cached_radar
from .scanners import (
    SECTOR_MAP_FALLBACK,
    _load_latest_rev_breadth,
    _load_recent_post_earnings_crashes,
    load_local_ohlcv_safe,
)
from .schema import Opportunity, TickerDossier


def _read_recent_headlines(ticker: str, news_db: str = "data/news_archive.db", limit: int = 3) -> List[Dict[str, Any]]:
    full = repo_path(news_db)
    if not os.path.exists(full):
        return []
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "articles" not in tables:
                return []
            cols = {r["name"] for r in con.execute("PRAGMA table_info(articles)")}
            sym_col = "ticker" if "ticker" in cols else ("symbol" if "symbol" in cols else None)
            title_col = "title" if "title" in cols else ("headline" if "headline" in cols else None)
            date_col = "published_at" if "published_at" in cols else ("date" if "date" in cols else None)
            if not sym_col or not title_col:
                return []
            order = f"ORDER BY {date_col} DESC" if date_col else ""
            rows = con.execute(
                f"SELECT {title_col} AS title{', ' + date_col + ' AS dt' if date_col else ''} "
                f"FROM articles WHERE UPPER({sym_col}) = ? {order} LIMIT ?",
                (ticker.upper(), limit * 2),
            ).fetchall()
            seen = set()
            out: List[Dict[str, Any]] = []
            for r in rows:
                t = str(r["title"] or "").strip()
                if not t or t.lower() in seen:
                    continue
                seen.add(t.lower())
                out.append({"title": t, "date": str(r["dt"]) if "dt" in r.keys() and r["dt"] else ""})
                if len(out) >= limit:
                    break
            return out
        finally:
            con.close()
    except Exception:
        return []


def _next_earnings_info(ticker: str, dolt_db: str = "data/dolt_options.db") -> tuple:
    full = repo_path(dolt_db)
    if not os.path.exists(full):
        return (None, None, "unknown")
    today = dt.date.today().isoformat()
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT MIN(date) FROM earnings_cal WHERE UPPER(symbol) = ? AND date >= ?",
                (ticker.upper(), today),
            ).fetchone()
        finally:
            con.close()
        if row and row[0]:
            dstr = str(row[0])
            days = (dt.date.fromisoformat(dstr) - dt.date.today()).days
            return (days, dstr, "announced")
    except Exception:
        pass
    return (None, None, "unknown")


def build_dossier(
    ticker: str,
    *,
    ohlcv_db: str = "data/equity_ohlcv.db",
    cache_path: str = "data/radar_cache.json",
    fetch_chain: bool = False,
) -> TickerDossier:
    """Consolidate price structure, levels, catalysts, risk flags, and optional RND onto one object."""
    sym = str(ticker).upper().strip()
    as_of = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ohlcv = load_local_ohlcv_safe(ohlcv_db)

    spot = chg1 = chg5 = chg21 = mom12_1 = None
    d20 = d50 = d200 = r2 = r14 = rv20 = None
    supports: List[Dict[str, Any]] = []
    resistances: List[Dict[str, Any]] = []
    closes_60d: List[float] = []

    if sym in ohlcv and len(ohlcv[sym][1]) >= 2:
        _ds, cs, _vs = ohlcv[sym]
        spot = round(float(cs[-1]), 2)
        closes_60d = [round(float(x), 2) for x in cs[-60:]]
        chg1 = round(float(cs[-1] / cs[-2] - 1.0) * 100.0, 2)
        if len(cs) >= 6:
            chg5 = round(float(cs[-1] / cs[-6] - 1.0) * 100.0, 2)
        if len(cs) >= 22:
            chg21 = round(float(cs[-1] / cs[-22] - 1.0) * 100.0, 2)
        if len(cs) >= 253:
            m = float(mom_12_1_series(cs)[-1])
            if np.isfinite(m):
                mom12_1 = round(m * 100.0, 2)
        s20 = float(sma(cs, 20)[-1]) if len(cs) >= 20 else np.nan
        s50 = float(sma(cs, 50)[-1]) if len(cs) >= 50 else np.nan
        s200 = float(sma(cs, 200)[-1]) if len(cs) >= 200 else np.nan
        if np.isfinite(s20) and s20 > 0:
            d20 = round((spot / s20 - 1.0) * 100.0, 2)
        if np.isfinite(s50) and s50 > 0:
            d50 = round((spot / s50 - 1.0) * 100.0, 2)
        if np.isfinite(s200) and s200 > 0:
            d200 = round((spot / s200 - 1.0) * 100.0, 2)
        rv_arr = trailing_rv(cs, 20)
        if np.isfinite(rv_arr[-1]):
            rv20 = round(float(rv_arr[-1]) * 100.0, 1)
        r2_arr = wilder_rsi(cs, 2)
        r14_arr = wilder_rsi(cs, 14)
        if np.isfinite(r2_arr[-1]):
            r2 = round(float(r2_arr[-1]), 1)
        if np.isfinite(r14_arr[-1]):
            r14 = round(float(r14_arr[-1]), 1)

        lv = support_resistance_levels(cs, current=spot)
        supports = (lv.get("supports") or [])[:2]
        resistances = (lv.get("resistances") or [])[:2]

    cdr_ticker = None
    try:
        from src.longterm.cdr import cdr_for

        cdr_ticker = cdr_for(sym)
    except Exception:
        pass

    sector_etf = SECTOR_MAP_FALLBACK.get(sym)
    sector_posture = None
    active_signals: List[Opportunity] = []
    cached = load_cached_radar(cache_path)
    next_macro = "none"
    if cached is not None:
        next_macro = cached.next_macro_event
        active_signals = [o for o in cached.opportunities if o.ticker == sym]
        if sector_etf:
            top_etfs = {c["ticker"] for c in cached.core_allocation if c.get("ticker") != "CASH"}
            if sector_etf in top_etfs:
                sector_posture = "FAVOR"

    days_earn, earn_date, earn_src = _next_earnings_info(sym)
    rev_map = _load_latest_rev_breadth()
    rev_b = round(rev_map[sym], 3) if sym in rev_map else None
    crash_map = _load_recent_post_earnings_crashes(ohlcv)
    post_crash = sym in crash_map
    headlines = _read_recent_headlines(sym)

    risk_flags: List[str] = []
    if days_earn is not None and days_earn <= 7:
        risk_flags.append(f"EARN_IN_{days_earn}D({earn_date})")
    if post_crash:
        risk_flags.append(crash_map[sym])
    if d200 is not None and d200 < 0:
        risk_flags.append("BELOW_200MA")

    # Optional lognormal / RND implied distribution (30-day horizon)
    atm_iv = iv_rank = imp_move = p_up5 = p_dn5 = None
    if rv20 is not None and spot is not None and spot > 0:
        sigma = (rv20 / 100.0)
        t_30 = 30.0 / 365.0
        imp_move = round(sigma * float(np.sqrt(t_30)) * 100.0, 2)
        from scipy.stats import norm

        mu = -0.5 * sigma * sigma * t_30
        sd = max(sigma * float(np.sqrt(t_30)), 1e-6)
        p_up5 = round(float(1.0 - norm.cdf((np.log(1.05) - mu) / sd)), 3)
        p_dn5 = round(float(norm.cdf((np.log(0.95) - mu) / sd)), 3)

    if fetch_chain and spot is not None:
        try:
            from src.probability_lab.rnd import rnd_from_smile
            from src.data_fetcher import fetch_option_chain_for_ticker  # pragma: no cover

            chain_info = fetch_option_chain_for_ticker(sym)
            if chain_info and chain_info.get("atm_iv"):
                atm_iv = round(float(chain_info["atm_iv"]) * 100.0, 1)
                iv_rank = chain_info.get("iv_rank")
        except Exception:
            pass

    return TickerDossier(
        ticker=sym,
        as_of=as_of,
        spot=spot,
        chg_1d_pct=chg1,
        chg_5d_pct=chg5,
        chg_21d_pct=chg21,
        mom_12_1_pct=mom12_1,
        dist_sma20_pct=d20,
        dist_sma50_pct=d50,
        dist_sma200_pct=d200,
        rsi_2=r2,
        rsi_14=r14,
        rv_20_ann=rv20,
        supports=supports,
        resistances=resistances,
        cdr_ticker=cdr_ticker,
        sector_etf=sector_etf,
        sector_posture=sector_posture,
        days_to_earnings=days_earn,
        earnings_date=earn_date,
        earnings_source=earn_src,
        next_macro_event=next_macro,
        rev_breadth_30d=rev_b,
        post_earnings_crash_flag=post_crash,
        headlines=headlines,
        atm_iv=atm_iv,
        iv_rank=iv_rank,
        implied_move_1s_pct=imp_move,
        rnd_prob_up_5pct=p_up5,
        rnd_prob_dn_5pct=p_dn5,
        active_signals=active_signals,
        risk_flags=risk_flags,
        closes_60d=closes_60d,
    )
