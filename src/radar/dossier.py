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
from src.structure.chain_data import load_chain_for_horizon
from .engine import load_cached_radar
from .scanners import (
    _load_latest_rev_breadth,
    _load_recent_post_earnings_crashes,
    _next_verified_macro_event,
    get_sector_etf,
    load_kalshi_macro_pulse,
    load_local_ohlcv_safe,
    load_open_exposure_map,
)
from .schema import Opportunity, TickerDossier


def _read_recent_headlines(ticker: str, news_db: str = "data/news_archive.db", limit: int = 3) -> List[Dict[str, Any]]:
    """Read recent deduplicated headlines from `data/news_archive.db` (`news_archive` or `articles`)."""
    full = repo_path(news_db)
    if not os.path.exists(full):
        return []
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            tbl = "news_archive" if "news_archive" in tables else ("articles" if "articles" in tables else None)
            if tbl is None:
                return []
            cols = {r["name"] for r in con.execute(f"PRAGMA table_info({tbl})")}
            sym_col = "symbol" if "symbol" in cols else ("ticker" if "ticker" in cols else None)
            title_col = "headline" if "headline" in cols else ("title" if "title" in cols else None)
            date_col = (
                "published"
                if "published" in cols
                else ("published_at" if "published_at" in cols else ("archived_at" if "archived_at" in cols else ("date" if "date" in cols else None)))
            )
            sent_col = "sentiment" if "sentiment" in cols else None
            if not sym_col or not title_col:
                return []
            select_parts = [f"{title_col} AS title"]
            if date_col:
                select_parts.append(f"{date_col} AS dt")
            if sent_col:
                select_parts.append(f"{sent_col} AS sentiment")
            order = f"ORDER BY {date_col} DESC" if date_col else ""
            rows = con.execute(
                f"SELECT {', '.join(select_parts)} FROM {tbl} WHERE {sym_col} = ? {order} LIMIT ?",
                (ticker.upper(), limit * 3),
            ).fetchall()
            seen = set()
            out: List[Dict[str, Any]] = []
            for r in rows:
                t = str(r["title"] or "").strip()
                if not t or t.lower() in seen:
                    continue
                seen.add(t.lower())
                item: Dict[str, Any] = {
                    "title": t,
                    "date": str(r["dt"])[:10] if "dt" in r.keys() and r["dt"] else "",
                }
                if "sentiment" in r.keys() and r["sentiment"] is not None:
                    try:
                        item["sentiment"] = round(float(r["sentiment"]), 2)
                    except (TypeError, ValueError):
                        pass
                out.append(item)
                if len(out) >= limit:
                    break
            return out
        finally:
            con.close()
    except Exception:
        return []


def _read_recent_analyst_actions(
    ticker: str,
    estimates_db: str = "data/estimates_archive.db",
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Read latest analyst grade/price-target actions from `data/estimates_archive.db`."""
    full = repo_path(estimates_db)
    if not os.path.exists(full):
        return []
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "analyst_actions" not in tables:
                return []
            rows = con.execute(
                """
                SELECT grade_date, firm, action, to_grade, current_pt, prior_pt
                FROM analyst_actions
                WHERE symbol = ?
                ORDER BY grade_date DESC
                LIMIT ?
                """,
                (ticker.upper(), limit),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "date": str(r["grade_date"] or "")[:10],
                        "firm": str(r["firm"] or ""),
                        "action": str(r["action"] or ""),
                        "to_grade": str(r["to_grade"] or ""),
                        "current_pt": float(r["current_pt"]) if r["current_pt"] is not None else None,
                        "prior_pt": float(r["prior_pt"]) if r["prior_pt"] is not None else None,
                    }
                )
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
                "SELECT MIN(date) FROM earnings_cal WHERE symbol = ? AND date >= ?",
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
    chain_db: str = "data/chain_archive.db",
    news_db: str = "data/news_archive.db",
    estimates_db: str = "data/estimates_archive.db",
    predmarkets_db: str = "data/predmarkets.db",
    fetch_chain: bool = False,
) -> TickerDossier:
    """Consolidate price structure, levels, catalysts, real chain IV/SVI RND, and risk flags onto one object."""
    sym = str(ticker).upper().strip()
    as_of = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ohlcv = load_local_ohlcv_safe(ohlcv_db, symbols=[sym])

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

        lv = support_resistance_levels(cs.tolist(), current=spot)
        supports = (lv.get("supports") or [])[:2]
        resistances = (lv.get("resistances") or [])[:2]

    cdr_ticker = None
    try:
        from src.longterm.cdr import cdr_for

        cdr_ticker = cdr_for(sym)
    except Exception:
        pass

    sector_etf = get_sector_etf(sym)
    sector_posture = None
    active_signals: List[Opportunity] = []
    cached = load_cached_radar(cache_path)
    next_macro = _next_verified_macro_event()
    kalshi_macro = load_kalshi_macro_pulse(predmarkets_db=predmarkets_db)
    if cached is not None:
        if cached.next_macro_event and cached.next_macro_event != "none":
            next_macro = cached.next_macro_event
        if not kalshi_macro and cached.kalshi_macro:
            kalshi_macro = list(cached.kalshi_macro)
        active_signals = [o for o in cached.opportunities if o.ticker == sym]
        if sector_etf:
            top_etfs = {c["ticker"] for c in cached.core_allocation if c.get("ticker") != "CASH"}
            if sector_etf in top_etfs:
                sector_posture = "FAVOR"

    days_earn, earn_date, earn_src = _next_earnings_info(sym)
    rev_map = _load_latest_rev_breadth(db_path=estimates_db)
    rev_b = round(rev_map[sym], 3) if sym in rev_map else None
    crash_map = _load_recent_post_earnings_crashes(ohlcv)
    post_crash = sym in crash_map
    headlines = _read_recent_headlines(sym, news_db=news_db)
    analyst_actions = _read_recent_analyst_actions(sym, estimates_db=estimates_db)

    # Open portfolio overlap note
    open_position_note: Optional[str] = None
    try:
        exp_map = load_open_exposure_map()
        held_srcs = exp_map["tickers"].get(sym)
        if held_srcs:
            open_position_note = f"Open in {', '.join(held_srcs)}"
        elif sector_etf and exp_map["sector_counts"].get(sector_etf, 0) >= 2:
            cnt = exp_map["sector_counts"][sector_etf]
            open_position_note = f"{cnt} open {sector_etf} sector position(s)"
    except Exception:
        pass

    risk_flags: List[str] = []
    if days_earn is not None and days_earn <= 7:
        risk_flags.append(f"EARN_IN_{days_earn}D({earn_date})")
    if post_crash:
        risk_flags.append(crash_map[sym])
    if d200 is not None and d200 < 0:
        risk_flags.append("BELOW_200MA")

    # Real chain IV + SVI Risk-Neutral Density (archived instant, live when fetch_chain=True)
    atm_iv = iv_rank = imp_move = p_up5 = p_dn5 = None
    iv_vs_rv = skew_25d = None
    chain_source = "rv_proxy"
    chain_as_of: Optional[str] = None
    rnd_source = "lognormal"

    chain_slice = None
    if spot is not None and spot > 0:
        chain_slice = load_chain_for_horizon(
            sym,
            target_dte=30,
            min_dte=14,
            max_dte=65,
            spot=spot,
            chain_db=chain_db,
            allow_live=fetch_chain,
            build_rnd=True,
        )

    if chain_slice is not None and spot is not None and spot > 0:
        chain_source = chain_slice.source
        chain_as_of = chain_slice.snap_date
        if chain_slice.atm_iv is not None:
            atm_iv = round(chain_slice.atm_iv * 100.0, 1)
            if rv20 is not None and rv20 > 0:
                iv_vs_rv = round(atm_iv / rv20, 2)
        skew_25d = chain_slice.skew_25d_vp
        if chain_slice.rnd_density is not None:
            dens = chain_slice.rnd_density
            rnd_source = chain_slice.rnd_source
            p_up5 = round(dens.prob_above(spot * 1.05), 3)
            p_dn5 = round(dens.prob_below(spot * 0.95), 3)
            t_slice = max(chain_slice.dte, 1) / 365.0
            if chain_slice.atm_iv is not None:
                imp_move = round(chain_slice.atm_iv * float(np.sqrt(t_slice)) * 100.0, 2)

    # Clean lognormal RV20 fallback whenever chain RND isn't available
    if (p_up5 is None or p_dn5 is None) and rv20 is not None and spot is not None and spot > 0:
        sigma = (atm_iv / 100.0) if atm_iv is not None else (rv20 / 100.0)
        t_30 = 30.0 / 365.0
        if imp_move is None:
            imp_move = round(sigma * float(np.sqrt(t_30)) * 100.0, 2)
        from scipy.stats import norm

        mu = -0.5 * sigma * sigma * t_30
        sd = max(sigma * float(np.sqrt(t_30)), 1e-6)
        p_up5 = round(float(1.0 - norm.cdf((np.log(1.05) - mu) / sd)), 3)
        p_dn5 = round(float(norm.cdf((np.log(0.95) - mu) / sd)), 3)

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
        chain_source=chain_source,
        chain_as_of=chain_as_of,
        rnd_source=rnd_source,
        iv_vs_rv_ratio=iv_vs_rv,
        skew_25d_vp=skew_25d,
        kalshi_macro=kalshi_macro,
        analyst_actions=analyst_actions,
        open_position_note=open_position_note,
    )
