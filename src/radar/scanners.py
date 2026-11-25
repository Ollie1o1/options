"""Failure-isolated scanner adapters feeding the Unified Radar Engine.

All default adapters operate on local SQLite/JSON caches (`data/equity_ohlcv.db`,
`data/chain_archive.db`, `data/candidates.db`, `data/dolt_options.db`,
`data/estimates_archive.db`, `data/longterm.db`, `data/insider_cache.json`,
`data/predmarkets.db`, `longterm_plan.json`) so interactive startup completes in
<1s without network hangs. Live option scans remain opt-in (`allow_network=True`).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.equity_playbooks.backtest import load_equity_ohlcv
from src.equity_playbooks.indicators import (
    mom_12_1_series,
    sma,
    trailing_median_dollar_vol,
    trailing_rv,
    wilder_rsi,
)
from src.paths import repo_path
from src.structure.chain_data import load_archived_chain_slice
from src.structure.execution_plan import build_walk_away_ladder
from .playbooks import PlaybookMeta
from .schema import Opportunity

OUTLOOK_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
    "XLC", "SMH", "GLD", "TLT", "IWM", "QQQ",
]

LIQUID_ETF_UNIVERSE = frozenset(OUTLOOK_ETFS) | frozenset(
    {"SPY", "DIA", "EEM", "EFA", "HYG", "KRE", "SLV", "GDX", "XBI"}
)

SECTOR_MAP_FALLBACK: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH",
    "INTC": "SMH", "QCOM": "SMH", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "PLTR": "XLK", "SMCI": "XLK", "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "RIVN": "XLY", "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "COIN": "XLF",
    "XOM": "XLE", "CVX": "XLE", "UNH": "XLV", "LLY": "XLV", "JNJ": "XLV",
    "PG": "XLP", "KO": "XLP", "WMT": "XLP", "COST": "XLP", "CAT": "XLI", "BA": "XLI",
}

_FULL_SECTOR_MAP: Optional[Dict[str, str]] = None

# GICS sector (as its SPDR ETF) for liquid names added by the 2026-11-24 universe
# expansion (`src/equity_playbooks/universe.py`) that `data_fetching.SECTOR_MAP`
# never covered. Without these, 71 of 371 archived names had no sector, so
# sector breadth and the sector posture badge silently skipped them.
# Semiconductors map to SMH, matching the existing NVDA/AVGO/AMD/INTC/QCOM override.
SECTOR_MAP_SUPPLEMENT: Dict[str, str] = {
    "ADI": "SMH", "ARM": "SMH", "ASML": "SMH", "MCHP": "SMH", "NXPI": "SMH", "RMBS": "SMH", "TSM": "SMH",
    "ANET": "XLK", "APP": "XLK", "CDNS": "XLK", "HUBS": "XLK", "MSTR": "XLK", "NET": "XLK", "RGTI": "XLK",
    "SHOP": "XLK", "SNPS": "XLK", "TEAM": "XLK", "WDAY": "XLK",
    "AIG": "XLF", "AON": "XLF", "APO": "XLF", "BRK-B": "XLF", "BX": "XLF", "CB": "XLF", "HOOD": "XLF",
    "KKR": "XLF", "MET": "XLF", "PGR": "XLF", "PRU": "XLF", "SOFI": "XLF", "TRV": "XLF",
    "CARR": "XLI", "GEV": "XLI", "GWW": "XLI", "ITW": "XLI", "PWR": "XLI", "TDG": "XLI", "TT": "XLI", "VRT": "XLI",
    "AZO": "XLY", "CMG": "XLY", "CVNA": "XLY", "DECK": "XLY", "GME": "XLY", "HLT": "XLY", "LULU": "XLY",
    "MAR": "XLY", "ORLY": "XLY",
    "DG": "XLP", "DLTR": "XLP", "KMB": "XLP", "KR": "XLP", "MNST": "XLP",
    "COR": "XLV", "DHR": "XLV", "HCA": "XLV", "MCK": "XLV", "NVO": "XLV", "ZTS": "XLV",
    "KMI": "XLE", "OKE": "XLE", "TRGP": "XLE", "WMB": "XLE",
    "CEG": "XLU", "SRE": "XLU", "VST": "XLU",
    "MLM": "XLB", "VMC": "XLB",
    "SPOT": "XLC",
    "VICI": "XLRE",
}


def get_sector_map() -> Dict[str, str]:
    """Return the full sector ETF map (data_fetching.SECTOR_MAP + expansion supplement + SMH overrides)."""
    global _FULL_SECTOR_MAP
    if _FULL_SECTOR_MAP is None:
        merged: Dict[str, str] = {}
        try:
            from src.data_fetching import SECTOR_MAP as _DF_SECTOR_MAP

            merged.update({str(k).upper(): str(v).upper() for k, v in _DF_SECTOR_MAP.items()})
        except Exception:
            pass
        for k, v in SECTOR_MAP_SUPPLEMENT.items():
            merged.setdefault(k, v)
        # Preserve SMH overrides for major semis so Outlook SMH posture flows cleanly
        merged.update(SECTOR_MAP_FALLBACK)
        _FULL_SECTOR_MAP = merged
    return _FULL_SECTOR_MAP


def get_sector_etf(symbol: str) -> Optional[str]:
    return get_sector_map().get(str(symbol).upper().strip())


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_REV_BREADTH_CACHE: Dict[Tuple[str, float], Dict[str, float]] = {}


def _load_latest_rev_breadth(db_path: str = "data/estimates_archive.db") -> Dict[str, float]:
    full = repo_path(db_path)
    if not os.path.exists(full):
        return {}
    try:
        mtime = os.path.getmtime(full)
        ck = (full, mtime)
        if ck in _REV_BREADTH_CACHE:
            return _REV_BREADTH_CACHE[ck]
        from scripts.estimate_revisions_test import compute_rev_breadth_30d

        by_date = compute_rev_breadth_30d(full)
        if not by_date:
            _REV_BREADTH_CACHE[ck] = {}
            return {}
        latest = max(by_date.keys())
        res = by_date[latest]
        _REV_BREADTH_CACHE[ck] = res
        return res
    except Exception:
        return {}


def _load_recent_post_earnings_crashes(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    dolt_db: str = "data/dolt_options.db",
    lookback_sessions: int = 20,
) -> Dict[str, str]:
    """Detect tickers that suffered a <= -5% / >= 2x vol earnings breakdown in the last 20 sessions.

    Backed by Phase 1 Playbook 4 diagnostic (`pead_neg_20d`: -1.16% excess vs SPY,
    95% CI [-2.19%, -0.19%]). Surfaced as a risk/AVOID flag rather than a strategy.
    """
    full = repo_path(dolt_db)
    if not os.path.exists(full):
        return {}
    flags: Dict[str, str] = {}
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT symbol, date, whn FROM earnings_cal WHERE date >= date('now', '-45 days')"
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return {}

    for sym, ev_date, whn in rows:
        s = str(sym).upper()
        if s not in ohlcv:
            continue
        ds, cs, vs = ohlcv[s]
        if len(ds) < 30:
            continue
        amc = str(whn or "").lower().startswith("after")
        idx = None
        for i in range(max(21, len(ds) - lookback_sessions - 2), len(ds)):
            if (amc and ds[i] > ev_date) or (not amc and ds[i] >= ev_date):
                idx = i
                break
        if idx is None or idx <= 20 or (len(ds) - 1 - idx) > lookback_sessions:
            continue
        ret = float(cs[idx] / cs[idx - 1] - 1.0)
        med_v = float(np.median(vs[idx - 20 : idx]))
        if ret <= -0.05 and med_v > 0 and float(vs[idx]) >= 2.0 * med_v:
            flags[s] = f"POST_EARN_CRASH({ret*100:.1f}%)"
    return flags


def _load_upcoming_earnings_map(
    dolt_db: str = "data/dolt_options.db",
    horizon_days: int = 10,
) -> Dict[str, int]:
    """Map ticker -> calendar days to next known (or cadence-projected) earnings within horizon_days."""
    full = repo_path(dolt_db)
    if not os.path.exists(full):
        return {}
    today = dt.date.today()
    out: Dict[str, int] = {}
    history_by_sym: Dict[str, List[str]] = {}
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT symbol, MIN(date) FROM earnings_cal "
                "WHERE date >= ? AND date <= ? GROUP BY symbol",
                (today.isoformat(), (today + dt.timedelta(days=horizon_days)).isoformat()),
            ).fetchall()
            hist_rows = con.execute(
                "SELECT symbol, date FROM earnings_cal WHERE date IS NOT NULL ORDER BY symbol, date"
            ).fetchall()
        finally:
            con.close()
        for sym, dstr in rows:
            days = (dt.date.fromisoformat(dstr) - today).days
            out[str(sym).upper()] = days
        for sym, dstr in hist_rows:
            history_by_sym.setdefault(str(sym).upper(), []).append(str(dstr)[:10])
    except Exception:
        return {}

    # Fallback: for symbols with a regular historical earnings cadence (>=8 quarters)
    # and no announced future date in cache, project the next report via earnings_gate.
    try:
        from src.earnings_gate import project_next_earnings

        for sym, dates in history_by_sym.items():
            if sym in out:
                continue
            if any(d >= today.isoformat() for d in dates):
                continue
            proj = project_next_earnings(dates, today=today)
            if proj:
                days = (dt.date.fromisoformat(proj) - today).days
                if 0 <= days <= horizon_days:
                    out[sym] = days
    except Exception:
        pass
    return out


def load_kalshi_macro_pulse(
    predmarkets_db: str = "data/predmarkets.db",
    *,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    """Read latest Kalshi macro probabilities (FOMC, CPI, Recession) from `data/predmarkets.db`."""
    full = repo_path(predmarkets_db)
    if not os.path.exists(full):
        return []
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "pm_quotes" not in tables:
                return []
            latest_row = con.execute("SELECT MAX(archived_at) FROM pm_quotes").fetchone()
            if not latest_row or not latest_row[0]:
                return []
            latest_ts = str(latest_row[0])
            rows = con.execute(
                """
                SELECT series, ticker, title, yes_bid, yes_ask, last, open_interest, close_time
                FROM pm_quotes
                WHERE archived_at = ?
                ORDER BY open_interest DESC
                """,
                (latest_ts,),
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return []

    def _prob_of(r: sqlite3.Row) -> Optional[float]:
        b, a, last = r["yes_bid"], r["yes_ask"], r["last"]
        if b is not None and a is not None and a >= b:
            return float(0.5 * (float(b) + float(a)))
        if last is not None:
            return float(last)
        return None

    # Group by series and pick the most informative/highest-probability upcoming contract
    by_series: Dict[str, List[Dict[str, Any]]] = {}
    today_iso = dt.date.today().isoformat()
    for r in rows:
        close_t = str(r["close_time"] or "")
        if close_t and close_t[:10] < today_iso:
            continue
        p = _prob_of(r)
        if p is None or not (0.0 <= p <= 1.0):
            continue
        ser = str(r["series"] or "").upper()
        by_series.setdefault(ser, []).append(
            {
                "series": ser,
                "ticker": str(r["ticker"] or ""),
                "title": str(r["title"] or ""),
                "prob": p,
                "open_interest": float(r["open_interest"] or 0.0),
                "close_date": close_t[:10] if close_t else "",
                "archived_at": latest_ts,
            }
        )

    out: List[Dict[str, Any]] = []
    # 1. Next FOMC decision: pick the modal (highest-prob) outcome for the earliest upcoming meeting
    fomc_rows = by_series.get("KXFEDDECISION") or []
    if fomc_rows:
        earliest_close = min(x["close_date"] for x in fomc_rows if x["close_date"])
        near_fomc = [x for x in fomc_rows if x["close_date"] == earliest_close] or fomc_rows
        best = max(near_fomc, key=lambda x: (x["prob"], x["open_interest"]))
        code = best["ticker"].split("-")[-1]
        code_map = {
            "H0": "Hold 0bp",
            "H25": "Hike +25bp",
            "H26": "Hike >25bp",
            "C25": "Cut -25bp",
            "C26": "Cut >25bp",
        }
        action = code_map.get(code, best["title"][:32])
        out.append(
            {
                "series": "KXFEDDECISION",
                "label": f"FOMC {earliest_close[5:]} {action}",
                "prob_pct": round(best["prob"] * 100.0, 0),
                "close_date": earliest_close,
                "archived_at": latest_ts,
            }
        )

    # 2. Next CPI threshold with highest OI
    cpi_rows = by_series.get("KXCPI") or []
    if cpi_rows:
        # Pick contract nearest 50% (most informative) among top-OI contracts
        top_cpi = sorted(cpi_rows, key=lambda x: -x["open_interest"])[:6]
        best = min(top_cpi, key=lambda x: abs(x["prob"] - 0.5))
        thresh = best["ticker"].split("-")[-1].lstrip("T")
        out.append(
            {
                "series": "KXCPI",
                "label": f"CPI >{thresh}% ({best['close_date'][5:]})",
                "prob_pct": round(best["prob"] * 100.0, 0),
                "close_date": best["close_date"],
                "archived_at": latest_ts,
            }
        )

    # 3. Recession odds (prefer active forward year with meaningful probability > 3%)
    rec_rows = by_series.get("KXRECSSNBER") or []
    if rec_rows:
        # Prefer the next-year or highest-prob meaningful recession contract
        meaningful = [x for x in rec_rows if x["prob"] >= 0.03] or rec_rows
        best = max(meaningful, key=lambda x: (x["open_interest"] if x["prob"] >= 0.05 else 0.0, x["prob"]))
        yr = best["ticker"].split("-")[-1]
        out.append(
            {
                "series": "KXRECSSNBER",
                "label": f"Recession 20{yr}" if len(yr) == 2 and yr.isdigit() else f"Recession {yr}",
                "prob_pct": round(best["prob"] * 100.0, 0),
                "close_date": best["close_date"],
                "archived_at": latest_ts,
            }
        )

    return out[:max_items]


def load_open_exposure_map(
    *,
    own_picks_db: str = "data/own_picks.db",
    paper_db: str = "paper_trades.db",
    longterm_db: str = "data/longterm.db",
) -> Dict[str, Any]:
    """Collect currently open stock/ETF/option tickers and sector counts for overlap warnings."""
    held_tickers: Dict[str, List[str]] = {}
    sector_counts: Dict[str, int] = {}

    def _add(sym: str, src_label: str) -> None:
        s = str(sym).upper().strip()
        if not s:
            return
        held_tickers.setdefault(s, [])
        if src_label not in held_tickers[s]:
            held_tickers[s].append(src_label)
        sec = get_sector_etf(s)
        if sec:
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # 1. Manual stock/ETF picks (data/own_picks.db)
    full_own = repo_path(own_picks_db)
    if os.path.exists(full_own):
        try:
            con = sqlite3.connect(f"file:{full_own}?mode=ro", uri=True)
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "equity_picks" in tables:
                    for (tk,) in con.execute("SELECT ticker FROM equity_picks WHERE status = 'OPEN'").fetchall():
                        _add(tk, "own_pick")
            finally:
                con.close()
        except Exception:
            pass

    # 2. Long-term TFSA fills (data/longterm.db)
    full_lt = repo_path(longterm_db)
    if os.path.exists(full_lt):
        try:
            from src.longterm.fills import book as lt_book

            for tk, slot in lt_book(full_lt).items():
                if float(slot.get("shares") or 0.0) > 0:
                    _add(tk, "longterm")
        except Exception:
            pass

    # 3. Open paper options (paper_trades.db)
    full_pm = repo_path(paper_db)
    if os.path.exists(full_pm) and os.path.getsize(full_pm) > 0:
        try:
            con = sqlite3.connect(f"file:{full_pm}?mode=ro", uri=True)
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "trades" in tables:
                    for (tk,) in con.execute(
                        "SELECT DISTINCT ticker FROM trades WHERE status = 'OPEN'"
                    ).fetchall():
                        _add(tk, "paper_opt")
            finally:
                con.close()
        except Exception:
            pass

    return {"tickers": held_tickers, "sector_counts": sector_counts}


def scan_vol_managed_outlook(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    registry: Dict[str, PlaybookMeta],
    *,
    sigma_target: float = 0.14,
) -> Tuple[List[Dict[str, Any]], List[Opportunity], Dict[str, str]]:
    """Validated Playbook `vol_managed_outlook` (VM_OUTLOOK_TOP3): target allocation + posture map.

    The weights come from `src.outlook.vm_rotation`, the one implementation the
    backtest parity test pins to `scripts/backtest_vol_managed_outlook.py`:
    top-3 Outlook ETFs with a positive composite, 1/3 each, scaled by
    clip(0.14 / RV20(selected basket), 0.25, 1.00). Posture: FAVOR = held by
    the sleeve; AVOID = a regime-gated BEARISH outlook call; else NEUTRAL.
    """
    from src.outlook.vm_rotation import TOP_K, align_closes, latest_complete_index, target_weights

    pb = registry.get("vol_managed_outlook")
    as_of = _now_utc()
    if "SPY" not in ohlcv:
        return [], [], {}
    dates, cols = align_closes(ohlcv, OUTLOOK_ETFS)
    t = latest_complete_index(cols, OUTLOOK_ETFS)
    if t is None:
        return [], [], {}
    tgt = target_weights(dates, cols, t, universe=OUTLOOK_ETFS, target_vol=sigma_target)
    if tgt is None:
        return [], [], {}

    spy_rv_arr = trailing_rv(cols["SPY"][: t + 1], window=20)
    spy_rv20 = float(spy_rv_arr[-1]) if len(spy_rv_arr) and np.isfinite(spy_rv_arr[-1]) else None
    n_ranked = len(tgt.ranked)

    posture_map: Dict[str, str] = {}
    for r in tgt.ranked:
        sym = str(r["ticker"])
        if sym in tgt.weights:
            posture_map[sym] = "FAVOR"
        elif r.get("direction") == "BEARISH":
            posture_map[sym] = "AVOID"
        else:
            posture_map[sym] = "NEUTRAL"

    def _meta(sym: str) -> Dict[str, float]:
        cs = cols[sym][: t + 1]
        return {
            "spot": float(cs[-1]),
            "chg_1d": float(cs[-1] / cs[-2] - 1.0) * 100.0,
            "chg_5d": float(cs[-1] / cs[-6] - 1.0) * 100.0,
            "mom_12_1": float(cs[-22] / cs[-253] - 1.0) * 100.0,
        }

    core_alloc: List[Dict[str, Any]] = []
    opps: List[Opportunity] = []
    rank_of = {str(r["ticker"]): i for i, r in enumerate(tgt.ranked, 1)}
    for sym, w in sorted(tgt.weights.items(), key=lambda kv: rank_of[kv[0]]):
        rank = rank_of[sym]
        comp = float(next(r["score"] for r in tgt.ranked if r["ticker"] == sym))
        m = _meta(sym)
        core_alloc.append(
            {
                "ticker": sym,
                "rank": rank,
                "target_weight": round(w, 4),
                "vol_scale": round(tgt.scale, 3),
                "basket_rv20": None if tgt.basket_rv20 is None else round(tgt.basket_rv20, 4),
                "spy_rv20": None if spy_rv20 is None else round(spy_rv20, 4),
                "composite": round(comp, 3),
                "spot": round(m["spot"], 2),
                "mom_12_1_pct": round(m["mom_12_1"], 1),
                "as_of": tgt.as_of,
            }
        )
        if pb and pb.is_default_radar_eligible:
            opps.append(
                Opportunity(
                    ticker=sym,
                    asset_class="etf",
                    playbook_id=pb.id,
                    playbook_label=pb.label,
                    evidence_tier=pb.evidence_tier,
                    direction="LONG",
                    spot=round(m["spot"], 2),
                    chg_1d_pct=round(m["chg_1d"], 2),
                    chg_5d_pct=round(m["chg_5d"], 2),
                    horizon_label=pb.horizon_label,
                    setup_summary=(
                        f"Rank #{rank}/{n_ranked} · Weight {w*100:.0f}% (scale {tgt.scale*100:.0f}% "
                        f"of 1/{TOP_K}) · 12-1m {m['mom_12_1']:+.1f}%"
                    ),
                    badges=(f"FAVOR:{sym}", f"WEIGHT:{w*100:.0f}%", "TFSA_OK"),
                    risk_flags=("VOL_SCALED_DOWN",) if tgt.scale < 0.75 else (),
                    friction_label="0.06% etf",
                    friction_value=0.0006,
                    confluence_count=1,
                    as_of=as_of,
                    extra={
                        "rank": rank, "target_weight": w, "vol_scale": tgt.scale,
                        "basket_rv20": tgt.basket_rv20, "prices_as_of": tgt.as_of,
                    },
                )
            )
    cash_w = round(tgt.cash, 4)
    if cash_w > 0.005:
        core_alloc.append(
            {
                "ticker": "CASH",
                "rank": 0,
                "target_weight": cash_w,
                "vol_scale": round(tgt.scale, 3),
                "basket_rv20": None if tgt.basket_rv20 is None else round(tgt.basket_rv20, 4),
                "spy_rv20": None if spy_rv20 is None else round(spy_rv20, 4),
                "composite": 0.0,
                "spot": 1.0,
                "mom_12_1_pct": 0.0,
                "as_of": tgt.as_of,
            }
        )
    return core_alloc, opps, posture_map


def scan_equity_pullbacks(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    registry: Dict[str, PlaybookMeta],
    *,
    posture_map: Optional[Dict[str, str]] = None,
    earnings_map: Optional[Dict[str, int]] = None,
    crash_flags: Optional[Dict[str, str]] = None,
    rev_breadth: Optional[Dict[str, float]] = None,
    regime_posture: str = "NEUTRAL",
    include_near_setups: bool = True,
) -> List[Opportunity]:
    """Scan local OHLCV universe for Phase 1 Playbook 1 (Uptrend Pullback) signals."""
    pb = registry.get("equity_pullback")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    as_of = _now_utc()
    posture_map = posture_map or {}
    earnings_map = earnings_map or {}
    crash_flags = crash_flags or {}
    rev_breadth = rev_breadth or {}
    sector_map = get_sector_map()
    opps: List[Opportunity] = []

    for sym, (ds, cs, vs) in sorted(ohlcv.items()):
        if len(cs) < 253:
            continue
        spot = float(cs[-1])
        if spot < 10.0:
            continue
        s200 = float(sma(cs, 200)[-1])
        s5 = float(sma(cs, 5)[-1])
        m12_1 = float(mom_12_1_series(cs)[-1])
        rsi2 = float(wilder_rsi(cs, 2)[-1])
        dv20 = float(trailing_median_dollar_vol(cs, vs, 20)[-1])
        if not (np.isfinite(s200) and np.isfinite(m12_1) and np.isfinite(rsi2)):
            continue
        ret3 = float(cs[-1] / cs[-4] - 1.0)
        ret1 = float(cs[-1] / cs[-2] - 1.0)
        ret5 = float(cs[-1] / cs[-6] - 1.0)

        strict_hit = (
            spot > s200
            and m12_1 > 0.0
            and rsi2 < 10.0
            and ret3 <= -0.03
            and dv20 >= 50_000_000.0
        )
        near_hit = (
            include_near_setups
            and not strict_hit
            and spot > s200
            and m12_1 > 0.05
            and rsi2 < 20.0
            and ret3 <= -0.015
            and dv20 >= 50_000_000.0
        )
        if not (strict_hit or near_hit):
            continue

        badges: List[str] = [">200MA", f"RSI2={rsi2:.1f}"]
        if strict_hit:
            badges.append("STRICT_PULLBACK")
        else:
            badges.append("NEAR_PULLBACK")
        sec = sector_map.get(sym)
        if sec and posture_map.get(sec) == "FAVOR":
            badges.append(f"FAVOR:{sec}")
        if sym in rev_breadth and rev_breadth[sym] >= 0.35:
            badges.append(f"REV+{rev_breadth[sym]*100:.0f}%")

        risk_flags: List[str] = []
        if regime_posture == "RISK_OFF":
            risk_flags.append("BEAR_REGIME")
        if sym in earnings_map and earnings_map[sym] <= 7:
            risk_flags.append(f"EARN:{earnings_map[sym]}d")
        if sym in crash_flags:
            risk_flags.append(crash_flags[sym])
        if sec and posture_map.get(sec) == "AVOID":
            risk_flags.append(f"AVOID_SEC:{sec}")

        dist200 = (spot / s200 - 1.0) * 100.0
        opps.append(
            Opportunity(
                ticker=sym,
                asset_class="etf" if sym in LIQUID_ETF_UNIVERSE else "stock",
                playbook_id=pb.id,
                playbook_label=pb.label,
                evidence_tier=pb.evidence_tier if strict_hit else "CONTEXT",
                direction="LONG",
                spot=round(spot, 2),
                chg_1d_pct=round(ret1 * 100.0, 2),
                chg_5d_pct=round(ret5 * 100.0, 2),
                horizon_label=pb.horizon_label,
                setup_summary=(
                    f"RSI2={rsi2:.1f} · 3d {ret3*100:+.1f}% · {dist200:+.1f}% >200MA · "
                    f"Exit >SMA5 (${s5:.2f}) or 10d"
                ),
                badges=tuple(badges),
                risk_flags=tuple(risk_flags),
                friction_label="0.10% stk",
                friction_value=0.0010,
                confluence_count=1,
                suggested_stop=round(spot * 0.94, 2),
                suggested_target=round(max(s5, spot * 1.02), 2),
                as_of=as_of,
                extra={"strict": strict_hit, "rsi2": round(rsi2, 2), "sma5": round(s5, 2)},
            )
        )
    return opps


def scan_holdings_buy_zones(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    registry: Dict[str, PlaybookMeta],
    *,
    plan_path: Optional[str] = None,
    fills_db: str = "data/longterm.db",
) -> List[Opportunity]:
    """Emit CONTEXT alerts for long-term plan holdings in or near unfilled buy-zone tranches."""
    pb = registry.get("holdings_buy_zone")
    if pb is None or not pb.is_default_radar_eligible:
        return []

    # Default path is repo-root `longterm_plan.json` (src/longterm/plan.py::DEFAULT_PATH)
    candidates_paths = (
        [plan_path]
        if plan_path is not None
        else ["longterm_plan.json", "configs/longterm_plan.json"]
    )
    resolved_plan: Optional[str] = None
    for p in candidates_paths:
        full = repo_path(p)
        if os.path.exists(full):
            resolved_plan = full
            break
    if resolved_plan is None:
        return []

    as_of = _now_utc()
    opps: List[Opportunity] = []
    try:
        from src.longterm.plan import load_plan

        plan = load_plan(resolved_plan)
    except Exception:
        return []

    filled_by_ticker: Dict[str, set] = {}
    full_fills = repo_path(fills_db)
    if os.path.exists(full_fills):
        try:
            from src.longterm.fills import filled_levels

            for n in plan.names:
                filled_by_ticker[n.ticker.upper()] = filled_levels(n.ticker.upper(), db_path=full_fills)
        except Exception:
            pass

    for name in plan.names:
        sym = name.ticker.upper()
        if sym not in ohlcv or not name.tranches:
            continue
        _ds, cs, _vs = ohlcv[sym]
        if len(cs) < 6:
            continue
        spot = float(cs[-1])
        ret1 = float(cs[-1] / cs[-2] - 1.0) * 100.0
        ret5 = float(cs[-1] / cs[-6] - 1.0) * 100.0
        already = filled_by_ticker.get(sym, set())
        open_levels = sorted(
            t.level for t in name.tranches if not any(math.isclose(t.level, fl, rel_tol=1e-4) for fl in already)
        )
        if not open_levels:
            continue
        top_level = open_levels[-1]
        dist_pct = (spot / top_level - 1.0) * 100.0
        if dist_pct <= 5.0:
            state = "IN_ZONE" if spot <= top_level else "NEAR_ZONE"
            opps.append(
                Opportunity(
                    ticker=sym,
                    asset_class="stock",
                    playbook_id=pb.id,
                    playbook_label=pb.label,
                    evidence_tier=pb.evidence_tier,
                    direction="LONG",
                    spot=round(spot, 2),
                    chg_1d_pct=round(ret1, 2),
                    chg_5d_pct=round(ret5, 2),
                    horizon_label=pb.horizon_label,
                    setup_summary=f"{state} · Tranche @ ${top_level:g} ({dist_pct:+.1f}% away)",
                    badges=("HOLDING", state),
                    risk_flags=(),
                    friction_label="0.05% stk",
                    friction_value=0.0005,
                    confluence_count=1,
                    suggested_target=round(top_level * 1.15, 2),
                    as_of=as_of,
                    extra={"tranche_level": top_level, "state": state},
                )
            )
    return opps


def _parse_strikes_from_contract_key(contract_key: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse short/long strikes from `SYM|EXP|Bear Call|124/125` style keys."""
    if not contract_key:
        return (None, None)
    parts = str(contract_key).split("|")
    if len(parts) < 4:
        return (None, None)
    strike_part = parts[-1]
    if "/" not in strike_part:
        return (None, None)
    a, b = strike_part.split("/", 1)
    try:
        return (float(a), float(b))
    except ValueError:
        return (None, None)


def scan_cached_etf_bear_calls(
    registry: Dict[str, PlaybookMeta],
    *,
    ohlcv: Optional[Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]] = None,
    candidates_db: str = "data/candidates.db",
    max_age_days: int = 7,
) -> List[Opportunity]:
    """Read recent gate-passed ETF Bear Calls (LEAD playbook, B1-F pending 2027-01-20) from candidates.db."""
    pb = registry.get("etf_bear_call")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    full = repo_path(candidates_db)
    if not os.path.exists(full):
        return []
    as_of = _now_utc()
    today = dt.date.today()
    cutoff = (today - dt.timedelta(days=max_age_days)).isoformat()
    opps: List[Opportunity] = []
    rows: List[Dict[str, Any]] = []

    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            cols = {r["name"] for r in con.execute("PRAGMA table_info(candidates)")}
            if {"symbol", "strategy_name", "ts"}.issubset(cols):
                # Real production candidates.db schema
                etf_placeholders = ",".join("?" for _ in LIQUID_ETF_UNIVERSE)
                sql = f"""
                    SELECT symbol, ts, expiration, strike, contract_key,
                           premium, ev_net, ev_cost, round_trip_pct, gate_passed, refused_by
                    FROM candidates
                    WHERE strategy_name = 'Bear Call'
                      AND UPPER(symbol) IN ({etf_placeholders})
                      AND ts >= ?
                      AND (gate_passed = 1 OR refused_by IS NULL)
                    ORDER BY ts DESC
                    LIMIT 120
                """
                params = list(sorted(LIQUID_ETF_UNIVERSE)) + [cutoff]
                for r in con.execute(sql, params).fetchall():
                    rows.append(dict(r))
            elif {"symbol", "strategy", "gate_passed", "scan_date"}.issubset(cols):
                # Legacy/test simplified schema
                for r in con.execute(
                    """
                    SELECT symbol, scan_date AS ts, expiration, strike, long_strike,
                           entry_price AS premium, underlying_price, dte, friction_to_credit AS round_trip_pct
                    FROM candidates
                    WHERE strategy = 'Bear Call'
                      AND gate_passed = 1
                      AND scan_date >= ?
                    ORDER BY scan_date DESC
                    LIMIT 60
                    """,
                    (cutoff,),
                ).fetchall():
                    rows.append(dict(r))
            else:
                return []
        finally:
            con.close()
    except Exception:
        return []

    seen = set()
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if sym not in LIQUID_ETF_UNIVERSE or sym in seen:
            continue
        k_s = r.get("strike")
        k_l = r.get("long_strike")
        if k_s is None or k_l is None:
            ps, pl = _parse_strikes_from_contract_key(r.get("contract_key"))
            k_s = ps if k_s is None else k_s
            k_l = pl if k_l is None else k_l

        exp = str(r.get("expiration") or "")
        dte = r.get("dte")
        if dte is None and exp:
            try:
                dte = (dt.date.fromisoformat(exp[:10]) - today).days
            except ValueError:
                dte = 30
        dte_int = int(dte or 30)
        if dte_int < 5:
            continue

        spot = float(r.get("underlying_price") or 0.0)
        if spot <= 0 and ohlcv and sym in ohlcv and len(ohlcv[sym][1]):
            spot = float(ohlcv[sym][1][-1])
        if spot <= 0 and k_s is not None:
            spot = float(k_s)

        prem = r.get("premium")
        fric = r.get("round_trip_pct")
        if fric is None:
            ev_cost = r.get("ev_cost")
            if ev_cost is not None and prem and float(prem) > 0:
                fric = float(ev_cost) / (float(prem) * 100.0)
            else:
                fric = 0.15
        fric_val = float(fric)
        if fric_val > 1.5:
            fric_val = fric_val / 100.0
        if fric_val > 0.25 or spot <= 0:
            continue

        seen.add(sym)
        scan_day = str(r.get("ts") or "")[:10]
        cr_str = f" · cr ${float(prem):.2f}" if prem is not None and float(prem) > 0 else ""
        ev_net = r.get("ev_net")
        ev_str = f" · EV ${float(ev_net):+.0f}" if ev_net is not None else ""
        opps.append(
            Opportunity(
                ticker=sym,
                asset_class="option",
                playbook_id=pb.id,
                playbook_label=pb.label,
                evidence_tier=pb.evidence_tier,
                direction="SHORT",
                spot=round(spot, 2),
                chg_1d_pct=None,
                chg_5d_pct=None,
                horizon_label=f"{dte_int}d",
                setup_summary=(
                    f"Bear Call {k_s:g}/{k_l:g} exp {exp}{cr_str}{ev_str} (scan {scan_day})"
                    if k_s is not None and k_l is not None
                    else f"Bear Call exp {exp}{cr_str}{ev_str} (scan {scan_day})"
                ),
                badges=("ETF_BEAR_CALL", f"DTE:{dte_int}", "LIMIT_ONLY", "TAXABLE_ONLY"),
                risk_flags=("FORWARD_B1F_PENDING", "TAXABLE_ONLY"),
                friction_label=f"{fric_val*100:.1f}% cr",
                friction_value=round(fric_val, 4),
                confluence_count=1,
                as_of=as_of,
                extra={"expiration": exp, "short_strike": k_s, "long_strike": k_l, "scan_date": scan_day},
            )
        )
    return opps


def scan_archived_etf_wide_spreads(
    registry: Dict[str, PlaybookMeta],
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    *,
    symbols: Sequence[str] = ("SPY", "QQQ", "IWM", "XLE", "XLV", "XLP", "XLK", "XLF", "SMH", "GLD", "TLT"),
    chain_db: str = "data/chain_archive.db",
    max_friction_to_credit: float = 0.25,
    max_snap_age_days: int = 21,
) -> List[Opportunity]:
    """Offline/instant Test O1 scanner (`60-120 DTE`, `5%-wide` ETF Bull Puts) from `data/chain_archive.db`.

    Surfaces cost-surviving wide ETF spreads immediately on `opt` / `opt radar` without
    blocking on live Yahoo calls, tagged `ARCHIVE_CHAIN` with the exact Walk-Away Limit Price.
    """
    pb = registry.get("etf_wide_spread")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return []

    from src.radar.wide_spread_rule import LegQuote, risk_free_rate, select_legs
    from src.structure.execution_plan import format_ladder_lines

    r_rate, _r_src = risk_free_rate()
    as_of = _now_utc()
    opps: List[Opportunity] = []
    for sym in symbols:
        sym_u = str(sym).upper()
        spot = float(ohlcv[sym_u][1][-1]) if sym_u in ohlcv and len(ohlcv[sym_u][1]) else None
        if spot is None or spot <= 0:
            continue
        sl = load_archived_chain_slice(
            sym_u,
            target_dte=90,
            min_dte=60,
            max_dte=120,
            spot=spot,
            chain_db=chain_db,
            max_snap_age_days=max_snap_age_days,
            build_rnd=False,
        )
        if sl is None or len(sl.puts) < 2:
            continue

        # Share the exact Test O1 -0.25 delta rule (`wide_spread_rule.select_legs`)
        # with the live scanner and backtest replay.
        leg_quotes: List[LegQuote] = []
        t_yr = max(float(sl.dte), 1.0) / 365.0
        for c in sl.puts:
            dl = c.delta
            if dl is None and c.bid > 0 and c.ask >= c.bid and c.mid > 0:
                from src.data_quality import implied_vol_from_price
                from src.utils import bs_delta

                iv_sol = c.iv or implied_vol_from_price("put", spot, c.strike, t_yr, r_rate, c.mid, 0.0)
                if iv_sol is not None:
                    dl = float(bs_delta("put", spot, c.strike, t_yr, r_rate, iv_sol, 0.0))
            leg_quotes.append(
                LegQuote(
                    strike=c.strike,
                    bid=c.bid,
                    ask=c.ask,
                    delta=dl,
                    iv=c.iv,
                    open_interest=float(c.open_interest),
                    volume=float(c.volume),
                )
            )
        legs, _reason = select_legs(
            leg_quotes,
            spot=spot,
            side="bull_put",
            expiration=sl.expiration,
            dte=float(sl.dte),
            max_friction_to_credit=max_friction_to_credit,
        )
        if legs is None:
            continue

        cr_mid, cr_cross, width, fric_ratio = (
            legs.credit_mid,
            legs.credit_cross,
            legs.width,
            legs.friction_ratio,
        )
        short_delta = float(legs.short.delta or 0.0)
        ladder = build_walk_away_ladder(
            structure="Wide Credit Spread",
            is_credit=True,
            p_mid=cr_mid,
            p_cross=cr_cross,
            ev_view_gross_per_share=cr_mid,
            max_friction_to_credit=max_friction_to_credit,
        )
        max_risk_usd = (width - ladder.p_walk) * 100.0
        flags = ["LIMIT_ONLY_NO_MKT", "TAXABLE_ONLY"]
        if sym_u != "SPY":
            flags.append("TESTED_ON_SPY_ONLY")
        if abs(legs.width_pct - 0.05) > 0.01:
            flags.append(f"WING_{legs.width_pct * 100:.1f}PCT_NOT_5")
        opps.append(
            Opportunity(
                ticker=sym_u,
                asset_class="option",
                playbook_id=pb.id,
                playbook_label=pb.label,
                evidence_tier=pb.evidence_tier,
                direction="LONG",
                spot=round(spot, 2),
                chg_1d_pct=None,
                chg_5d_pct=None,
                horizon_label=f"{sl.dte}d",
                setup_summary=(
                    f"Bull Put {legs.short.strike:g}/{legs.long.strike:g} ({short_delta:+.2f}Δ) "
                    f"exp {sl.expiration} · Mid ${cr_mid:.2f} · Min Limit ${ladder.p_walk:.2f} "
                    f"· Risk ${max_risk_usd:,.0f} (snap {sl.snap_date})"
                ),
                badges=("ARCHIVE_CHAIN", f"DTE:{sl.dte}", f"LIMIT>={ladder.p_walk:.2f}", "TP50_21DTE", "TAXABLE_ONLY"),
                risk_flags=tuple(flags),
                friction_label=f"{fric_ratio*100:.1f}% cr",
                friction_value=round(fric_ratio, 4),
                confluence_count=1,
                as_of=as_of,
                extra={
                    "expiration": sl.expiration,
                    "dte": sl.dte,
                    "snap_date": sl.snap_date,
                    "short_strike": legs.short.strike,
                    "long_strike": legs.long.strike,
                    "short_delta": round(short_delta, 3),
                    "width_pct": round(legs.width_pct, 4),
                    "credit_mid": round(cr_mid, 2),
                    "credit_cross": round(cr_cross, 2),
                    "walk_away_limit": ladder.p_walk,
                    "walk_away_steps": list(format_ladder_lines(ladder)),
                    "max_risk_usd": round(max_risk_usd, 2),
                },
            )
        )
    return opps


def scan_covered_call_holdings(
    registry: Dict[str, PlaybookMeta],
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    *,
    longterm_db: str = "data/longterm.db",
    own_picks_db: str = "data/own_picks.db",
    chain_db: str = "data/chain_archive.db",
    earnings_map: Optional[Dict[str, int]] = None,
) -> List[Opportunity]:
    """Scan >=100-share holdings in longterm.db / own_picks.db for 25-60 DTE ~0.25d Covered Calls.

    Implements the registered `covered_call_holdings` CONTEXT playbook (TFSA-legal).
    Always requires strike >= cost basis and strike > spot so calls never lock in a loss.
    """
    pb = registry.get("covered_call_holdings")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    earnings_map = earnings_map or {}
    as_of = _now_utc()

    # Aggregate >=100-share lots from longterm.db and open own_picks.db
    lots: Dict[str, Dict[str, float]] = {}
    full_lt = repo_path(longterm_db)
    if os.path.exists(full_lt):
        try:
            from src.longterm.fills import book as lt_book

            for tk, slot in lt_book(full_lt).items():
                sh = float(slot.get("shares") or 0.0)
                if sh > 0:
                    entry = lots.setdefault(tk.upper(), {"shares": 0.0, "cost": 0.0})
                    entry["shares"] += sh
                    entry["cost"] += float(slot.get("cost") or 0.0)
        except Exception:
            pass

    full_own = repo_path(own_picks_db)
    if os.path.exists(full_own):
        try:
            con = sqlite3.connect(f"file:{full_own}?mode=ro", uri=True)
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "equity_picks" in tables:
                    for tk, sh, px in con.execute(
                        "SELECT ticker, shares, entry_price FROM equity_picks WHERE status = 'OPEN' AND side = 'LONG'"
                    ).fetchall():
                        sh_f = float(sh or 0.0)
                        px_f = float(px or 0.0)
                        if sh_f > 0 and px_f > 0:
                            entry = lots.setdefault(str(tk).upper(), {"shares": 0.0, "cost": 0.0})
                            entry["shares"] += sh_f
                            entry["cost"] += sh_f * px_f
            finally:
                con.close()
        except Exception:
            pass

    opps: List[Opportunity] = []
    for sym, info in sorted(lots.items()):
        contracts = int(info["shares"] // 100)
        if contracts < 1:
            continue
        avg_cost = info["cost"] / info["shares"] if info["shares"] > 0 else 0.0
        spot = float(ohlcv[sym][1][-1]) if sym in ohlcv and len(ohlcv[sym][1]) else avg_cost
        if spot <= 0:
            continue

        min_strike = max(spot * 1.02, avg_cost)
        sl = load_archived_chain_slice(
            sym,
            target_dte=40,
            min_dte=25,
            max_dte=60,
            spot=spot,
            chain_db=chain_db,
            build_rnd=False,
        )

        chosen_k: Optional[float] = None
        exp = ""
        dte = 40
        cr_mid = 0.0
        cr_cross = 0.0
        delta_s = 0.25
        source_tag = "ARCHIVE_CHAIN"

        if sl is not None and sl.calls:
            cand = sl.match_by_delta("call", 0.25, min_strike=min_strike, min_bid=0.05)
            if cand is None:
                cands = [c for c in sl.calls if c.strike >= min_strike and c.bid >= 0.05 and c.ask >= c.bid]
                if cands:
                    cand = min(cands, key=lambda c: abs(c.strike - max(min_strike, spot * 1.05)))
            if cand is not None:
                chosen_k = cand.strike
                exp = sl.expiration
                dte = sl.dte
                cr_mid = cand.mid
                cr_cross = cand.bid
                delta_s = abs(cand.delta) if cand.delta is not None else 0.25

        if chosen_k is None:
            # Clean BS estimate fallback when symbol isn't in chain_archive.db
            from src.structure.express import _bs_px_delta

            rv = 0.25
            if sym in ohlcv and len(ohlcv[sym][1]) >= 25:
                rv_val = float(trailing_rv(ohlcv[sym][1], 20)[-1])
                if np.isfinite(rv_val) and rv_val > 0.05:
                    rv = rv_val
            step = 1.0 if spot < 50 else (2.5 if spot < 150 else 5.0)
            chosen_k = math.ceil(max(min_strike, spot * 1.05) / step) * step
            px, dl, _ = _bs_px_delta(spot, chosen_k, 40.0 / 365.0, 0.045, rv, "call")
            cr_mid = px
            cr_cross = max(0.05, px * 0.96)
            delta_s = abs(dl)
            exp = (dt.date.today() + dt.timedelta(days=40)).isoformat()
            dte = 40
            source_tag = "EST_BS"

        if cr_cross <= 0:
            continue

        ann_yield_pct = (cr_cross / spot) * (365.0 / max(dte, 1)) * 100.0
        half_sp = max(0.0, cr_mid - cr_cross)
        fric_ratio = (half_sp / cr_cross) if cr_cross > 0 else 0.20
        ladder = build_walk_away_ladder(
            structure="Covered Call",
            is_credit=True,
            p_mid=cr_mid,
            p_cross=cr_cross,
            ev_view_gross_per_share=cr_mid,
            max_friction_to_credit=0.25,
        )
        risk_flags: List[str] = []
        if sym in earnings_map and earnings_map[sym] <= dte:
            risk_flags.append(f"EARN:{earnings_map[sym]}d")
        if source_tag == "EST_BS":
            risk_flags.append("NO_ARCHIVED_CHAIN")

        opps.append(
            Opportunity(
                ticker=sym,
                asset_class="option",
                playbook_id=pb.id,
                playbook_label=pb.label,
                evidence_tier=pb.evidence_tier,
                direction="NEUTRAL",
                spot=round(spot, 2),
                chg_1d_pct=None,
                chg_5d_pct=None,
                horizon_label=f"{dte}d",
                setup_summary=(
                    f"Sell {contracts}x {chosen_k:g}C exp {exp} (~{delta_s:.2f}Δ) · "
                    f"Mid ${cr_mid:.2f} · Min ${ladder.p_walk:.2f} · ~{ann_yield_pct:.1f}% ann"
                ),
                badges=("COVERED_CALL", "TFSA_OK", source_tag, f"LOTS:{contracts}"),
                risk_flags=tuple(risk_flags),
                friction_label=f"{fric_ratio*100:.1f}% cr",
                friction_value=round(fric_ratio, 4),
                confluence_count=1,
                as_of=as_of,
                extra={
                    "contracts": contracts,
                    "avg_cost": round(avg_cost, 2),
                    "strike": chosen_k,
                    "expiration": exp,
                    "dte": dte,
                    "credit_mid": round(cr_mid, 2),
                    "walk_away_limit": ladder.p_walk,
                    "ann_yield_pct": round(ann_yield_pct, 1),
                    "source": source_tag,
                },
            )
        )
    return opps


def scan_cached_insider_clusters(
    registry: Dict[str, PlaybookMeta],
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    *,
    insider_json: str = "data/insider_cache.json",
    insider_db: str = "data/insider_cache.db",
) -> List[Opportunity]:
    """Read cached SEC Form 4 cluster buys (`data/insider_cache.json`, with optional SQLite fallback)."""
    pb = registry.get("insider_cluster")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    as_of = _now_utc()
    opps: List[Opportunity] = []

    # Primary live cache format: data/insider_cache.json
    full_json = repo_path(insider_json)
    if os.path.exists(full_json):
        try:
            with open(full_json, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                hits = []
                for sym, payload in raw.items():
                    if not isinstance(payload, dict):
                        continue
                    summ = payload.get("summary") or {}
                    n_buyers = int(summ.get("n_buyers") or 0)
                    buy_val = float(summ.get("buy_value") or 0.0)
                    if n_buyers >= 2:
                        hits.append((n_buyers, buy_val, str(sym).upper(), str(payload.get("as_of") or "")))
                hits.sort(key=lambda x: (-x[0], -x[1], x[2]))
                for n_buyers, buy_val, sym, seen_dt in hits[:15]:
                    spot = float(ohlcv[sym][1][-1]) if sym in ohlcv and len(ohlcv[sym][1]) else 0.0
                    opps.append(
                        Opportunity(
                            ticker=sym,
                            asset_class="stock",
                            playbook_id=pb.id,
                            playbook_label=pb.label,
                            evidence_tier=pb.evidence_tier,
                            direction="LONG",
                            spot=round(spot, 2),
                            chg_1d_pct=None,
                            chg_5d_pct=None,
                            horizon_label=pb.horizon_label,
                            setup_summary=f"{n_buyers} insiders bought ${buy_val:,.0f} (as of {seen_dt})",
                            badges=(f"INSIDERx{n_buyers}",),
                            risk_flags=(),
                            friction_label="0.30% stk",
                            friction_value=0.0030,
                            confluence_count=1,
                            as_of=as_of,
                            extra={"buyer_count": n_buyers, "buy_value_usd": buy_val, "seen_date": seen_dt},
                        )
                    )
                return opps
        except Exception:
            pass

    # Fallback: legacy SQLite schema if explicitly present
    full_db = repo_path(insider_db)
    if not os.path.exists(full_db):
        return []
    try:
        con = sqlite3.connect(f"file:{full_db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "insider_clusters" not in tables:
                return []
            rows = con.execute(
                "SELECT ticker, buyer_count, total_usd, latest_date FROM insider_clusters "
                "WHERE buyer_count >= 2 ORDER BY latest_date DESC LIMIT 15"
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return []

    for r in rows:
        sym = str(r["ticker"]).upper()
        spot = float(ohlcv[sym][1][-1]) if sym in ohlcv and len(ohlcv[sym][1]) else 0.0
        opps.append(
            Opportunity(
                ticker=sym,
                asset_class="stock",
                playbook_id=pb.id,
                playbook_label=pb.label,
                evidence_tier=pb.evidence_tier,
                direction="LONG",
                spot=round(spot, 2),
                chg_1d_pct=None,
                chg_5d_pct=None,
                horizon_label=pb.horizon_label,
                setup_summary=f"{int(r['buyer_count'])} insiders bought ${float(r['total_usd']):,.0f} ({r['latest_date']})",
                badges=(f"INSIDERx{int(r['buyer_count'])}",),
                risk_flags=(),
                friction_label="0.30% stk",
                friction_value=0.0030,
                confluence_count=1,
                as_of=as_of,
            )
        )
    return opps


def _next_verified_macro_event(today: Optional[dt.date] = None) -> str:
    """Return the nearest upcoming verified FOMC/CPI/NFP event from `src.macro_analyzer`."""
    ref = today or dt.date.today()
    try:
        from src.macro_analyzer import _DEFAULT_EVENTS

        upcoming: List[Tuple[int, str, str]] = []
        for ev in _DEFAULT_EVENTS:
            try:
                ev_dt = dt.date.fromisoformat(str(ev["date"]))
            except (KeyError, ValueError):
                continue
            delta = (ev_dt - ref).days
            if 0 <= delta <= 45:
                upcoming.append((delta, str(ev["name"]), str(ev["date"])))
        if upcoming:
            upcoming.sort()
            days, name, dstr = upcoming[0]
            return f"{name} {dstr} ({days}d)"
    except Exception:
        pass
    return "none"


def compute_regime_header(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    *,
    predmarkets_db: str = "data/predmarkets.db",
) -> Dict[str, Any]:
    """Build deterministic macro/regime header from local SPY/QQQ + verified macro calendar + Kalshi odds."""
    spy = ohlcv.get("SPY")
    qqq = ohlcv.get("QQQ")
    spy_last = spy_chg = qqq_chg = None
    ohlcv_as_of: Optional[str] = None
    posture = "NEUTRAL"
    vix = vix_3m = None
    spy_rv20: Optional[float] = None
    vix_term = "UNKNOWN"

    if spy is not None and len(spy[1]) >= 200:
        ds, cs, _ = spy
        ohlcv_as_of = str(ds[-1]) if ds else None
        spy_last = round(float(cs[-1]), 2)
        spy_chg = round(float(cs[-1] / cs[-2] - 1.0) * 100.0, 2)
        s200 = float(np.mean(cs[-200:]))
        rv20 = float(trailing_rv(cs, 20)[-1])
        if float(cs[-1]) > s200 and rv20 < 0.22:
            posture = "RISK_ON"
        elif float(cs[-1]) < s200 or rv20 >= 0.28:
            posture = "RISK_OFF"
        else:
            posture = "NEUTRAL"
        spy_rv20 = round(rv20 * 100.0, 1)

    # Real Cboe VIX / VIX3M (archived daily histories); never SPY's realized vol.
    vix_as_of: Optional[str] = None
    try:
        from src.topdown.macro import vol_regime

        vr = vol_regime()
        if vr.get("vix") is not None and not vr.get("stale"):
            vix, vix_3m = vr.get("vix"), vr.get("vix3m")
            vix_term = str(vr.get("term") or "UNKNOWN")
            vix_as_of = vr.get("as_of")
    except Exception:
        pass

    if qqq is not None and len(qqq[1]) >= 2:
        qcs = qqq[1]
        qqq_chg = round(float(qcs[-1] / qcs[-2] - 1.0) * 100.0, 2)

    next_macro = _next_verified_macro_event()
    kalshi_macro = load_kalshi_macro_pulse(predmarkets_db=predmarkets_db)

    return {
        "regime_posture": posture,
        "vix": vix,
        "vix_3m": vix_3m,
        "vix_term": vix_term,
        "spy_last": spy_last,
        "spy_chg_1d_pct": spy_chg,
        "qqq_chg_1d_pct": qqq_chg,
        "next_macro_event": next_macro,
        "kalshi_macro": kalshi_macro,
        "ohlcv_as_of": ohlcv_as_of,
        "ohlcv_symbol_count": len(ohlcv),
        "spy_rv20": spy_rv20,
        "vix_as_of": vix_as_of,
    }


def load_local_ohlcv_safe(
    ohlcv_db: str = "data/equity_ohlcv.db",
    *,
    symbols: Optional[Sequence[str]] = None,
    lookback_days: Optional[int] = 450,
) -> Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]:
    full = repo_path(ohlcv_db)
    if not os.path.exists(full):
        return {}
    since_date: Optional[str] = None
    if lookback_days is not None and lookback_days > 0:
        try:
            con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
            try:
                probe_tk = str(symbols[0]).upper() if symbols else "SPY"
                row = con.execute("SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (probe_tk,)).fetchone()
                if not row or not row[0]:
                    row = con.execute("SELECT MAX(date) FROM ohlcv").fetchone()
                if row and row[0]:
                    mx = dt.date.fromisoformat(str(row[0])[:10])
                    since_date = (mx - dt.timedelta(days=int(lookback_days))).isoformat()
            finally:
                con.close()
        except Exception:
            since_date = None
    return load_equity_ohlcv(full, symbols=symbols, since_date=since_date)
