"""Failure-isolated scanner adapters feeding the Unified Radar Engine.

All default adapters operate on local SQLite caches (`data/equity_ohlcv.db`,
`data/dolt_options.db`, `data/estimates_archive.db`, `data/longterm.db`,
`data/catalysts.db`) so interactive startup completes in <1s without network hangs.
Live option scans are opt-in (`allow_network=True`).
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

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
from .playbooks import PlaybookMeta
from .schema import Opportunity

OUTLOOK_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
    "XLC", "SMH", "GLD", "TLT", "IWM", "QQQ",
]

SECTOR_MAP_FALLBACK: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH",
    "INTC": "SMH", "QCOM": "SMH", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "PLTR": "XLK", "SMCI": "XLK", "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "RIVN": "XLY", "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "COIN": "XLF",
    "XOM": "XLE", "CVX": "XLE", "UNH": "XLV", "LLY": "XLV", "JNJ": "XLV",
    "PG": "XLP", "KO": "XLP", "WMT": "XLP", "COST": "XLP", "CAT": "XLI", "BA": "XLI",
}


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_latest_rev_breadth(db_path: str = "data/estimates_archive.db") -> Dict[str, float]:
    full = repo_path(db_path)
    if not os.path.exists(full):
        return {}
    try:
        from scripts.estimate_revisions_test import compute_rev_breadth_30d

        by_date = compute_rev_breadth_30d(full)
        if not by_date:
            return {}
        latest = max(by_date.keys())
        return by_date[latest]
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
        # Find reaction session index
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
    """Map ticker -> calendar days to next known earnings within horizon_days."""
    full = repo_path(dolt_db)
    if not os.path.exists(full):
        return {}
    today = dt.date.today()
    out: Dict[str, int] = {}
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT symbol, MIN(date) FROM earnings_cal "
                "WHERE date >= ? AND date <= ? GROUP BY symbol",
                (today.isoformat(), (today + dt.timedelta(days=horizon_days)).isoformat()),
            ).fetchall()
        finally:
            con.close()
        for sym, dstr in rows:
            days = (dt.date.fromisoformat(dstr) - today).days
            out[str(sym).upper()] = days
    except Exception:
        return {}
    return out


def scan_vol_managed_outlook(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    registry: Dict[str, PlaybookMeta],
    *,
    sigma_target: float = 0.14,
) -> Tuple[List[Dict[str, Any]], List[Opportunity], Dict[str, str]]:
    """Compute Validated Playbook 2 (VM_OUTLOOK_TOP3) target allocation + sector posture map."""
    pb = registry.get("vol_managed_outlook")
    as_of = _now_utc()
    spy_tuple = ohlcv.get("SPY")
    if spy_tuple is None:
        return [], [], {}

    spy_ds, spy_cs, _ = spy_tuple
    if len(spy_cs) < 252:
        return [], [], {}
    spy_63 = float(spy_cs[-1] / spy_cs[-64] - 1.0)
    spy_rv20 = trailing_rv(spy_cs, window=20)[-1]
    if not np.isfinite(spy_rv20) or spy_rv20 <= 0:
        spy_rv20 = sigma_target
    vol_scale = float(np.clip(sigma_target / float(spy_rv20), 0.25, 1.00))

    raw_factors: Dict[str, Dict[str, Optional[float]]] = {}
    etf_meta: Dict[str, Dict[str, float]] = {}
    for sym in OUTLOOK_ETFS:
        if sym not in ohlcv:
            continue
        ds, cs, _ = ohlcv[sym]
        if len(cs) < 253:
            continue
        m12_1 = float(cs[-22] / cs[-253] - 1.0)
        s50 = float(np.mean(cs[-50:]))
        s200 = float(np.mean(cs[-200:]))
        trend = (float(cs[-1]) / s200 - 1.0) + 0.5 * (s50 / s200 - 1.0)
        rel_str = float(cs[-1] / cs[-64] - 1.0) - spy_63
        rev_1m = -float(cs[-1] / cs[-22] - 1.0)
        raw_factors[sym] = {
            "mom_12_1": m12_1,
            "trend_score": trend,
            "relative_strength": rel_str,
            "reversal_1m": rev_1m,
        }
        etf_meta[sym] = {
            "spot": float(cs[-1]),
            "chg_1d": float(cs[-1] / cs[-2] - 1.0) * 100.0,
            "chg_5d": float(cs[-1] / cs[-6] - 1.0) * 100.0,
            "mom_12_1": m12_1 * 100.0,
            "above_200": float(cs[-1] > s200),
        }

    if not raw_factors:
        return [], [], {}

    from src.outlook.engine import zscore_map

    weights = {"mom_12_1": 0.35, "trend_score": 0.30, "relative_strength": 0.20, "reversal_1m": 0.15}
    z_by_factor = {
        fname: zscore_map({sym: vals[fname] for sym, vals in raw_factors.items()})
        for fname in weights
    }
    scored: List[Tuple[str, float]] = []
    for sym in raw_factors:
        comp = sum(weights[f] * z_by_factor[f].get(sym, 0.0) for f in weights)
        scored.append((sym, comp))
    scored.sort(key=lambda x: x[1], reverse=True)

    posture_map: Dict[str, str] = {}
    for idx, (sym, comp) in enumerate(scored):
        if idx < 4 and comp >= 0.25:
            posture_map[sym] = "FAVOR"
        elif comp <= -0.35:
            posture_map[sym] = "AVOID"
        else:
            posture_map[sym] = "NEUTRAL"

    # Top-3 equal-weighted scaled by vol_scale (remainder in CASH/T-bills)
    top3 = scored[:3]
    each_w = round(vol_scale / max(len(top3), 1), 4)
    core_alloc: List[Dict[str, Any]] = []
    opps: List[Opportunity] = []
    for rank, (sym, comp) in enumerate(top3, 1):
        m = etf_meta[sym]
        core_alloc.append(
            {
                "ticker": sym,
                "rank": rank,
                "target_weight": each_w,
                "vol_scale": round(vol_scale, 3),
                "spy_rv20": round(float(spy_rv20), 4),
                "composite": round(comp, 3),
                "spot": round(m["spot"], 2),
                "mom_12_1_pct": round(m["mom_12_1"], 1),
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
                        f"Rank #{rank}/16 · Weight {each_w*100:.0f}% (scale {vol_scale*100:.0f}%) "
                        f"· 12-1m {m['mom_12_1']:+.1f}%"
                    ),
                    badges=(f"FAVOR:{sym}", f"WEIGHT:{each_w*100:.0f}%", "TFSA_OK"),
                    risk_flags=("VOL_SCALED_DOWN",) if vol_scale < 0.75 else (),
                    friction_label="0.06% etf",
                    friction_value=0.0006,
                    confluence_count=1,
                    as_of=as_of,
                    extra={"rank": rank, "target_weight": each_w, "vol_scale": vol_scale},
                )
            )
    cash_w = round(max(0.0, 1.0 - each_w * len(top3)), 4)
    if cash_w > 0.005:
        core_alloc.append(
            {
                "ticker": "CASH",
                "rank": 0,
                "target_weight": cash_w,
                "vol_scale": round(vol_scale, 3),
                "spy_rv20": round(float(spy_rv20), 4),
                "composite": 0.0,
                "spot": 1.0,
                "mom_12_1_pct": 0.0,
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
        sec = SECTOR_MAP_FALLBACK.get(sym)
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
                asset_class="etf" if sym in OUTLOOK_ETFS else "stock",
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
    plan_path: str = "configs/longterm_plan.json",
    fills_db: str = "data/longterm.db",
) -> List[Opportunity]:
    """Emit CONTEXT alerts for long-term plan holdings in or near buy zones."""
    pb = registry.get("holdings_buy_zone")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    full_plan = repo_path(plan_path)
    if not os.path.exists(full_plan):
        return []
    as_of = _now_utc()
    opps: List[Opportunity] = []
    try:
        from src.longterm.plan import load_plan

        plan = load_plan(full_plan)
    except Exception:
        return []

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
        levels = sorted(t.level for t in name.tranches)
        top_level = levels[-1]
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


def scan_cached_etf_bear_calls(
    registry: Dict[str, PlaybookMeta],
    *,
    candidates_db: str = "data/candidates.db",
    max_age_days: int = 5,
) -> List[Opportunity]:
    """Read recent gate-passed ETF Bear Calls (LEAD playbook, B1-F pending 2027-01-20) from candidates.db."""
    pb = registry.get("etf_bear_call")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    full = repo_path(candidates_db)
    if not os.path.exists(full):
        return []
    as_of = _now_utc()
    etf_set = set(OUTLOOK_ETFS) | {"SPY", "DIA", "EEM", "EFA", "HYG", "KRE", "SLV", "GDX", "XBI"}
    cutoff = (dt.date.today() - dt.timedelta(days=max_age_days)).isoformat()
    opps: List[Opportunity] = []
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            cols = {r["name"] for r in con.execute("PRAGMA table_info(candidates)")}
            if not {"symbol", "strategy", "gate_passed", "scan_date"}.issubset(cols):
                return []
            rows = con.execute(
                """
                SELECT symbol, scan_date, expiration, strike, long_strike, entry_price,
                       underlying_price, dte, friction_to_credit
                FROM candidates
                WHERE strategy = 'Bear Call'
                  AND gate_passed = 1
                  AND scan_date >= ?
                ORDER BY scan_date DESC
                LIMIT 40
                """,
                (cutoff,),
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return []

    seen = set()
    for r in rows:
        sym = str(r["symbol"]).upper()
        if sym not in etf_set or sym in seen:
            continue
        seen.add(sym)
        spot = float(r["underlying_price"] or 0.0)
        fric = float(r["friction_to_credit"] or 0.15)
        if fric > 0.25 or spot <= 0:
            continue
        k_s = r["strike"]
        k_l = r["long_strike"]
        dte = int(r["dte"] or 30)
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
                horizon_label=f"{dte}d",
                setup_summary=(
                    f"Bear Call {k_s}/{k_l} exp {r['expiration']} · cr ${float(r['entry_price'] or 0):.2f} "
                    f"(from {r['scan_date']})"
                ),
                badges=("ETF_BEAR_CALL", f"DTE:{dte}", "LIMIT_ONLY"),
                risk_flags=("FORWARD_B1F_PENDING",),
                friction_label=f"{fric*100:.1f}% cr",
                friction_value=fric,
                confluence_count=1,
                as_of=as_of,
                extra={"expiration": r["expiration"], "short_strike": k_s, "long_strike": k_l},
            )
        )
    return opps


def scan_cached_insider_clusters(
    registry: Dict[str, PlaybookMeta],
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    *,
    insider_db: str = "data/insider_cache.db",
) -> List[Opportunity]:
    """Read cached SEC Form 4 cluster buys without blocking on live EDGAR crawls."""
    pb = registry.get("insider_cluster")
    if pb is None or not pb.is_default_radar_eligible:
        return []
    full = repo_path(insider_db)
    if not os.path.exists(full):
        return []
    as_of = _now_utc()
    opps: List[Opportunity] = []
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
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


def compute_regime_header(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Build deterministic macro/regime header from local SPY/QQQ + optional VIX/macro calendar."""
    spy = ohlcv.get("SPY")
    qqq = ohlcv.get("QQQ")
    spy_last = spy_chg = qqq_chg = None
    posture = "NEUTRAL"
    vix = vix_3m = None
    vix_term = "UNKNOWN"

    if spy is not None and len(spy[1]) >= 200:
        cs = spy[1]
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
        # Proxy implied vol when live VIX isn't queried offline
        vix = round(rv20 * 100.0, 1)

    if qqq is not None and len(qqq[1]) >= 2:
        qcs = qqq[1]
        qqq_chg = round(float(qcs[-1] / qcs[-2] - 1.0) * 100.0, 2)

    next_macro = "none"
    try:
        from src.macro_calendar import upcoming_events

        evs = upcoming_events(days_ahead=14)
        if evs:
            ev0 = evs[0]
            next_macro = f"{ev0.get('event', ev0.get('name', 'Macro'))} {ev0.get('date', '')}".strip()
    except Exception:
        pass

    return {
        "regime_posture": posture,
        "vix": vix,
        "vix_3m": vix_3m,
        "vix_term": vix_term,
        "spy_last": spy_last,
        "spy_chg_1d_pct": spy_chg,
        "qqq_chg_1d_pct": qqq_chg,
        "next_macro_event": next_macro,
    }


def load_local_ohlcv_safe(ohlcv_db: str = "data/equity_ohlcv.db") -> Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]:
    full = repo_path(ohlcv_db)
    if not os.path.exists(full):
        return {}
    return load_equity_ohlcv(full)
