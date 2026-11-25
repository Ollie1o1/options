"""Option-Implied Signals Executed in SHARES + Dealer Gamma Exposure (GEX).

Economic basis (unexhausted in this repo's NULL_LEDGER):
  1. Cremers & Weinbaum (2010): ATM Call-Put IV spread (`IV_call_ATM - IV_put_ATM`)
     reflects informed options order flow; equity arbitrage lags. Rather than paying
     wide option bid-ask spreads, harvest the signal in $0-commission Wealthsimple shares!
  2. Xing, Zhang & Zhao (2010): OTM Put Smirk (`IV_put_25d - IV_call_ATM`). Unusually
     steep put skew predicts underperformance / crash fear (landmine / avoid signal).
  3. Dealer Gamma Exposure (GEX) & OI Walls: Computed free from archived CBOE chains
     (gamma * OI * 100 * spot^2 * 0.01). Positive net GEX dampens realized volatility
     (mean-reverting tape); negative net GEX amplifies moves (trending/gapping tape).

Evidence label: WATCHLIST (descriptive/literature-backed; preregister OOS before sizing up).
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import repo_path

# 60 liquid ETFs & large caps ideal for expanding chain_archive.db coverage
EXPANDED_CHAIN_UNIVERSE: Tuple[str, ...] = (
    # Core Sector & Macro ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
    "XLY", "XLU", "XLB", "XLRE", "XLC", "SMH", "GLD", "SLV", "TLT", "HYG",
    "EFA", "EEM", "GDX", "XBI",
    # Liquid Single-Name Leaders (TFSA-friendly USD & CDR overlap)
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
    "JPM", "BAC", "GS", "V", "MA", "UNH", "LLY", "JNJ", "ABBV", "MRK",
    "XOM", "CVX", "COP", "COST", "WMT", "HD", "MCD", "KO", "PG", "CAT",
    "GE", "RTX", "ORCL", "CRM", "PLTR", "COIN",
)


@dataclass(frozen=True)
class OptionShareSignal:
    ticker: str
    snap_date: str
    spot: float
    cw_iv_spread_vp: Optional[float]      # (ATM Call IV - ATM Put IV) * 100
    put_smirk_vp: Optional[float]         # (25d Put IV - ATM Call IV) * 100
    atm_iv_pct: Optional[float]
    net_gex_m: Optional[float]            # Net dealer GEX in $M per 1% spot move
    gex_regime: str                       # "POSITIVE_GEX" | "NEGATIVE_GEX" | "UNKNOWN"
    gamma_flip: Optional[float]
    call_wall: Optional[float]
    put_wall: Optional[float]
    signal_label: str                     # "BULL_FLOW_SHARES" | "CRASH_SMIRK_AVOID" | "NEUTRAL"
    evidence_label: str = "WATCHLIST"
    source: str = "data/chain_archive.db"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compute_dealer_gex(
    rows: Sequence[Dict[str, Any]],
    spot: float,
) -> Dict[str, Any]:
    """Compute Dealer Net GEX ($M per 1% spot move), Gamma Flip strike, and Call/Put OI Walls.

    Standard dealer positioning convention:
      - Call GEX = +gamma * OI * 100 * spot^2 * 0.01  (dealers net long calls / stabilizing)
      - Put  GEX = -gamma * OI * 100 * spot^2 * 0.01  (dealers net short puts / destabilizing)
    """
    if spot <= 0 or not rows:
        return {
            "net_gex_m": None,
            "gex_regime": "UNKNOWN",
            "gamma_flip": None,
            "call_wall": None,
            "put_wall": None,
            "by_strike": [],
        }

    net_by_strike: Dict[float, float] = {}
    call_oi: Dict[float, float] = {}
    put_oi: Dict[float, float] = {}
    total_gex = 0.0
    valid_n = 0

    for r in rows:
        k = _clean(r.get("strike"))
        gamma = _clean(r.get("gamma"))
        oi = _clean(r.get("open_interest"))
        ot = str(r.get("type") or "").lower()
        if k is None or k <= 0 or oi is None or oi <= 0:
            continue
        if abs(k / spot - 1.0) > 0.20:
            continue
        if ot.startswith("c"):
            call_oi[k] = call_oi.get(k, 0.0) + oi
        elif ot.startswith("p"):
            put_oi[k] = put_oi.get(k, 0.0) + oi
        if gamma is None or gamma <= 0:
            continue
        sign = 1.0 if ot.startswith("c") else (-1.0 if ot.startswith("p") else 0.0)
        if sign == 0.0:
            continue
        gex_usd = sign * gamma * oi * 100.0 * (spot * spot) * 0.01
        total_gex += gex_usd
        net_by_strike[k] = net_by_strike.get(k, 0.0) + gex_usd
        valid_n += 1

    if valid_n == 0:
        return {
            "net_gex_m": None,
            "gex_regime": "UNKNOWN",
            "gamma_flip": None,
            "call_wall": max(call_oi, key=lambda k: call_oi[k]) if call_oi else None,
            "put_wall": max(put_oi, key=lambda k: put_oi[k]) if put_oi else None,
            "by_strike": [],
        }

    net_m = round(total_gex / 1e6, 2)
    regime = "POSITIVE_GEX" if net_m >= 0 else "NEGATIVE_GEX"

    # Estimate gamma flip where cumulative net GEX by strike crosses zero near spot
    sorted_k = sorted(net_by_strike.keys())
    cum = 0.0
    prev_k: Optional[float] = None
    prev_cum: Optional[float] = None
    flip: Optional[float] = None
    for k in sorted_k:
        cum += net_by_strike[k]
        if prev_k is not None and prev_cum is not None:
            if (prev_cum < 0 <= cum) or (prev_cum > 0 >= cum):
                frac = abs(prev_cum) / max(abs(cum - prev_cum), 1e-9)
                cand = prev_k + frac * (k - prev_k)
                if flip is None or abs(cand - spot) < abs(flip - spot):
                    flip = round(cand, 2)
        prev_k, prev_cum = k, cum

    call_wall = max(call_oi, key=lambda k: call_oi[k]) if call_oi else None
    put_wall = max(put_oi, key=lambda k: put_oi[k]) if put_oi else None
    top_strikes = sorted(
        [{"strike": k, "gex_m": round(v / 1e6, 2)} for k, v in net_by_strike.items()],
        key=lambda d: abs(d["gex_m"]),
        reverse=True,
    )[:8]

    return {
        "net_gex_m": net_m,
        "gex_regime": regime,
        "gamma_flip": flip,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "by_strike": top_strikes,
    }


def _compute_ticker_signal(
    symbol: str,
    snap_date: str,
    rows: Sequence[Dict[str, Any]],
) -> Optional[OptionShareSignal]:
    spots = [_clean(r.get("spot")) for r in rows]
    valid_spots = [s for s in spots if s is not None and s > 0]
    if not valid_spots:
        return None
    spot = float(np.median(valid_spots))

    # Filter to 14-60 DTE slice for clean Cremers-Weinbaum / Xing-Zhang-Zhao measurements
    try:
        d0 = dt.date.fromisoformat(snap_date[:10])
    except ValueError:
        return None

    in_win: List[Dict[str, Any]] = []
    for r in rows:
        exp = str(r.get("expiration") or "")[:10]
        try:
            dte = (dt.date.fromisoformat(exp) - d0).days
        except ValueError:
            continue
        if 14 <= dte <= 65:
            in_win.append(r)

    pool = in_win if len(in_win) >= 6 else list(rows)

    # 1. Cremers-Weinbaum ATM Call-Put IV Spread (matched strikes within +-3% of spot)
    call_atm: Dict[Tuple[str, float], float] = {}
    put_atm: Dict[Tuple[str, float], float] = {}
    all_atm_iv: List[float] = []
    put_25d_cands: List[Tuple[float, float]] = []

    for r in pool:
        k = _clean(r.get("strike"))
        iv = _clean(r.get("iv"))
        bid = _clean(r.get("bid"))
        ask = _clean(r.get("ask"))
        delta = _clean(r.get("delta"))
        ot = str(r.get("type") or "").lower()
        exp = str(r.get("expiration") or "")[:10]
        if k is None or iv is None or not (0.03 <= iv <= 3.5):
            continue
        if bid is None or ask is None or bid <= 0 or ask < bid:
            continue
        rel = abs(k / spot - 1.0)
        if rel <= 0.03:
            all_atm_iv.append(iv)
            if ot.startswith("c"):
                call_atm[(exp, k)] = iv
            elif ot.startswith("p"):
                put_atm[(exp, k)] = iv
        if ot.startswith("p") and delta is not None and -0.35 <= delta <= -0.15:
            put_25d_cands.append((abs(delta + 0.25), iv))

    paired_diffs = [
        (call_atm[key] - put_atm[key]) * 100.0
        for key in call_atm
        if key in put_atm
    ]
    cw_spread = round(float(np.median(paired_diffs)), 2) if paired_diffs else None
    atm_call_iv = float(np.median(list(call_atm.values()))) if call_atm else (float(np.median(all_atm_iv)) if all_atm_iv else None)
    atm_iv_pct = round(float(np.median(all_atm_iv)) * 100.0, 1) if all_atm_iv else None

    put_smirk: Optional[float] = None
    if put_25d_cands and atm_call_iv is not None:
        put_25d_cands.sort(key=lambda x: x[0])
        p25_iv = put_25d_cands[0][1]
        put_smirk = round((p25_iv - atm_call_iv) * 100.0, 2)

    # 2. Dealer GEX across all near-term expirations (<= 65 DTE)
    gex_info = compute_dealer_gex(pool, spot)

    # 3. Classify signal for TFSA shares execution
    if cw_spread is not None and cw_spread >= 1.5 and (put_smirk is None or put_smirk < 8.0):
        sig = "BULL_FLOW_SHARES"
    elif (put_smirk is not None and put_smirk >= 10.0) or (cw_spread is not None and cw_spread <= -2.5):
        sig = "CRASH_SMIRK_AVOID"
    else:
        sig = "NEUTRAL"

    return OptionShareSignal(
        ticker=symbol.upper(),
        snap_date=snap_date,
        spot=round(spot, 2),
        cw_iv_spread_vp=cw_spread,
        put_smirk_vp=put_smirk,
        atm_iv_pct=atm_iv_pct,
        net_gex_m=gex_info["net_gex_m"],
        gex_regime=str(gex_info["gex_regime"]),
        gamma_flip=gex_info["gamma_flip"],
        call_wall=gex_info["call_wall"],
        put_wall=gex_info["put_wall"],
        signal_label=sig,
    )


def scan_option_signals_for_shares(
    *,
    chain_db: str = "data/chain_archive.db",
    symbols: Optional[Sequence[str]] = None,
    max_snap_age_days: int = 30,
) -> List[OptionShareSignal]:
    """Scan latest archived chains in `chain_archive.db` for CW IV spread, Put Smirk & Dealer GEX."""
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return []
    cutoff = (dt.date.today() - dt.timedelta(days=max_snap_age_days)).isoformat()
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        if symbols:
            syms_u = [s.upper() for s in symbols]
            ph = ",".join("?" * len(syms_u))
            latest_rows = con.execute(
                f"SELECT symbol, MAX(snap_date) AS mx FROM chain_snapshots "
                f"WHERE symbol IN ({ph}) AND snap_date >= ? GROUP BY symbol",
                (*syms_u, cutoff),
            ).fetchall()
        else:
            latest_rows = con.execute(
                "SELECT symbol, MAX(snap_date) AS mx FROM chain_snapshots "
                "WHERE snap_date >= ? GROUP BY symbol",
                (cutoff,),
            ).fetchall()

        results: List[OptionShareSignal] = []
        for lr in latest_rows:
            sym = str(lr["symbol"]).upper()
            sdate = str(lr["mx"])
            q_rows = [
                dict(r)
                for r in con.execute(
                    "SELECT type, strike, expiration, bid, ask, iv, delta, gamma, open_interest, spot "
                    "FROM chain_snapshots WHERE symbol = ? AND snap_date = ?",
                    (sym, sdate),
                ).fetchall()
            ]
            sig = _compute_ticker_signal(sym, sdate, q_rows)
            if sig is not None:
                results.append(sig)
    finally:
        con.close()

    # Sort bullish flow first, then highest CW IV spread, lowest put smirk
    rank_order = {"BULL_FLOW_SHARES": 0, "NEUTRAL": 1, "CRASH_SMIRK_AVOID": 2}
    results.sort(
        key=lambda x: (
            rank_order.get(x.signal_label, 9),
            -(x.cw_iv_spread_vp if x.cw_iv_spread_vp is not None else -999.0),
            x.put_smirk_vp if x.put_smirk_vp is not None else 999.0,
            x.ticker,
        )
    )
    return results
