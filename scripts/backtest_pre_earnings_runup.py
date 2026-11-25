#!/usr/bin/env python3
"""Test O2: Pre-Earnings Anticipation Window (T-10 -> T-1, Closed STRICTLY Before Report).

Frozen registration: docs/PREREG_OPTIONS_DTE_AND_PRE_EARNINGS_20261124.md
Data: data/dolt_options.db (earnings_cal, dolt_chain, stocks_close) + data/squeeze_prices.db (SPY)
Output: reports/backtest_pre_earnings_runup_20261124.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.equity_playbooks.backtest import BenchmarkSeries

DOLT_DB = "data/dolt_options.db"
OHLCV_DB = "data/equity_ohlcv.db"
OUT_PATH = "reports/backtest_pre_earnings_runup_20261124.json"

SEED = 20261124
N_BOOT = 10_000
ALPHA_BONF = 0.05 / 3.0  # 3 claimants (O2-EQ, O2-LC, O2-ST) -> 98.33% CI
MIN_EVENTS = 200
MIN_WEEKS = 50
EQ_RT_FRICTION = 0.0010  # 10 bps round-trip

MEGACAPS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "AVGO", "JPM", "V", "MA", "UNH", "XOM", "COST", "HD", "WMT", "LLY",
    "NFLX", "AMD", "CRM", "ORCL", "ADBE", "INTC", "QCOM", "BAC", "GS",
}


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _iso_week(s: str) -> str:
    iso = _d(s).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _ok_entry(r) -> bool:
    return (
        r.bid is not None
        and r.ask is not None
        and r.mid is not None
        and r.bid > 0
        and r.ask >= r.bid
        and r.mid > 0
        and (r.ask - r.bid) / r.mid <= 0.25
    )


def _interp(side: pd.DataFrame, k: float) -> Optional[Tuple[float, float, float]]:
    """Convex linear interpolation of (bid, ask, mid) between adjacent quoted strikes."""
    lo = side[side["strike"] < k]
    hi = side[side["strike"] > k]
    if lo.empty or hi.empty:
        return None
    a = lo.loc[lo["strike"].idxmax()]
    b = hi.loc[hi["strike"].idxmin()]
    w = float(k - a["strike"]) / float(b["strike"] - a["strike"])
    return (
        float((1.0 - w) * a["bid"] + w * b["bid"]),
        float((1.0 - w) * a["ask"] + w * b["ask"]),
        float((1.0 - w) * a["mid"] + w * b["mid"]),
    )


def _load_spy_benchmark(con_dolt: sqlite3.Connection) -> BenchmarkSeries:
    # Prefer equity_ohlcv.db adjusted SPY via shared loader, fallback to dolt stocks_close
    if os.path.exists(OHLCV_DB):
        from src.equity_playbooks.backtest import load_equity_ohlcv

        ohlcv = load_equity_ohlcv(OHLCV_DB, symbols=["SPY"])
        if "SPY" in ohlcv:
            ds, cs, _ = ohlcv["SPY"]
            return BenchmarkSeries(ds, cs)
    rows = con_dolt.execute(
        "SELECT date, close FROM stocks_close WHERE symbol='SPY' ORDER BY date"
    ).fetchall()
    return BenchmarkSeries(
        [r[0] for r in rows],
        np.array([float(r[1]) for r in rows], dtype=float),
    )


def _build_events(con: sqlite3.Connection) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    ec = con.execute(
        "SELECT symbol, date, whn FROM earnings_cal WHERE whn IN ('Before market open', 'After market close')"
    ).fetchall()
    snaps = con.execute("SELECT DISTINCT symbol, date FROM dolt_chain").fetchall()
    snap_days: Dict[str, List[dt.date]] = {}
    for sym, d in snaps:
        snap_days.setdefault(sym, []).append(_d(d))
    for sym in snap_days:
        snap_days[sym].sort()

    kept: List[Dict[str, str]] = []
    dropped: Dict[str, int] = {"no_symbol_snaps": 0, "no_t_in": 0, "no_t_out": 0}
    for sym, ev_str, whn in ec:
        ds = snap_days.get(sym)
        if not ds:
            dropped["no_symbol_snaps"] += 1
            continue
        ed = _d(ev_str)
        amc = str(whn).startswith("After")
        # Reaction session D; last safe pre-report close is D_safe = D - 1 (ed for AMC, ed-1 for BMO)
        d_react = ed + dt.timedelta(days=1) if amc else ed
        d_safe = ed if amc else ed - dt.timedelta(days=1)

        # T_in: last snapshot on or before D - 11 within [D - 16, D - 10]
        in_lo = d_react - dt.timedelta(days=16)
        in_hi_pref = d_react - dt.timedelta(days=11)
        in_hi_alt = d_react - dt.timedelta(days=10)
        cand_in = [x for x in ds if in_lo <= x <= in_hi_pref]
        if not cand_in:
            cand_in = [x for x in ds if in_lo <= x <= in_hi_alt]
        if not cand_in:
            dropped["no_t_in"] += 1
            continue
        t_in = cand_in[-1]

        # T_out: last snapshot strictly <= d_safe within [d_react - 3, d_safe]
        out_lo = d_react - dt.timedelta(days=3)
        cand_out = [x for x in ds if out_lo <= x <= d_safe and x > t_in]
        if not cand_out:
            dropped["no_t_out"] += 1
            continue
        t_out = cand_out[-1]

        kept.append(
            {
                "symbol": sym,
                "event_date": ev_str,
                "D": d_react.isoformat(),
                "t_in": t_in.isoformat(),
                "t_out": t_out.isoformat(),
                "week": _iso_week(d_react.isoformat()),
            }
        )
    return kept, dropped


def _cluster_boot(rows: List[Dict], metric: str, is_eq: bool = False) -> Dict[str, object]:
    if not rows:
        return {"n": 0, "weeks": 0, "verdict": "INSUFFICIENT"}
    by_wk: Dict[str, List[Dict]] = {}
    for r in rows:
        by_wk.setdefault(r["week"], []).append(r)
    weeks = sorted(by_wk.keys())
    k = len(weeks)
    vals = np.array([float(r["ret"]) for r in rows], dtype=float)
    exs = np.array([float(r.get("excess", r["ret"])) for r in rows], dtype=float)

    def _pf(a: np.ndarray) -> float:
        loss = float(-np.sum(a[a < 0]))
        gain = float(np.sum(a[a > 0]))
        return gain / loss if loss > 0 else float("inf")

    wk_r = [np.array([float(x["ret"]) for x in by_wk[w]], dtype=float) for w in weeks]
    wk_ex = [np.array([float(x.get("excess", x["ret"])) for x in by_wk[w]], dtype=float) for w in weeks]

    rng = np.random.default_rng(SEED)
    b_r = np.empty(N_BOOT, dtype=float)
    b_ex = np.empty(N_BOOT, dtype=float)
    b_pf = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, k, size=k)
        sr = np.concatenate([wk_r[j] for j in idx])
        sx = np.concatenate([wk_ex[j] for j in idx])
        b_r[i] = float(np.mean(sr))
        b_ex[i] = float(np.mean(sx))
        b_pf[i] = _pf(sr)

    lo_b, hi_b = 100.0 * (ALPHA_BONF / 2.0), 100.0 * (1.0 - ALPHA_BONF / 2.0)
    pf_m = "nearest" if np.any(np.isinf(b_pf)) else "linear"

    r_ci = [float(np.percentile(b_r, lo_b)), float(np.percentile(b_r, hi_b))]
    ex_ci = [float(np.percentile(b_ex, lo_b)), float(np.percentile(b_ex, hi_b))]
    pf_ci = [
        float(np.percentile(b_pf, lo_b, method=pf_m)),
        float(np.percentile(b_pf, hi_b, method=pf_m)),
    ]

    if len(rows) < MIN_EVENTS or k < MIN_WEEKS:
        verdict = "INSUFFICIENT"
    elif is_eq:
        verdict = "PASS" if (ex_ci[0] > 0.0 and pf_ci[0] > 1.0) else "FAIL"
    else:
        verdict = "PASS" if (r_ci[0] > 0.0 and pf_ci[0] > 1.0) else "FAIL"

    by_year: Dict[str, Dict[str, float]] = {}
    for yr in sorted({r["D"][:4] for r in rows}):
        yr_rows = [r for r in rows if r["D"][:4] == yr]
        ya = np.array([float(r["ret"]) for r in yr_rows], dtype=float)
        yx = np.array([float(r.get("excess", r["ret"])) for r in yr_rows], dtype=float)
        by_year[yr] = {
            "n": len(yr_rows),
            "win_rate": round(float(np.mean(ya > 0)), 4),
            "mean_ret": round(float(np.mean(ya)), 5),
            "mean_excess": round(float(np.mean(yx)), 5),
            "pf": round(_pf(ya), 3),
        }

    return {
        "label": metric,
        "n": len(rows),
        "n_symbols": len({r["symbol"] for r in rows}),
        "weeks": k,
        "win_rate": round(float(np.mean(vals > 0)), 4),
        "median_ret": round(float(np.median(vals)), 5),
        "mean_ret": round(float(np.mean(vals)), 5),
        "ret_ci9833": [round(r_ci[0], 5), round(r_ci[1], 5)],
        "mean_excess": round(float(np.mean(exs)), 5),
        "excess_ci9833": [round(ex_ci[0], 5), round(ex_ci[1], 5)],
        "pf": round(_pf(vals), 4),
        "pf_ci9833": [round(pf_ci[0], 4), round(pf_ci[1], 4)],
        "verdict": verdict,
        "by_year": by_year,
    }


def main() -> int:
    con = sqlite3.connect(f"file:{DOLT_DB}?mode=ro", uri=True)
    try:
        spy = _load_spy_benchmark(con)
        events, dropped = _build_events(con)
        print(f"Pre-earnings events matched (strictly closed <= D-1): {len(events)} (dropped={dropped})")

        # Load stocks_close into memory dict
        close_map: Dict[Tuple[str, str], float] = {
            (s, d): float(c)
            for s, d, c in con.execute("SELECT symbol, date, close FROM stocks_close").fetchall()
            if c is not None and float(c) > 0
        }

        eq_rows: List[Dict] = []
        lc_crossed: List[Dict] = []
        lc_mid: List[Dict] = []
        st_crossed: List[Dict] = []
        st_mid: List[Dict] = []

        opt_skips: Dict[str, int] = {}
        chain_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

        def _get_chain(sym: str, day: str) -> pd.DataFrame:
            k = (sym, day)
            if k not in chain_cache:
                chain_cache[k] = pd.read_sql(
                    "SELECT expiration, strike, type, bid, ask, mid, delta "
                    "FROM dolt_chain WHERE symbol=? AND date=?",
                    con,
                    params=(sym, day),
                )
            return chain_cache[k]

        for ev in events:
            sym, d_react, t_in, t_out, wk = ev["symbol"], ev["D"], ev["t_in"], ev["t_out"], ev["week"]

            # 1. O2-EQ (Shares)
            p0 = close_map.get((sym, t_in))
            p1 = close_map.get((sym, t_out))
            b_spy = spy.window_return(t_in, t_out)
            if p0 is not None and p1 is not None and b_spy is not None and p0 > 0:
                r_eq = (p1 / p0 - 1.0) - EQ_RT_FRICTION
                eq_rows.append(
                    {
                        "symbol": sym,
                        "D": d_react,
                        "week": wk,
                        "ret": r_eq,
                        "excess": r_eq - b_spy,
                        "megacap": sym in MEGACAPS,
                    }
                )

            # 2. Option arms (O2-LC, O2-ST)
            c_in = _get_chain(sym, t_in)
            if c_in.empty:
                opt_skips["empty_in"] = opt_skips.get("empty_in", 0) + 1
                continue
            d_in = _d(t_in)
            exps = sorted(
                e for e in c_in["expiration"].unique()
                if e > d_react and (_d(e) - d_in).days <= 45
            )
            if not exps:
                opt_skips["no_expiry"] = opt_skips.get("no_expiry", 0) + 1
                continue
            exp = exps[0]
            e_in = c_in[c_in["expiration"] == exp]
            calls_in = e_in[(e_in["type"] == "call") & e_in["delta"].notna()]
            calls_in = calls_in[[_ok_entry(r) for r in calls_in.itertuples()]] if len(calls_in) else calls_in
            if calls_in.empty:
                opt_skips["no_atm_call"] = opt_skips.get("no_atm_call", 0) + 1
                continue
            atm_c = calls_in.loc[(calls_in["delta"] - 0.50).abs().idxmin()]
            k_atm = float(atm_c["strike"])

            c_out = _get_chain(sym, t_out)
            e_out = c_out[c_out["expiration"] == exp]
            if e_out.empty:
                opt_skips["empty_out_exp"] = opt_skips.get("empty_out_exp", 0) + 1
                continue

            # Exit quote for call
            calls_out = e_out[e_out["type"] == "call"]
            c_exact = calls_out[calls_out["strike"] == k_atm]
            c_interp_flag = False
            if not c_exact.empty:
                c_out_q = (float(c_exact.iloc[0]["bid"]), float(c_exact.iloc[0]["ask"]), float(c_exact.iloc[0]["mid"]))
            else:
                c_out_q = _interp(calls_out, k_atm)
                c_interp_flag = True

            if c_out_q is not None and c_out_q[1] > 0:
                c_in_ask, c_in_mid = float(atm_c["ask"]), float(atm_c["mid"])
                if c_in_ask > 0 and c_in_mid > 0:
                    r_lc_x = (c_out_q[0] - c_in_ask) / c_in_ask
                    r_lc_m = (c_out_q[2] - c_in_mid) / c_in_mid
                    lc_crossed.append(
                        {"symbol": sym, "D": d_react, "week": wk, "ret": r_lc_x, "excess": r_lc_x, "interp": c_interp_flag, "megacap": sym in MEGACAPS}
                    )
                    lc_mid.append(
                        {"symbol": sym, "D": d_react, "week": wk, "ret": r_lc_m, "excess": r_lc_m, "interp": c_interp_flag, "megacap": sym in MEGACAPS}
                    )

            # Put at same ATM strike for Straddle
            puts_in = e_in[(e_in["type"] == "put") & (e_in["strike"] == k_atm)]
            if not puts_in.empty and _ok_entry(puts_in.iloc[0]) and c_out_q is not None and c_out_q[1] > 0:
                atm_p = puts_in.iloc[0]
                puts_out = e_out[e_out["type"] == "put"]
                p_exact = puts_out[puts_out["strike"] == k_atm]
                p_interp_flag = False
                if not p_exact.empty:
                    p_out_q = (float(p_exact.iloc[0]["bid"]), float(p_exact.iloc[0]["ask"]), float(p_exact.iloc[0]["mid"]))
                else:
                    p_out_q = _interp(puts_out, k_atm)
                    p_interp_flag = True

                if p_out_q is not None and p_out_q[1] > 0:
                    st_in_ask = float(atm_c["ask"]) + float(atm_p["ask"])
                    st_in_mid = float(atm_c["mid"]) + float(atm_p["mid"])
                    st_out_bid = c_out_q[0] + p_out_q[0]
                    st_out_mid = c_out_q[2] + p_out_q[2]
                    if st_in_ask > 0 and st_in_mid > 0:
                        r_st_x = (st_out_bid - st_in_ask) / st_in_ask
                        r_st_m = (st_out_mid - st_in_mid) / st_in_mid
                        st_crossed.append(
                            {"symbol": sym, "D": d_react, "week": wk, "ret": r_st_x, "excess": r_st_x, "interp": (c_interp_flag or p_interp_flag), "megacap": sym in MEGACAPS}
                        )
                        st_mid.append(
                            {"symbol": sym, "D": d_react, "week": wk, "ret": r_st_m, "excess": r_st_m, "interp": (c_interp_flag or p_interp_flag), "megacap": sym in MEGACAPS}
                        )
    finally:
        con.close()

    report = {
        "prereg": "docs/PREREG_OPTIONS_DTE_AND_PRE_EARNINGS_20261124.md",
        "events_matched": len(events),
        "events_dropped": dropped,
        "option_skips": opt_skips,
        "primary": {
            "O2_EQ": _cluster_boot(eq_rows, "O2-EQ_shares_10bps", is_eq=True),
            "O2_LC_crossed": _cluster_boot(lc_crossed, "O2-LC_long_call_crossed", is_eq=False),
            "O2_ST_crossed": _cluster_boot(st_crossed, "O2-ST_straddle_crossed", is_eq=False),
        },
        "sensitivities": {
            "O2_LC_mid": _cluster_boot(lc_mid, "O2-LC_mid", is_eq=False),
            "O2_ST_mid": _cluster_boot(st_mid, "O2-ST_mid", is_eq=False),
            "O2_LC_crossed_exact_only": _cluster_boot([r for r in lc_crossed if not r["interp"]], "O2-LC_crossed_exact", is_eq=False),
            "O2_ST_crossed_exact_only": _cluster_boot([r for r in st_crossed if not r["interp"]], "O2-ST_crossed_exact", is_eq=False),
            "O2_EQ_megacap": _cluster_boot([r for r in eq_rows if r["megacap"]], "O2-EQ_megacap", is_eq=True),
            "O2_EQ_non_megacap": _cluster_boot([r for r in eq_rows if not r["megacap"]], "O2-EQ_non_megacap", is_eq=True),
            "O2_ST_crossed_megacap": _cluster_boot([r for r in st_crossed if r["megacap"]], "O2-ST_crossed_megacap", is_eq=False),
        },
    }

    print("\n=== TEST O2 PRIMARY (T-10 -> T-1, CLOSED BEFORE REPORT) ===")
    for k, s in report["primary"].items():  # type: ignore[union-attr]
        print(
            f"  {k:16s}: n={s['n']:4d} wks={s['weeks']:3d} win={s['win_rate']*100:5.1f}% "
            f"mean={s['mean_ret']*100:+6.2f}% {s['ret_ci9833']} "
            f"ex={s['mean_excess']*100:+6.2f}% {s['excess_ci9833']} "
            f"PF={s['pf']:.3f} {s['pf_ci9833']} -> {s['verdict']}"
        )
    print("\n=== TEST O2 SENSITIVITIES ===")
    for k, s in report["sensitivities"].items():  # type: ignore[union-attr]
        print(
            f"  {k:26s}: n={s['n']:4d} wks={s['weeks']:3d} win={s['win_rate']*100:5.1f}% "
            f"mean={s['mean_ret']*100:+6.2f}% {s['ret_ci9833']} PF={s['pf']:.3f}"
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
