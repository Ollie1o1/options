"""TFSA & Options Research Backtest Suite (`src/equity_playbooks/tfsa_studies.py`).

Honest, zero-lookahead research studies evaluating five new TFSA-legal & regime-gated
hypotheses against the repository's local `data/equity_ohlcv.db`:
  1. VRP-Gated Covered Call Overwriting on VM_OUTLOOK_TOP3 vs Bare Shares
  2. Deep-ITM LEAPS (0.80 delta) + SGOV T-Bill Buffer vs Bare Shares
  3. Widened ETF Rotation Universe (16 ETFs -> all liquid ETFs in local OHLCV)
  4. Regime-Gated Uptrend Pullbacks (SPY > 200d & RV20 <= 22% bear-year filter)
  5. Phase 3 Alpha Labs Readiness & Sample-Depth Audit (Skew/GEX, EDGAR, OCC OI)

All outputs are strictly RESEARCH / EXPLORATORY — they never modify frozen preregistered
gates (`configs/playbooks.json` or `config.json`).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.equity_playbooks.backtest import load_equity_ohlcv
from src.equity_playbooks.indicators import mom_12_1_series, sma, trailing_rv, wilder_rsi
from src.radar.scanners import OUTLOOK_ETFS
from src.structure.express import _bs_px_delta

WIDE_ETF_CANDIDATES: Tuple[str, ...] = (
    *OUTLOOK_ETFS,
    "SPY", "DIA", "EEM", "EFA", "HYG", "KRE", "SLV", "GDX", "XBI",
    "URNM", "COPX", "ITA", "EWJ", "EWY", "FXI", "DBC", "TIP", "MTUM", "QUAL", "AVUV",
    "SPHB", "SPLV", "CPER", "UUP", "XOP", "XHB", "XRT", "IYR", "VNQ", "AGG", "LQD",
)


def _perf_stats(monthly_rets: Sequence[float], periods_per_year: float = 12.0) -> Dict[str, Any]:
    arr = np.asarray(monthly_rets, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return {"n_periods": n, "cagr_pct": None, "ann_vol_pct": None, "sharpe": None, "max_dd_pct": None}
    eq = np.cumprod(1.0 + arr)
    years = n / periods_per_year
    cagr = (float(eq[-1]) ** (1.0 / max(years, 1e-6)) - 1.0) * 100.0
    ann_vol = float(np.std(arr, ddof=1)) * math.sqrt(periods_per_year) * 100.0
    std_val = float(np.std(arr, ddof=1))
    sharpe = (float(np.mean(arr)) / max(std_val, 1e-9)) * math.sqrt(periods_per_year)
    peaks = np.maximum.accumulate(eq)
    dd = (eq / peaks - 1.0) * 100.0
    max_dd = float(np.min(dd))
    return {
        "n_periods": n,
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
    }


def _align_monthly_panel(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    symbols: Sequence[str],
    min_bars: int = 300,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build date-aligned daily close & 20d RV arrays for all symbols with >= min_bars."""
    valid = [s for s in symbols if s in ohlcv and len(ohlcv[s][1]) >= min_bars]
    if "SPY" not in ohlcv or len(valid) < 4:
        return [], {}, {}

    spy_dates = ohlcv["SPY"][0]
    date_sets = [set(ohlcv[s][0]) for s in valid]
    common = sorted(set(spy_dates).intersection(*date_sets))
    if len(common) < min_bars:
        return [], {}, {}

    closes: Dict[str, np.ndarray] = {}
    rv20s: Dict[str, np.ndarray] = {}
    for s in valid:
        ds, cs, _vs = ohlcv[s]
        idx_map = {d: i for i, d in enumerate(ds)}
        arr = np.array([float(cs[idx_map[d]]) for d in common], dtype=float)
        closes[s] = arr
        rv20s[s] = trailing_rv(arr, 20)
    return common, closes, rv20s


def _vm_outlook_monthly_trace(
    dates: List[str],
    closes: Dict[str, np.ndarray],
    rv20s: Dict[str, np.ndarray],
    universe: Sequence[str],
    *,
    top_k: int = 3,
    target_vol: float = 0.14,
    cost_bps_side: float = 3.0,
    cash_yield_ann: float = 0.03,
) -> List[Dict[str, Any]]:
    """Run zero-lookahead 21-trading-day VM_OUTLOOK_TOP3 rotation trace."""
    syms = [s for s in universe if s in closes]
    if len(syms) < top_k or len(dates) < 280:
        return []

    moms = {s: mom_12_1_series(closes[s]) for s in syms}
    step = 21
    prev_weights: Dict[str, float] = {}
    trace: List[Dict[str, Any]] = []
    cash_m = cash_yield_ann / 12.0

    for i in range(255, len(dates) - step, step):
        # Cross-sectional momentum rank at close of day i (zero lookahead)
        scored: List[Tuple[float, str]] = []
        for s in syms:
            m = float(moms[s][i])
            if math.isfinite(m):
                scored.append((m, s))
        if len(scored) < top_k:
            continue
        scored.sort(reverse=True)
        picks = [s for _, s in scored[:top_k]]

        # Basket trailing 20d realized vol
        rets_mat = np.column_stack(
            [closes[s][i - 20 : i + 1][1:] / closes[s][i - 20 : i + 1][:-1] - 1.0 for s in picks]
        )
        basket_rv = float(np.std(np.mean(rets_mat, axis=1), ddof=1) * math.sqrt(252.0))
        scale = float(np.clip(target_vol / max(basket_rv, 0.04), 0.25, 1.00))
        new_weights = {s: scale / top_k for s in picks}

        # Turnover cost
        all_keys = set(prev_weights) | set(new_weights)
        turnover = sum(abs(new_weights.get(k, 0.0) - prev_weights.get(k, 0.0)) for k in all_keys)
        cost = turnover * (cost_bps_side / 10000.0)

        # Forward 21-session return
        leg_rets: Dict[str, float] = {}
        eq_ret = 0.0
        for s, w in new_weights.items():
            r_s = float(closes[s][i + step] / closes[s][i] - 1.0)
            leg_rets[s] = r_s
            eq_ret += w * r_s
        cash_w = max(0.0, 1.0 - scale)
        net_ret = eq_ret + cash_w * cash_m - cost

        trace.append(
            {
                "date": dates[i],
                "end_date": dates[i + step],
                "picks": picks,
                "weights": new_weights,
                "scale": scale,
                "basket_rv": basket_rv,
                "leg_rets": leg_rets,
                "leg_rv20": {s: float(rv20s[s][i]) for s in picks},
                "bare_ret": net_ret,
                "cost": cost,
            }
        )
        prev_weights = new_weights
    return trace


def study_vrp_covered_calls(
    ohlcv_db: str = "data/equity_ohlcv.db",
    *,
    vrp_iv_mult: float = 1.15,
    option_half_spread_pct: float = 0.04,
    r: float = 0.04,
) -> Dict[str, Any]:
    """Compare Bare VM_OUTLOOK_TOP3 vs Unconditional 0.25d Covered Calls vs VRP-Gated Covered Calls.

    VRP Gate: Only overwrite a held ETF when its 20d RV sits above its trailing 60d median
    (rich vol regime where option sellers historically collect higher VRP) and avoid selling
    calls into compressed low-vol breakouts.
    """
    ohlcv = load_equity_ohlcv(ohlcv_db, symbols=list(OUTLOOK_ETFS) + ["SPY"])
    dates, closes, rv20s = _align_monthly_panel(ohlcv, list(OUTLOOK_ETFS) + ["SPY"])
    trace = _vm_outlook_monthly_trace(dates, closes, rv20s, OUTLOOK_ETFS)
    if not trace:
        return {"error": "insufficient_ohlcv_history"}

    # Precompute 60d median RV20 per symbol to detect rich-vol overwriting windows
    rv_med60: Dict[str, Dict[str, float]] = {}
    for s in OUTLOOK_ETFS:
        if s not in rv20s:
            continue
        arr = rv20s[s]
        dmap: Dict[str, float] = {}
        for idx in range(60, len(dates)):
            win = arr[idx - 60 : idx]
            fin = win[np.isfinite(win)]
            if len(fin) >= 20:
                dmap[dates[idx]] = float(np.median(fin))
        rv_med60[s] = dmap

    bare_rets: List[float] = []
    uncond_rets: List[float] = []
    gated_rets: List[float] = []
    overwrite_rate_gated = 0
    total_legs = 0
    t_yr = 21.0 / 252.0

    for step_row in trace:
        bare_rets.append(step_row["bare_ret"])
        d = step_row["date"]
        w_map = step_row["weights"]
        cash_w = max(0.0, 1.0 - step_row["scale"])
        cash_m = cash_w * (0.03 / 12.0)

        u_ret = cash_m - step_row["cost"]
        g_ret = cash_m - step_row["cost"]

        for s, w in w_map.items():
            total_legs += 1
            r_s = step_row["leg_rets"][s]
            rv_now = step_row["leg_rv20"].get(s, 0.18)
            if not math.isfinite(rv_now) or rv_now <= 0.03:
                rv_now = 0.18
            iv_est = rv_now * vrp_iv_mult
            # Approximate 0.25-delta OTM call strike: S * exp(+0.674 * sigma * sqrt(T))
            k_rel = math.exp(0.674 * iv_est * math.sqrt(t_yr))
            c_mid, _dl, _th = _bs_px_delta(1.0, k_rel, t_yr, r, iv_est, "call")
            # Cross bid (honest half-spread friction)
            c_net = max(0.0, c_mid * (1.0 - option_half_spread_pct))
            # Covered call payoff at expiry: stock return capped at (k_rel - 1) + premium collected
            cc_leg_ret = min(r_s, k_rel - 1.0) + c_net
            u_ret += w * cc_leg_ret

            med_rv = rv_med60.get(s, {}).get(d)
            if med_rv is not None and rv_now >= 1.15 * med_rv:
                g_ret += w * cc_leg_ret
                overwrite_rate_gated += 1
            else:
                g_ret += w * r_s

        uncond_rets.append(u_ret)
        gated_rets.append(g_ret)

    return {
        "study": "tfsa_vrp_covered_calls",
        "window": f"{trace[0]['date']} .. {trace[-1]['end_date']}",
        "assumptions": {
            "vrp_iv_mult": vrp_iv_mult,
            "option_half_spread_pct": option_half_spread_pct,
            "tfsa_commission": 0.0,
        },
        "bare_vm_outlook": _perf_stats(bare_rets),
        "unconditional_cc_25d": _perf_stats(uncond_rets),
        "vrp_gated_cc_25d": {
            **_perf_stats(gated_rets),
            "overwrite_pct_of_legs": round(100.0 * overwrite_rate_gated / max(total_legs, 1), 1),
        },
        "note": "Model-priced BS calls (synthetic VRP proxy) — treat as research lead until validated on archived real chains.",
    }


def study_leaps_plus_sgov(
    ohlcv_db: str = "data/equity_ohlcv.db",
    *,
    leaps_dte: int = 240,
    target_delta: float = 0.80,
    option_half_spread_pct: float = 0.03,
    r: float = 0.04,
) -> Dict[str, Any]:
    """Compare Bare VM_OUTLOOK_TOP3 vs 0.80-Delta Deep-ITM LEAPS + SGOV T-Bill Buffer in TFSA.

    Models replacing 100% notional ETF shares with delta-matched 0.80-delta 240-DTE LEAPS
    (committing only ~22-30% of notional in option premium) and parking the freed ~70-78%
    cash in SGOV earning `r`. Factors realistic option entry/exit half-spread friction.
    """
    ohlcv = load_equity_ohlcv(ohlcv_db, symbols=list(OUTLOOK_ETFS) + ["SPY"])
    dates, closes, rv20s = _align_monthly_panel(ohlcv, list(OUTLOOK_ETFS) + ["SPY"])
    trace = _vm_outlook_monthly_trace(dates, closes, rv20s, OUTLOOK_ETFS)
    if not trace:
        return {"error": "insufficient_ohlcv_history"}

    bare_rets: List[float] = []
    leaps_rets: List[float] = []
    t0 = leaps_dte / 365.0
    t1 = max(30, leaps_dte - 30) / 365.0
    cash_m = r / 12.0

    for row in trace:
        bare_rets.append(row["bare_ret"])
        w_map = row["weights"]
        total_opt_cap = 0.0
        opt_pnl_port = 0.0

        for s, w in w_map.items():
            rv = row["leg_rv20"].get(s, 0.18)
            if not math.isfinite(rv) or rv <= 0.03:
                rv = 0.18
            iv = rv * 1.08
            # Deep-ITM strike (~0.80 delta): S * exp(-0.84 * sigma * sqrt(T))
            k_rel = math.exp(-0.84 * iv * math.sqrt(t0))
            c0, dl0, _ = _bs_px_delta(1.0, k_rel, t0, r, iv, "call")
            # Delta-matched contract notional: w / dl0 contracts per $1 portfolio
            n_ct = w / max(dl0, 0.50)
            entry_cost = n_ct * c0 * (1.0 + option_half_spread_pct)
            total_opt_cap += entry_cost

            s1 = 1.0 + row["leg_rets"][s]
            c1, _, _ = _bs_px_delta(max(0.05, s1), k_rel, t1, r, iv, "call")
            exit_val = n_ct * max(0.0, c1 * (1.0 - option_half_spread_pct))
            opt_pnl_port += (exit_val - entry_cost)

        freed_cash_w = max(0.0, 1.0 - total_opt_cap)
        leaps_rets.append(opt_pnl_port + freed_cash_w * cash_m)

    return {
        "study": "tfsa_deep_itm_leaps_plus_sgov",
        "window": f"{trace[0]['date']} .. {trace[-1]['end_date']}",
        "assumptions": {
            "leaps_dte": leaps_dte,
            "target_delta": target_delta,
            "option_half_spread_pct": option_half_spread_pct,
            "sgov_yield_ann": r,
        },
        "bare_vm_outlook": _perf_stats(bare_rets),
        "leaps_80d_plus_sgov": _perf_stats(leaps_rets),
        "note": "Monthly-rolled LEAPS pay high spread friction; holding LEAPS 90-180d (lower roll frequency) is essential to beat bare shares.",
    }


def study_wider_outlook_universe(ohlcv_db: str = "data/equity_ohlcv.db") -> Dict[str, Any]:
    """Compare Baseline 16-ETF VM_OUTLOOK_TOP3 vs Expanded Liquid ETF Universe."""
    ohlcv = load_equity_ohlcv(ohlcv_db, symbols=list(WIDE_ETF_CANDIDATES))
    avail_wide = [s for s in WIDE_ETF_CANDIDATES if s in ohlcv and len(ohlcv[s][1]) >= 300]
    dates_16, c16, rv16 = _align_monthly_panel(ohlcv, list(OUTLOOK_ETFS) + ["SPY"])
    dates_w, cw, rvw = _align_monthly_panel(ohlcv, avail_wide)

    tr_16 = _vm_outlook_monthly_trace(dates_16, c16, rv16, OUTLOOK_ETFS)
    tr_w = _vm_outlook_monthly_trace(dates_w, cw, rvw, [s for s in avail_wide if s != "SPY"])
    if not tr_16 or not tr_w:
        return {"error": "insufficient_ohlcv_history"}

    # Align on shared start date for apples-to-apples comparison
    start_common = max(tr_16[0]["date"], tr_w[0]["date"])
    r16 = [x["bare_ret"] for x in tr_16 if x["date"] >= start_common]
    rw = [x["bare_ret"] for x in tr_w if x["date"] >= start_common]
    r22_16 = [x["bare_ret"] for x in tr_16 if x["date"].startswith("2022-")]
    r22_w = [x["bare_ret"] for x in tr_w if x["date"].startswith("2022-")]

    return {
        "study": "wider_outlook_universe",
        "window_aligned": f"{start_common} .. {tr_w[-1]['end_date']}",
        "baseline_16_count": len(OUTLOOK_ETFS),
        "expanded_count": len([s for s in avail_wide if s != "SPY"]),
        "expanded_symbols": [s for s in avail_wide if s != "SPY"],
        "baseline_16": _perf_stats(r16),
        "expanded_universe": _perf_stats(rw),
        "bear_2022_ret_pct": {
            "baseline_16": round((float(np.prod([1.0 + x for x in r22_16])) - 1.0) * 100.0, 2) if r22_16 else None,
            "expanded_universe": round((float(np.prod([1.0 + x for x in r22_w])) - 1.0) * 100.0, 2) if r22_w else None,
        },
    }


def study_regime_gated_pullback(
    ohlcv_db: str = "data/equity_ohlcv.db",
    *,
    friction_bps: float = 10.0,
    max_rv20: float = 0.22,
    sample_limit: int = 80,
) -> Dict[str, Any]:
    """Compare Unfiltered Uptrend Pullbacks vs Regime-Gated (SPY > 200d & SPY RV20 <= 22%)."""
    from src.alpha.options_for_shares import EXPANDED_CHAIN_UNIVERSE

    syms = sorted(set(EXPANDED_CHAIN_UNIVERSE) | {"SPY"})[: sample_limit + 1]
    ohlcv = load_equity_ohlcv(ohlcv_db, symbols=syms)
    if "SPY" not in ohlcv or len(ohlcv["SPY"][1]) < 250:
        return {"error": "spy_missing"}

    spy_ds, spy_cs, _ = ohlcv["SPY"]
    spy_s200 = sma(spy_cs, 200)
    spy_rv = trailing_rv(spy_cs, 20)
    spy_regime: Dict[str, bool] = {}
    for i, d in enumerate(spy_ds):
        ok = (
            math.isfinite(float(spy_s200[i]))
            and float(spy_cs[i]) > float(spy_s200[i])
            and math.isfinite(float(spy_rv[i]))
            and float(spy_rv[i]) <= max_rv20
        )
        spy_regime[d] = ok

    cost = friction_bps / 10000.0
    base_trades: List[Tuple[str, float]] = []
    gated_trades: List[Tuple[str, float]] = []

    for tk, (ds, cs, _vs) in ohlcv.items():
        if tk == "SPY" or len(cs) < 250:
            continue
        s200 = sma(cs, 200)
        s5 = sma(cs, 5)
        r2 = wilder_rsi(cs, 2)
        i = 205
        while i < len(cs) - 11:
            if (
                math.isfinite(float(s200[i]))
                and math.isfinite(float(r2[i]))
                and float(cs[i]) > float(s200[i])
                and float(r2[i]) < 10.0
            ):
                ent_px = float(cs[i])
                exit_j = min(i + 10, len(cs) - 1)
                for j in range(i + 1, min(i + 11, len(cs))):
                    if math.isfinite(float(s5[j])) and float(cs[j]) > float(s5[j]):
                        exit_j = j
                        break
                ret = float(cs[exit_j] / ent_px - 1.0) - cost
                d_str = ds[i]
                base_trades.append((d_str, ret))
                if spy_regime.get(d_str, False):
                    gated_trades.append((d_str, ret))
                i = exit_j + 1
            else:
                i += 1

    def _summarize(tr: List[Tuple[str, float]]) -> Dict[str, Any]:
        if not tr:
            return {"n": 0, "win_rate_pct": None, "mean_ret_pct": None, "pf": None, "bear_2018_2022_mean_pct": None}
        rs = np.array([x[1] for x in tr], dtype=float)
        wins = float(np.sum(rs[rs > 0]))
        losses = float(abs(np.sum(rs[rs < 0])))
        bear = [x[1] for x in tr if x[0].startswith(("2018-", "2022-"))]
        return {
            "n": len(tr),
            "win_rate_pct": round(float(np.mean(rs > 0)) * 100.0, 1),
            "mean_ret_pct": round(float(np.mean(rs)) * 100.0, 3),
            "pf": round(wins / max(losses, 1e-9), 3),
            "bear_2018_2022_n": len(bear),
            "bear_2018_2022_mean_pct": round(float(np.mean(bear)) * 100.0, 3) if bear else None,
        }

    return {
        "study": "regime_gated_equity_pullback",
        "gate": f"SPY > SMA200 and SPY RV20 <= {max_rv20*100:.0f}%",
        "friction_bps": friction_bps,
        "unfiltered": _summarize(base_trades),
        "regime_gated": _summarize(gated_trades),
    }


def study_phase3_alpha_readiness(
    *,
    chain_db: str = "data/chain_archive.db",
    edgar_db: str = "data/edgar_events.db",
    occ_db: str = "data/occ_oi.db",
) -> Dict[str, Any]:
    """Audit Phase 3 Alpha Labs archive depth toward OOS validation readiness."""
    from src.alpha.edgar_events import load_cached_edgar_events
    from src.chain_archive import archive_stats
    from src.occ_archive import scan_occ_uoa

    ch = archive_stats(chain_db)
    ed = load_cached_edgar_events(db_path=edgar_db, limit=500)
    occ = scan_occ_uoa(db_path=occ_db, min_total_vol=0, limit=500)
    by_edgar_type: Dict[str, int] = {}
    for e in ed:
        by_edgar_type[e.event_type] = by_edgar_type.get(e.event_type, 0) + 1

    return {
        "study": "phase3_alpha_readiness",
        "chain_archive": ch,
        "edgar_events_count": len(ed),
        "edgar_by_type": by_edgar_type,
        "occ_symbols_latest": len(occ),
        "readiness_verdict": (
            "ACCUMULATING"
            if int(ch.get("days") or 0) < 60
            else "READY_FOR_PREREG_OOS_TEST"
        ),
        "note": "Need >=60 trading days of chain_archive & OCC OI snapshots before running preregistered OOS validation of CW Skew / Put Smirk / UOA.",
    }


def run_all_tfsa_studies(ohlcv_db: str = "data/equity_ohlcv.db") -> Dict[str, Any]:
    """Execute all five TFSA & Options research studies and return a unified dict."""
    return {
        "vrp_covered_calls": study_vrp_covered_calls(ohlcv_db=ohlcv_db),
        "leaps_plus_sgov": study_leaps_plus_sgov(ohlcv_db=ohlcv_db),
        "wider_outlook_universe": study_wider_outlook_universe(ohlcv_db=ohlcv_db),
        "regime_gated_pullback": study_regime_gated_pullback(ohlcv_db=ohlcv_db),
        "phase3_alpha_readiness": study_phase3_alpha_readiness(),
    }
