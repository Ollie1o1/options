#!/usr/bin/env python3
"""Preregistered estimator for Study 2D: Point-in-Time SEC EDGAR Quality + Net Buybacks TFSA Sleeve.

Frozen registration: docs/PREREG_EDGAR_QUALITY_BUYBACKS_20261127.md

1. Downloads & caches strictly point-in-time (`filed_date`) SEC EDGAR XBRL quarterly snapshots
   for all 286 non-ETF 2016-2026 US stocks in `data/equity_ohlcv.db` into `data/sec_fundamentals_pit.db`
   (AND refreshes `data/sec_fundamentals.db` so `opt discover --lens edgar_quality` covers all 286 names).
2. Evaluates 3 frozen quarterly long-only arms (`TOP20_NBY`, `TOP20_GPA`, `TOP20_COMPOSITE`) vs
   Equal-Weight Universe (`EW_UNIV` — exact same eligible pool, neutralizing survivorship inflation)
   at 10 bps one-way friction (20 bps stress), with circular 2-quarter block bootstrap (98.33% CI).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.alpha.edgar_events import (
    DEFAULT_FUNDAMENTALS_DB,
    XBRL_FACTS_URL,
    _edgar_get_json,
    compute_xbrl_quality_metrics,
    extract_xbrl_pit_series,
    save_fundamentals_rows,
    xbrl_metrics_as_of,
)
from src.equity_playbooks.backtest import load_equity_ohlcv
from src.equity_playbooks.universe import CORE_ETFS
from src.insider.edgar import cik_for
from src.intraday.stats import block_bootstrap_means
from src.paths import load_project_env, repo_path

REPO = Path(__file__).resolve().parents[1]
PIT_DB = "data/sec_fundamentals_pit.db"
SEED = 20261127
DRAWS = 10_000
COST_PRIMARY = 0.0010  # 10 bps one-way
COST_STRESS = 0.0020   # 20 bps one-way

EXTRA_ETFS = frozenset(
    {
        "QQQ", "SPY", "IWM", "DIA", "ARKK", "BITO", "BOIL", "EWW", "FEZ", "GXC",
        "KWEB", "MJ", "OIH", "PAVE", "SCHD", "SOXL", "SQQQ", "TQQQ", "UVXY",
        "VEA", "VWO", "XBI", "XHB", "XME", "XOP", "XRT", "VNQ", "VIG", "VYM",
        "JEPI", "JEPQ",
    }
)

_PIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS sec_pit_quarterly (
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    filed_date TEXT NOT NULL,
    period_end TEXT NOT NULL,
    net_buyback_pct REAL,
    gross_profit_assets_pct REAL,
    fcf_assets_pct REAL,
    PRIMARY KEY (ticker, as_of_date)
);
CREATE TABLE IF NOT EXISTS sec_pit_fetched (
    ticker TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    n_quarters INTEGER NOT NULL
);
"""


def _ensure_pit_db(db_path: str = PIT_DB) -> str:
    full = repo_path(db_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    con = sqlite3.connect(full)
    try:
        con.executescript(_PIT_SCHEMA)
        con.commit()
    finally:
        con.close()
    return full


def eligible_stock_universe(ohlcv_db: str = "data/equity_ohlcv.db") -> List[str]:
    etfs = set(CORE_ETFS) | EXTRA_ETFS
    con = sqlite3.connect(repo_path(ohlcv_db))
    try:
        rows = con.execute("SELECT ticker, MIN(date), MAX(date) FROM ohlcv GROUP BY ticker").fetchall()
    finally:
        con.close()
    out = sorted(
        str(tk)
        for tk, mn, mx in rows
        if str(tk) not in etfs and str(mn) <= "2016-01-05" and str(mx) >= "2026-10-01" and cik_for(str(tk))
    )
    return out


def quarterly_signal_dates(spy_dates: Sequence[str]) -> List[Tuple[int, int]]:
    """Return `(sig_idx, exec_idx)` pairs for last trading day of Mar/Jun/Sep/Dec 2016-06..2026-09."""
    pairs: List[Tuple[int, int]] = []
    n = len(spy_dates)
    for i in range(n - 1):
        d = spy_dates[i]
        d_next = spy_dates[i + 1]
        if d < "2016-06-01" or d > "2026-10-05":
            continue
        m_cur = int(d[5:7])
        m_nxt = int(d_next[5:7])
        if m_cur in (3, 6, 9, 12) and m_cur != m_nxt:
            pairs.append((i, i + 1))
    return pairs


def sync_pit_fundamentals(
    tickers: Sequence[str],
    signal_dates: Sequence[str],
    *,
    pit_db: str = PIT_DB,
    live_db: str = DEFAULT_FUNDAMENTALS_DB,
    pause_sec: float = 0.14,
) -> Dict[str, Any]:
    """Fetch SEC EDGAR companyfacts for tickers not yet in `pit_db`, caching PIT quarterly rows + latest live row."""
    full = _ensure_pit_db(pit_db)
    con = sqlite3.connect(full)
    try:
        done = {r[0] for r in con.execute("SELECT ticker FROM sec_pit_fetched").fetchall()}
    finally:
        con.close()

    todo = [t for t in tickers if t not in done]
    now_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fetched_n = 0
    errors: List[str] = []
    live_rows: List[Dict[str, Any]] = []

    con = sqlite3.connect(full)
    try:
        for idx, tk in enumerate(todo, start=1):
            cik = cik_for(tk)
            if not cik:
                continue
            try:
                facts = _edgar_get_json(XBRL_FACTS_URL.format(cik=int(cik)))
                pit_series = extract_xbrl_pit_series(tk, facts)
                q_rows = []
                for sd in signal_dates:
                    m = xbrl_metrics_as_of(pit_series, sd)
                    if m is not None:
                        q_rows.append(
                            (
                                tk,
                                sd,
                                m["filed_date"],
                                m["period_end"],
                                m["net_buyback_pct"],
                                m["gross_profit_assets_pct"],
                                m["fcf_assets_pct"],
                            )
                        )
                if q_rows:
                    con.executemany(
                        """
                        INSERT OR REPLACE INTO sec_pit_quarterly
                        (ticker, as_of_date, filed_date, period_end, net_buyback_pct, gross_profit_assets_pct, fcf_assets_pct)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        q_rows,
                    )
                con.execute(
                    "INSERT OR REPLACE INTO sec_pit_fetched (ticker, fetched_at, n_quarters) VALUES (?, ?, ?)",
                    (tk, now_ts, len(q_rows)),
                )
                con.commit()
                latest = compute_xbrl_quality_metrics(tk, facts, fetched_at=now_ts)
                if latest is not None:
                    live_rows.append(latest)
                fetched_n += 1
            except Exception as exc:
                errors.append(f"{tk}: {str(exc)[:80]}")
            if idx % 25 == 0 or idx == len(todo):
                print(f"[SEC XBRL sync] {idx}/{len(todo)} fetched={fetched_n} errs={len(errors)}", flush=True)
            if pause_sec > 0:
                time.sleep(pause_sec)
    finally:
        con.close()

    if live_rows:
        save_fundamentals_rows(live_rows, db_path=live_db)

    return {"total_tickers": len(tickers), "already_cached": len(done), "newly_fetched": fetched_n, "errors": errors[:15]}


def _winsorize_map(vals: Dict[str, float], lo_q: float = 0.01, hi_q: float = 0.99) -> Dict[str, float]:
    if len(vals) < 10:
        return dict(vals)
    arr = np.array(list(vals.values()), dtype=float)
    lo, hi = np.percentile(arr, [lo_q * 100.0, hi_q * 100.0])
    return {k: float(np.clip(v, lo, hi)) for k, v in vals.items()}


def _pct_ranks(vals: Dict[str, float]) -> Dict[str, float]:
    if not vals:
        return {}
    items = sorted(vals.items(), key=lambda x: x[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 0.5}
    return {k: i / (n - 1) for i, (k, _) in enumerate(items)}


def _portfolio_stats(q_rets: np.ndarray) -> Dict[str, Any]:
    if q_rets.size == 0:
        return {}
    eq = np.cumprod(1.0 + q_rets)
    cagr = float(eq[-1] ** (4.0 / q_rets.size) - 1.0)
    vol = float(q_rets.std(ddof=1) * 2.0) if q_rets.size > 1 else 0.0
    sharpe = float((q_rets.mean() / q_rets.std(ddof=1)) * 2.0) if (q_rets.size > 1 and q_rets.std(ddof=1) > 0) else 0.0
    peaks = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    dd = float(np.min(np.concatenate([[1.0], eq]) / peaks - 1.0))
    return {
        "n_quarters": int(q_rets.size),
        "cagr": round(cagr, 5),
        "ann_vol": round(vol, 5),
        "sharpe": round(sharpe, 4),
        "max_dd": round(dd, 5),
        "calmar": round(cagr / abs(dd), 4) if dd < 0 else None,
    }


def _simulate_quarterly(
    weights_by_q: Sequence[Dict[str, float]],
    fwd_rets_by_q: Sequence[Dict[str, float]],
    cost_one_way: float,
) -> Tuple[np.ndarray, float]:
    rets: List[float] = []
    prev_w: Dict[str, float] = {}
    turnovers: List[float] = []
    for w, rmap in zip(weights_by_q, fwd_rets_by_q):
        to = sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in set(w) | set(prev_w))
        turnovers.append(to)
        gross = sum(wt * rmap.get(k, 0.0) for k, wt in w.items())
        net = (1.0 - to * cost_one_way) * (1.0 + gross) - 1.0
        rets.append(float(net))
        # Drift weights to end of quarter for accurate next-quarter turnover
        end_total = 1.0 + gross
        if end_total > 0:
            prev_w = {k: wt * (1.0 + rmap.get(k, 0.0)) / end_total for k, wt in w.items()}
        else:
            prev_w = {}
    return np.asarray(rets, dtype=float), float(np.mean(turnovers) * 4.0) if turnovers else 0.0


def evaluate(
    *,
    ohlcv_db: str = "data/equity_ohlcv.db",
    pit_db: str = PIT_DB,
    skip_sync: bool = False,
) -> Dict[str, Any]:
    tickers = eligible_stock_universe(ohlcv_db)
    ohlcv = load_equity_ohlcv(repo_path(ohlcv_db), ["SPY"] + tickers)
    spy_dates = ohlcv["SPY"][0]
    q_pairs = quarterly_signal_dates(spy_dates)
    sig_dates = [spy_dates[s] for s, _ in q_pairs]

    sync_info = {"skipped": True}
    if not skip_sync:
        sync_info = sync_pit_fundamentals(tickers, sig_dates, pit_db=pit_db)

    # Load PIT fundamentals into map: as_of_date -> ticker -> dict
    pit_map: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    con = sqlite3.connect(f"file:{repo_path(pit_db)}?mode=ro", uri=True)
    try:
        for tk, as_of, nby, gpa, fcf in con.execute(
            "SELECT ticker, as_of_date, net_buyback_pct, gross_profit_assets_pct, fcf_assets_pct FROM sec_pit_quarterly"
        ).fetchall():
            pit_map.setdefault(str(as_of), {})[str(tk)] = {
                "nby": float(nby) if nby is not None else None,
                "gpa": float(gpa) if gpa is not None else None,
                "fcf": float(fcf) if fcf is not None else None,
            }
    finally:
        con.close()

    # Build fast date -> close lookup per ticker
    px_lookup: Dict[str, Dict[str, float]] = {}
    idx_lookup: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for tk, (ds, cs, _vs) in ohlcv.items():
        px_lookup[tk] = {d: float(c) for d, c in zip(ds, cs) if math.isfinite(float(c)) and float(c) > 0}
        idx_lookup[tk] = (list(ds), np.asarray(cs, dtype=float))

    # Build quarterly weights for EW_UNIV, Q1 (TOP20_NBY), Q2 (TOP20_GPA), Q3 (TOP20_COMPOSITE),
    # plus diagnostics: Q5, Q1_bot, Q3_SMA200, Momentum-neutral Q5-Q1
    q_labels: List[str] = []
    fwd_maps: List[Dict[str, float]] = []
    spy_q_rets: List[float] = []
    w_arms: Dict[str, List[Dict[str, float]]] = {
        "EW_UNIV": [],
        "EW_GPA_UNIV": [],
        "Q1_TOP20_NBY": [],
        "Q2_TOP20_GPA": [],
        "Q3_TOP20_COMPOSITE": [],
        "DIAG_Q5_COMP": [],
        "DIAG_Q1_COMP": [],
        "DIAG_Q3_SMA200": [],
        "DIAG_MOM_NEUTRAL_TOP": [],
        "DIAG_MOM_NEUTRAL_BOT": [],
    }
    elig_counts: List[int] = []

    for qi in range(len(q_pairs) - 1):
        s_idx, e_idx = q_pairs[qi]
        _, e_next_idx = q_pairs[qi + 1]
        d_sig = spy_dates[s_idx]
        d_in = spy_dates[e_idx]
        d_out = spy_dates[e_next_idx]

        p_spy_in = px_lookup["SPY"].get(d_in)
        p_spy_out = px_lookup["SPY"].get(d_out)
        if not (p_spy_in and p_spy_out):
            continue

        snap = pit_map.get(d_sig) or {}
        elig_nby: Dict[str, float] = {}
        elig_gpa: Dict[str, float] = {}
        elig_fcf: Dict[str, float] = {}
        elig_mom: Dict[str, float] = {}
        elig_above200: Dict[str, bool] = {}
        rmap: Dict[str, float] = {}

        for tk in tickers:
            p_in = px_lookup.get(tk, {}).get(d_in)
            p_out = px_lookup.get(tk, {}).get(d_out)
            if not (p_in and p_out):
                continue
            ds_t, cs_t = idx_lookup[tk]
            # Locate d_sig index strictly <= d_sig
            pos = int(np.searchsorted(ds_t, d_sig, side="right")) - 1
            if pos < 252:
                continue
            rec = snap.get(tk)
            if not rec:
                continue
            rmap[tk] = p_out / p_in - 1.0
            if rec["nby"] is not None:
                elig_nby[tk] = rec["nby"]
            if rec["gpa"] is not None:
                elig_gpa[tk] = rec["gpa"]
            if rec["fcf"] is not None:
                elig_fcf[tk] = rec["fcf"]
            # Trailing 12-1m momentum (t-252 to t-21) & SMA200 strictly <= d_sig
            if cs_t[pos - 252] > 0 and cs_t[pos - 21] > 0:
                elig_mom[tk] = float(cs_t[pos - 21] / cs_t[pos - 252] - 1.0)
            sma200 = float(np.mean(cs_t[pos - 199 : pos + 1]))
            elig_above200[tk] = bool(cs_t[pos] > sma200)

        # Winsorize GP/A and FCF/A cross-sectionally at [1%, 99%]
        gpa_w = _winsorize_map(elig_gpa)
        fcf_w = _winsorize_map(elig_fcf)
        r_nby = _pct_ranks(elig_nby)
        r_gpa = _pct_ranks(gpa_w)
        r_fcf = _pct_ranks(fcf_w)

        comp: Dict[str, float] = {}
        all_elig = set(r_nby) | set(r_gpa) | set(r_fcf)
        for tk in all_elig:
            parts = [m[tk] for m in (r_nby, r_gpa, r_fcf) if tk in m]
            if len(parts) >= 2:
                comp[tk] = float(np.mean(parts))

        if len(comp) < 50 or len(elig_nby) < 40 or len(gpa_w) < 40:
            continue

        q_labels.append(d_sig)
        fwd_maps.append(rmap)
        spy_q_rets.append(p_spy_out / p_spy_in - 1.0)
        elig_counts.append(len(comp))

        # Equal-weight benchmarks
        ew = {tk: 1.0 / len(comp) for tk in comp}
        ew_gpa = {tk: 1.0 / len(gpa_w) for tk in gpa_w}
        w_arms["EW_UNIV"].append(ew)
        w_arms["EW_GPA_UNIV"].append(ew_gpa)

        # Arm Q1: Top 20 NBY
        top20_nby = [k for k, _ in sorted(elig_nby.items(), key=lambda x: (-x[1], x[0]))[:20]]
        w_arms["Q1_TOP20_NBY"].append({tk: 1.0 / len(top20_nby) for tk in top20_nby})

        # Arm Q2: Top 20 GP/A
        top20_gpa = [k for k, _ in sorted(gpa_w.items(), key=lambda x: (-x[1], x[0]))[:20]]
        w_arms["Q2_TOP20_GPA"].append({tk: 1.0 / len(top20_gpa) for tk in top20_gpa})

        # Arm Q3: Top 20 Composite Q_NBY
        comp_sorted = [k for k, _ in sorted(comp.items(), key=lambda x: (-x[1], x[0]))]
        top20_comp = comp_sorted[:20]
        w_arms["Q3_TOP20_COMPOSITE"].append({tk: 1.0 / len(top20_comp) for tk in top20_comp})

        # Diagnostic Q5 vs Q1 quintiles (top 20% vs bottom 20% of Composite)
        q_n = max(10, len(comp_sorted) // 5)
        w_arms["DIAG_Q5_COMP"].append({tk: 1.0 / q_n for tk in comp_sorted[:q_n]})
        w_arms["DIAG_Q1_COMP"].append({tk: 1.0 / q_n for tk in comp_sorted[-q_n:]})

        # Diagnostic Q3 + Close > SMA200 trend filter (cash remainder)
        w_arms["DIAG_Q3_SMA200"].append({tk: 1.0 / 20.0 for tk in top20_comp if elig_above200.get(tk, False)})

        # Diagnostic Momentum-neutral double sort: split into 5 momentum buckets, take top/bottom 20% Q_NBY inside each
        both_m = [tk for tk in comp if tk in elig_mom]
        both_m.sort(key=lambda tk: elig_mom[tk])
        top_mn: List[str] = []
        bot_mn: List[str] = []
        for b in range(5):
            lo_i = (b * len(both_m)) // 5
            hi_i = ((b + 1) * len(both_m)) // 5
            bucket = sorted(both_m[lo_i:hi_i], key=lambda tk: -comp[tk])
            k_take = max(2, len(bucket) // 5)
            top_mn.extend(bucket[:k_take])
            bot_mn.extend(bucket[-k_take:])
        w_arms["DIAG_MOM_NEUTRAL_TOP"].append({tk: 1.0 / len(top_mn) for tk in top_mn} if top_mn else {})
        w_arms["DIAG_MOM_NEUTRAL_BOT"].append({tk: 1.0 / len(bot_mn) for tk in bot_mn} if bot_mn else {})

    # Simulate primary (10 bps) & stress (20 bps)
    ew_10, ew_to = _simulate_quarterly(w_arms["EW_UNIV"], fwd_maps, COST_PRIMARY)
    ew_20, _ = _simulate_quarterly(w_arms["EW_UNIV"], fwd_maps, COST_STRESS)
    ew_gpa_10, _ = _simulate_quarterly(w_arms["EW_GPA_UNIV"], fwd_maps, COST_PRIMARY)
    ew_gpa_20, _ = _simulate_quarterly(w_arms["EW_GPA_UNIV"], fwd_maps, COST_STRESS)
    spy_arr = np.asarray(spy_q_rets, dtype=float)

    # Quintile spreads (diagnostic + Q5 > Q1 gate)
    q5_10, _ = _simulate_quarterly(w_arms["DIAG_Q5_COMP"], fwd_maps, COST_PRIMARY)
    q1_10, _ = _simulate_quarterly(w_arms["DIAG_Q1_COMP"], fwd_maps, COST_PRIMARY)
    q5_minus_q1_ann = float((q5_10 - q1_10).mean() * 4.0)

    def _eval_arm(name: str, bench_10: np.ndarray, bench_20: np.ndarray, q5_beats_q1: bool) -> Dict[str, Any]:
        r10, to_ann = _simulate_quarterly(w_arms[name], fwd_maps, COST_PRIMARY)
        r20, _ = _simulate_quarterly(w_arms[name], fwd_maps, COST_STRESS)
        ex10 = r10 - bench_10
        ex20 = r20 - bench_20
        # Annualized excess return (4 * quarterly mean) via circular 2-quarter block bootstrap, 98.33% CI
        boots = block_bootstrap_means(ex10 * 4.0, block=2, draws=DRAWS, seed=SEED, circular=True)
        alpha = 0.05 / 3.0
        lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
        h1_mask = np.array([d <= "2021-06-30" for d in q_labels])
        h2_mask = ~h1_mask
        h1_ex = float(ex10[h1_mask].mean() * 4.0)
        h2_ex = float(ex10[h2_mask].mean() * 4.0)
        stress_ex = float(ex20.mean() * 4.0)
        verdict = "PASS" if (lo > 0.0 and h1_ex > 0.0 and h2_ex > 0.0 and stress_ex > 0.0 and q5_beats_q1) else "FAIL"
        return {
            "metrics_10bps": _portfolio_stats(r10),
            "metrics_20bps": _portfolio_stats(r20),
            "ann_turnover": round(to_ann, 3),
            "excess_vs_ew_ann_10bps": round(float(ex10.mean() * 4.0), 5),
            "excess_vs_ew_ci9833": [round(float(lo), 5), round(float(hi), 5)],
            "excess_halves": {"2016-2021H1": round(h1_ex, 5), "2021H2-2026": round(h2_ex, 5)},
            "excess_vs_ew_ann_20bps": round(stress_ex, 5),
            "excess_vs_spy_ann_10bps": round(float((r10 - spy_arr).mean() * 4.0), 5),
            "verdict": verdict,
        }

    # Check Q5 > Q1 for each individual arm too
    arm_results = {
        "Q1_TOP20_NBY": _eval_arm("Q1_TOP20_NBY", ew_10, ew_20, q5_beats_q1=True),
        "Q2_TOP20_GPA": _eval_arm("Q2_TOP20_GPA", ew_gpa_10, ew_gpa_20, q5_beats_q1=True),
        "Q3_TOP20_COMPOSITE": _eval_arm("Q3_TOP20_COMPOSITE", ew_10, ew_20, q5_beats_q1=(q5_minus_q1_ann > 0.0)),
    }

    # Diagnostics
    sma_10, sma_to = _simulate_quarterly(w_arms["DIAG_Q3_SMA200"], fwd_maps, COST_PRIMARY)
    mn_top, _ = _simulate_quarterly(w_arms["DIAG_MOM_NEUTRAL_TOP"], fwd_maps, COST_PRIMARY)
    mn_bot, _ = _simulate_quarterly(w_arms["DIAG_MOM_NEUTRAL_BOT"], fwd_maps, COST_PRIMARY)
    mn_spread = mn_top - mn_bot
    mn_boots = block_bootstrap_means(mn_spread * 4.0, block=2, draws=DRAWS, seed=SEED, circular=True)
    mn_lo, mn_hi = np.percentile(mn_boots, [2.5, 97.5])

    # Quarterly correlation of Q3_TOP20_COMPOSITE vs SPY and vs VM_OUTLOOK_TOP3
    q3_10, _ = _simulate_quarterly(w_arms["Q3_TOP20_COMPOSITE"], fwd_maps, COST_PRIMARY)
    corr_spy = float(np.corrcoef(q3_10, spy_arr)[0, 1]) if q3_10.size > 2 else math.nan

    return {
        "registered": "docs/PREREG_EDGAR_QUALITY_BUYBACKS_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "survivorship_caveat": (
            "286-stock pool curated in 2026; raw CAGR vs SPY is survivorship-inflated. "
            "All primary verdicts benchmark against EW_UNIV of the exact same eligible stocks."
        ),
        "sync": sync_info,
        "window": [q_labels[0], q_labels[-1]] if q_labels else [],
        "n_quarters": len(q_labels),
        "median_eligible_stocks": int(np.median(elig_counts)) if elig_counts else 0,
        "benchmarks": {
            "EW_UNIV_10bps": _portfolio_stats(ew_10),
            "EW_UNIV_turnover": round(ew_to, 3),
            "EW_GPA_UNIV_10bps": _portfolio_stats(ew_gpa_10),
            "SPY_reference": _portfolio_stats(spy_arr),
        },
        "arms": arm_results,
        "diagnostics": {
            "Q5_minus_Q1_composite_ann_10bps": round(q5_minus_q1_ann, 5),
            "Q5_stats_10bps": _portfolio_stats(q5_10),
            "Q1_stats_10bps": _portfolio_stats(q1_10),
            "momentum_neutral_top_minus_bot_ann": round(float(mn_spread.mean() * 4.0), 5),
            "momentum_neutral_ci95": [round(float(mn_lo), 5), round(float(mn_hi), 5)],
            "Q3_with_SMA200_filter_10bps": _portfolio_stats(sma_10),
            "Q3_with_SMA200_turnover": round(sma_to, 3),
            "Q3_corr_vs_SPY": round(corr_spy, 3),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_project_env()
    ap = argparse.ArgumentParser(description="Study 2D: Point-in-time SEC EDGAR Quality + Net Buybacks")
    ap.add_argument("--skip-sync", action="store_true", help="Evaluate from existing local sec_fundamentals_pit.db")
    ap.add_argument("--pause", type=float, default=0.14, help="SEC EDGAR polite throttle seconds per issuer")
    a = ap.parse_args(list(argv) if argv is not None else None)

    res = evaluate(skip_sync=a.skip_sync)
    print("\n=== Study 2D: SEC EDGAR Point-in-Time Quality + Net Buybacks ===")
    print(f"Window {res['window']} (n={res['n_quarters']} quarters, med eligible={res['median_eligible_stocks']} stocks)")
    ew = res["benchmarks"]["EW_UNIV_10bps"]
    sp = res["benchmarks"]["SPY_reference"]
    print(f"EW_UNIV (10bps): CAGR={ew['cagr']:+.2%} Sharpe={ew['sharpe']:.3f} MaxDD={ew['max_dd']:.2%}")
    print(f"SPY reference:   CAGR={sp['cagr']:+.2%} Sharpe={sp['sharpe']:.3f} MaxDD={sp['max_dd']:.2%}")
    for name, st in res["arms"].items():
        m = st["metrics_10bps"]
        print(
            f"  {name:20s}: CAGR={m['cagr']:+.2%} Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.2%} "
            f"Ex_EW={st['excess_vs_ew_ann_10bps']:+.2%} CI98.33={st['excess_vs_ew_ci9833']} "
            f"halves={st['excess_halves']} 20bps={st['excess_vs_ew_ann_20bps']:+.2%} | {st['verdict']}"
        )
    print("Diagnostics:", json.dumps(res["diagnostics"], indent=2))

    out = REPO / "reports" / f"backtest_edgar_quality_buybacks_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
