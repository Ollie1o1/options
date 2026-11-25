#!/usr/bin/env python3
"""Preregistered estimator for Equity Playbook 6: Funded Pre-Catalyst Biotech Run-Up.

Frozen registration: docs/PREREG_BIOTECH_RUNUP_AND_INSIDER_20261124.md
100% offline using cached point-in-time data/catalyst_pit.db + data/squeeze_prices.db + XBI/SPY.
Exits strictly BEFORE primary completion (PCD - 3 trading days) to avoid binary trial gaps.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.catalyst import pit
from src.catalyst.backtest import panel as cat_panel
from src.equity_playbooks.backtest import (
    BenchmarkSeries,
    EquityTrade,
    cluster_bootstrap_stats,
    load_equity_ohlcv,
)
from src.squeeze.backtest import PRICES_DB
from src.squeeze.backtest.panel import PriceBook

SEED = 20261124
RT_COST = 0.0050     # 50 bps round-trip (small/mid biotech spread realism)
STOP_LOSS = -0.20    # -20% hard stop (executed next close)
MIN_TICKERS = 40


def _quarter_key(d: str) -> str:
    y = int(d[:4])
    m = int(d[5:7])
    return f"{y}-Q{(m - 1) // 3 + 1}"


def _load_pit_prices(con: sqlite3.Connection) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    for ticker, payload in con.execute("SELECT ticker, payload FROM pit_prices"):
        try:
            d = json.loads(payload)
            if isinstance(d, dict):
                out[str(ticker).upper()].update({k: float(v) for k, v in d.items() if v is not None})
        except Exception:
            continue
    return out


def _anchor_pcd(raw_date: str) -> Tuple[str, bool]:
    """Return (conservative_iso_date, is_day_precision). Month-only anchors at 1st of month."""
    s = str(raw_date or "").strip()
    if len(s) == 7 and s[4] == "-":
        return f"{s}-01", False
    if len(s) >= 10:
        return s[:10], True
    return "", False


def _eval_arm(trades: List[EquityTrade], label: str) -> dict:
    s_ticker = cluster_bootstrap_stats(trades, lambda t: t.ticker, n_boot=10_000, seed=SEED, min_clusters=MIN_TICKERS)
    s_qtr = cluster_bootstrap_stats(trades, lambda t: _quarter_key(t.entry_date), n_boot=10_000, seed=SEED, min_clusters=6)

    # Wider CI governs
    if s_ticker["insufficient"]:
        verdict_ha = "INSUFFICIENT"
        verdict_hb = "INSUFFICIENT"
    else:
        lo_r = min(s_ticker["mean_ret_ci"][0], s_qtr["mean_ret_ci"][0] if s_qtr["mean_ret_ci"] else s_ticker["mean_ret_ci"][0])
        lo_pf = min(s_ticker["pf_ci"][0], s_qtr["pf_ci"][0] if s_qtr["pf_ci"] else s_ticker["pf_ci"][0])
        verdict_ha = "PASS" if (lo_r > 0.0 and lo_pf > 1.0) else "FAIL"

        if s_ticker["mean_excess_ci"] is None:
            verdict_hb = "INSUFFICIENT"
        else:
            lo_x = min(s_ticker["mean_excess_ci"][0], s_qtr["mean_excess_ci"][0] if s_qtr["mean_excess_ci"] else s_ticker["mean_excess_ci"][0])
            verdict_hb = "PASS" if lo_x > 0.0 else "FAIL"

    return {
        "label": label,
        "verdict_H6A": verdict_ha,
        "verdict_H6B_xbi": verdict_hb,
        "cluster_by_ticker": s_ticker,
        "cluster_by_quarter": s_qtr,
    }


def main():
    p = argparse.ArgumentParser(description="Backtest Equity Playbook 6 (Funded Biotech Pre-Catalyst Run-Up)")
    p.add_argument("--out", default="reports/backtest_biotech_runup_20261124.json")
    args = p.parse_args()

    ohlcv = load_equity_ohlcv("data/equity_ohlcv.db", ["XBI", "SPY"])
    xbi = BenchmarkSeries(ohlcv["XBI"][0], ohlcv["XBI"][1])
    spy = BenchmarkSeries(ohlcv["SPY"][0], ohlcv["SPY"][1])

    con = sqlite3.connect("file:data/catalyst_pit.db?mode=ro", uri=True)
    # Keep 100% offline & deterministic: serve caps strictly from cached pit_caps
    # so pit.board_as_of never hits Yahoo or tries to write to the readonly DB.
    cached_caps = {str(t).upper(): float(c) for t, c in con.execute("SELECT ticker, cap FROM pit_caps")}
    pit._caps = lambda tickers, conn=None: {t: cached_caps[t] for t in tickers if t in cached_caps}  # type: ignore[assignment]
    # Collect all cached nct_ids from pit_universe
    nct_ids_set: Set[str] = set()
    for (payload,) in con.execute("SELECT payload FROM pit_universe"):
        try:
            items = json.loads(payload)
            for it in items:
                if isinstance(it, str):
                    nct_ids_set.add(it)
                elif isinstance(it, dict) and it.get("nct_id"):
                    nct_ids_set.add(str(it["nct_id"]))
        except Exception:
            continue
    if not nct_ids_set:
        nct_ids_set = {r[0] for r in con.execute("SELECT DISTINCT nct_id FROM pit_study")}
    nct_ids = sorted(nct_ids_set)
    pit_px = _load_pit_prices(con)

    vints = cat_panel.vintages("2023-01-01", "2025-10-01")
    print(f"Reconstructing PIT biotech panel across {len(vints)} vintages, {len(nct_ids)} cached trials...")

    # Deduplicate to 1 event per (ticker, pcd_month) keeping earliest vintage where trial appeared
    candidates: Dict[Tuple[str, str], dict] = {}
    for v in vints:
        rows, _cov = cat_panel.build(v, nct_ids, con)
        for r in rows:
            pcd_iso, day_prec = _anchor_pcd(r.event_date)
            if not pcd_iso or pcd_iso <= v:
                continue
            key = (r.ticker.upper(), pcd_iso[:7])
            # Prefer a vintage close to T-45 trading days (~65 cal days before PCD)
            days_ahead = (dt.date.fromisoformat(pcd_iso) - dt.date.fromisoformat(v)).days
            if not (60 <= days_ahead <= 165):
                continue
            if key not in candidates or days_ahead < candidates[key]["days_ahead"]:
                candidates[key] = {
                    "ticker": r.ticker.upper(),
                    "vintage": v,
                    "pcd_iso": pcd_iso,
                    "day_prec": day_prec,
                    "phase": r.phase,
                    "funded_through": r.funded_through,
                    "days_ahead": days_ahead,
                }
    con.close()

    cand_tickers = sorted({c["ticker"] for c in candidates.values()})
    print(f"Found {len(candidates)} unique (ticker, pcd_month) candidates across {len(cand_tickers)} tickers. Loading prices...")
    sq_book = PriceBook(PRICES_DB, cand_tickers)

    funded_trades: List[EquityTrade] = []
    raise_trades: List[EquityTrade] = []
    unknown_trades: List[EquityTrade] = []

    for cand in sorted(candidates.values(), key=lambda x: (x["pcd_iso"], x["ticker"])):
        sym = cand["ticker"]
        pcd_iso = cand["pcd_iso"]
        # Prefer squeeze_prices.db (carries volume for $1M liquidity gate); fallback to pit_prices
        if sym in sq_book:
            dates = sq_book._dates[sym]
            closes = sq_book._close[sym]
            vols = sq_book._vol[sym]
            has_vol = True
        elif sym in pit_px and len(pit_px[sym]) >= 60:
            dates = sorted(pit_px[sym].keys())
            closes = np.asarray([pit_px[sym][d] for d in dates], dtype=float)
            vols = np.zeros_like(closes)
            has_vol = False
        else:
            continue

        pcd_idx = bisect.bisect_left(dates, pcd_iso)
        if pcd_idx >= len(dates):
            # PCD may be slightly past last bar if unelapsed; require exit bar <= last bar
            if not dates or dates[-1] < (dt.date.fromisoformat(pcd_iso) - dt.timedelta(days=5)).isoformat():
                continue
            pcd_idx = len(dates)

        t_entry_sig = pcd_idx - 45
        entry_i = t_entry_sig + 1
        exit_target_i = pcd_idx - 3
        if t_entry_sig < 20 or entry_i >= exit_target_i or exit_target_i >= len(dates):
            continue
        # Ensure entry is on or after the vintage where trial+runway were observed!
        if dates[t_entry_sig] < cand["vintage"]:
            continue

        c_sig = float(closes[t_entry_sig])
        if c_sig < 2.0 or not np.isfinite(c_sig):
            continue
        if has_vol:
            med_dv = float(np.median(closes[t_entry_sig - 19 : t_entry_sig + 1] * vols[t_entry_sig - 19 : t_entry_sig + 1]))
            if med_dv < 1_000_000.0:
                continue

        entry_px = float(closes[entry_i])
        entry_d = dates[entry_i]
        if entry_px <= 0.0 or not np.isfinite(entry_px):
            continue

        exit_i = exit_target_i
        reason = "pre_event_t_minus_3"
        for k in range(entry_i + 1, exit_target_i):
            if closes[k] / entry_px - 1.0 <= STOP_LOSS:
                exit_i = min(k + 1, exit_target_i)
                reason = "stop_20pct"
                break

        exit_px = float(closes[exit_i])
        exit_d = dates[exit_i]
        if exit_px <= 0.0 or not np.isfinite(exit_px):
            continue
        net_r = (exit_px / entry_px - 1.0) - RT_COST
        xbi_r = xbi.window_return(entry_d, exit_d)
        spy_r = spy.window_return(entry_d, exit_d)

        tr = EquityTrade(
            ticker=sym,
            signal_date=dates[t_entry_sig],
            entry_date=entry_d,
            exit_date=exit_d,
            entry_price=entry_px,
            exit_price=exit_px,
            net_return=net_r,
            bench_return=xbi_r,
            exit_reason=reason,
            stratum="broad",
            extra={"spy_ret": spy_r, "day_prec": cand["day_prec"], "phase": cand["phase"]},
        )
        if cand["funded_through"] is True:
            funded_trades.append(tr)
        elif cand["funded_through"] is False:
            raise_trades.append(tr)
        else:
            unknown_trades.append(tr)

    payload = {
        "prereg": "docs/PREREG_BIOTECH_RUNUP_AND_INSIDER_20261124.md",
        "primary_funded_through": _eval_arm(funded_trades, "B_FUNDED_T45_T3"),
        "control_raise_before": _eval_arm(raise_trades, "CONTROL_RAISE_BEFORE"),
        "unknown_runway": _eval_arm(unknown_trades, "UNKNOWN_RUNWAY"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    for k in ("primary_funded_through", "control_raise_before", "unknown_runway"):
        r = payload[k]
        t = r["cluster_by_ticker"]
        print(
            f"\n[{r['label']}] H6-A={r['verdict_H6A']} H6-B(XBI)={r['verdict_H6B_xbi']}"
            f"\n  n={t['n_trades']} tickers={t['n_clusters']} mean={t['mean_ret']} CI={t['mean_ret_ci']} PF={t['pf']} CI={t['pf_ci']} exXBI={t['mean_excess']} CI={t['mean_excess_ci']}"
        )


if __name__ == "__main__":
    main()
