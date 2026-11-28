#!/usr/bin/env python3
"""Preregistered estimator for Study H4: Intraday Options Execution Clock & Multi-ETF OPRA Liquidity.

Frozen registration: docs/PREREG_H4_INTRADAY_OPTIONS_EXEC_20261127.md
Defaults to `--estimate` ($0 dry-run). Pass `--download --max-usd 8.0` to fetch & evaluate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.databento_opra import DEFAULT_DB, fetch_opra_snapshots
from src.data_quality import is_trading_day
from src.equity_playbooks.backtest import load_equity_ohlcv
from src.paths import load_project_env, repo_path
from src.radar.wide_spread_rule import LegQuote, choose_expiration, quotes_from_rows, select_legs

REPO = Path(__file__).resolve().parents[1]
ETFS: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "XLE", "GLD", "TLT")
CLOCKS: Tuple[str, ...] = ("09:45", "10:30", "12:00", "14:00", "15:45")
SEED = 20261127
DRAWS = 10_000


def sampled_dates() -> List[str]:
    """First full-session (16:00 ET close) trading Wednesday of each month 2024-01..2025-12.

    Uses the exchange calendar (`data/intraday/calendar.parquet`) so US market holidays
    (e.g. 2025-01-01) and early-close sessions (e.g. 2024-07-03 13:00 ET close) roll
    forward deterministically to the next full trading session.
    """
    from src.intraday.alpaca_bars import fetch_calendar

    cal = fetch_calendar()
    full_days = set(cal[cal["close"] == "16:00"]["date"].tolist())
    out: List[str] = []
    for y in (2024, 2025):
        for m in range(1, 13):
            d = dt.date(y, m, 1)
            while d.weekday() != 2:  # Wednesday
                d += dt.timedelta(days=1)
            while d not in full_days:
                d += dt.timedelta(days=1)
            out.append(d.isoformat())
    return out


def _bootstrap_ci(vals: np.ndarray, alpha: float) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = vals.size
    idx = rng.integers(0, n, size=(DRAWS, n))
    means = np.mean(vals[idx], axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def _clock_bucket(ts_event: str, trade_date: str) -> Optional[str]:
    """Map UTC ts_event string to the nearest preregistered ET clock label."""
    from zoneinfo import ZoneInfo

    try:
        clean = ts_event.replace("Z", "+00:00")
        t_utc = dt.datetime.fromisoformat(clean[:26] + ("+00:00" if "+" not in clean[10:] else ""))
        t_et = t_utc.astimezone(ZoneInfo("America/New_York"))
        mins = t_et.hour * 60 + t_et.minute
    except Exception:
        return None
    best_lbl: Optional[str] = None
    best_gap = 999
    for clk in CLOCKS:
        hh, mm = (int(x) for x in clk.split(":"))
        gap = abs(mins - (hh * 60 + mm))
        if gap <= 3 and gap < best_gap:
            best_gap = gap
            best_lbl = clk
    return best_lbl


def _raw_spot_map(symbols: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """Split-unadjusted official daily closes for moneyness/BS-delta on historical OPRA strikes."""
    from src.raw_closes import official_raw_closes

    out: Dict[Tuple[str, str], float] = {}
    series_map = official_raw_closes(list(symbols), start="2023-12-01", end="2026-01-15")
    for sym, ser in series_map.items():
        for d_obj, px in ser.items():
            if math.isfinite(float(px)) and float(px) > 0:
                out[(sym, str(d_obj)[:10])] = float(px)
    return out


def download_or_estimate(
    *,
    download: bool = False,
    max_usd: float = 8.0,
    db_path: str = DEFAULT_DB,
) -> Dict[str, Any]:
    dates = sampled_dates()
    # Skip any (date, clock) already stored in opra_chain so reruns cost $0
    existing: set[Tuple[str, str]] = set()
    full = repo_path(db_path)
    if Path(full).exists():
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            for d, clk in con.execute("SELECT DISTINCT date, clock_et FROM opra_chain").fetchall():
                if clk in CLOCKS:
                    existing.add((str(d), str(clk)))
        except sqlite3.Error:
            pass
        finally:
            con.close()

    total_est = 0.0
    per_clock: Dict[str, Any] = {}
    plans_by_clock: Dict[str, List[Dict[str, Any]]] = {}
    for clk in CLOCKS:
        todo_dates = [d for d in dates if (d, clk) not in existing]
        if not todo_dates:
            per_clock[clk] = {"dates_needed": 0, "est_usd": 0.0, "status": "cached"}
            continue
        res = fetch_opra_snapshots(
            ETFS, todo_dates, hhmm_et=clk, minutes=1, max_usd=max_usd, dry_run=True, db_path=db_path
        )
        total_est += float(res["total_est_usd"])
        plans_by_clock[clk] = list(res["plan"])
        per_clock[clk] = {"dates_needed": len(todo_dates), "est_usd": res["total_est_usd"], "status": "estimated"}
        print(f"[H4 estimate] {clk}: {len(todo_dates)} dates = ${res['total_est_usd']:.4f}", flush=True)

    summary: Dict[str, Any] = {
        "etfs": list(ETFS),
        "dates": len(dates),
        "clocks": list(CLOCKS),
        "total_est_usd": round(total_est, 4),
        "max_usd": max_usd,
        "per_clock": per_clock,
        "downloaded": False,
    }
    if not download:
        return summary
    if total_est > max_usd:
        raise RuntimeError(f"Refused H4 download: est ${total_est:.4f} > max_usd ${max_usd:.2f}")

    total_rows = 0
    for clk in CLOCKS:
        todo_dates = [d for d in dates if (d, clk) not in existing]
        if not todo_dates:
            continue
        r = fetch_opra_snapshots(
            ETFS,
            todo_dates,
            hhmm_et=clk,
            minutes=1,
            max_usd=max_usd,
            dry_run=False,
            db_path=db_path,
            precomputed_plan=plans_by_clock.get(clk),
        )
        total_rows += int(r["rows_written"])
        per_clock[clk]["status"] = f"downloaded ({r['rows_written']:,} rows)"
        print(f"[H4 download] {clk}: {r['rows_written']:,} rows (cum spend=${r.get('new_cumulative_spend_usd', 0):.4f})", flush=True)
    summary["downloaded"] = True
    summary["rows_written"] = total_rows
    return summary


def evaluate(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    full = repo_path(db_path)
    if not Path(full).exists():
        raise FileNotFoundError(f"No OPRA DB at {full}; run with --download first.")

    spots = _raw_spot_map(ETFS)
    dates = sampled_dates()

    # Load all rows for sampled dates grouped by (symbol, date, clock_et)
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        ph = ",".join("?" * len(dates))
        raw = con.execute(
            f"SELECT symbol, date, clock_et, expiration, strike, type, bid, ask, bid_size, ask_size "
            f"FROM opra_chain WHERE date IN ({ph})",
            dates,
        ).fetchall()
    finally:
        con.close()

    grouped: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for sym, d, clk, exp, strike, ot, bid, ask, bsz, asz in raw:
        clk_s = str(clk)
        if clk_s not in CLOCKS or sym not in ETFS:
            continue
        grouped.setdefault((str(sym), str(d), clk_s), {}).setdefault(str(exp), []).append(
            {
                "strike": float(strike),
                "type": str(ot),
                "bid": bid,
                "ask": ask,
                "bid_size": bsz,
                "ask_size": asz,
            }
        )

    # Evaluate O1 bull put (60-120 DTE) & near-ATM single leg (20-50 DTE) per (sym, date, clock)
    o1_records: List[Dict[str, Any]] = []
    atm_records: List[Dict[str, Any]] = []

    for (sym, d, clk), by_exp in grouped.items():
        spot = spots.get((sym, d))
        if not spot or spot <= 0:
            continue
        d_obj = dt.date.fromisoformat(d)
        exp_dtes = {e: float((dt.date.fromisoformat(e) - d_obj).days) for e in by_exp}

        # 1. O1 wide bull put (60-120 DTE)
        o1_exp = choose_expiration(exp_dtes)
        if o1_exp is not None:
            dte = exp_dtes[o1_exp]
            put_rows = [r for r in by_exp[o1_exp] if r["type"] == "put"]
            # Deduplicate per strike keeping latest quote in the 1-min bar
            by_k = {r["strike"]: r for r in put_rows}
            qs = quotes_from_rows(by_k.values(), spot=spot, dte=dte, opt_type="put", r=0.045)
            # Attach bid/ask size map for depth check
            sz_map = {k: (r.get("bid_size"), r.get("ask_size")) for k, r in by_k.items()}
            legs, reason = select_legs(qs, spot=spot, side="bull_put", expiration=o1_exp, dte=dte, max_friction_to_credit=1.50)
            if legs is not None:
                s_bsz, _ = sz_map.get(legs.short.strike, (None, None))
                _, l_asz = sz_map.get(legs.long.strike, (None, None))
                depths = [float(x) for x in (s_bsz, l_asz) if x is not None and float(x) > 0]
                o1_records.append(
                    {
                        "symbol": sym,
                        "date": d,
                        "year": int(d[:4]),
                        "clock": clk,
                        "fric_mid": legs.round_trip_spread / legs.credit_mid if legs.credit_mid > 0 else math.nan,
                        "fric_cross": legs.friction_ratio,
                        "pass_25pct": bool(legs.friction_ratio <= 0.25),
                        "min_depth": min(depths) if depths else None,
                        "credit_mid": legs.credit_mid,
                    }
                )

        # 2. Near-ATM single-leg option (20-50 DTE) relative spread
        atm_cands = [(abs(v - 35.0), e, v) for e, v in exp_dtes.items() if 20.0 <= v <= 50.0]
        if atm_cands:
            atm_cands.sort()
            _, atm_exp, _ = atm_cands[0]
            two_sided = [
                r for r in by_exp[atm_exp]
                if r.get("bid") is not None and r.get("ask") is not None
                and float(r["bid"]) > 0 and float(r["ask"]) >= float(r["bid"])
                and abs(float(r["strike"]) / spot - 1.0) <= 0.03
            ]
            if two_sided:
                rel_sprs = [
                    (float(r["ask"]) - float(r["bid"])) / (0.5 * (float(r["ask"]) + float(r["bid"])))
                    for r in two_sided
                ]
                atm_records.append(
                    {"symbol": sym, "date": d, "year": int(d[:4]), "clock": clk, "rel_spread": float(np.median(rel_sprs))}
                )

    # Aggregate paired day-level diffs vs 09:45 (H4-A) and vs 15:45 (H4-B)
    def _paired_table(records: List[Dict[str, Any]], val_key: str, ref_clock: str, alpha: float) -> Dict[str, Any]:
        by_dc: Dict[Tuple[str, str], List[float]] = {}
        for r in records:
            v = r.get(val_key)
            if v is not None and math.isfinite(float(v)):
                by_dc.setdefault((r["date"], r["clock"]), []).append(float(v))
        day_means = {k: float(np.mean(vs)) for k, vs in by_dc.items()}
        out: Dict[str, Any] = {}
        for clk in CLOCKS:
            if clk == ref_clock:
                continue
            common_d = sorted({d for d, c in day_means if c == clk} & {d for d, c in day_means if c == ref_clock})
            if len(common_d) < 6:
                out[clk] = {"n_days": len(common_d), "verdict": "INSUFFICIENT"}
                continue
            diffs = np.array([day_means[(d, clk)] - day_means[(d, ref_clock)] for d in common_d], dtype=float)
            yrs = np.array([int(d[:4]) for d in common_d])
            lo, hi = _bootstrap_ci(diffs, alpha=alpha)
            h24 = float(diffs[yrs == 2024].mean()) if np.any(yrs == 2024) else float("nan")
            h25 = float(diffs[yrs == 2025].mean()) if np.any(yrs == 2025) else float("nan")
            verdict = "PASS_TIGHTER" if (hi < 0.0 and h24 < 0.0 and h25 < 0.0) else (
                "PASS_WIDER" if (lo > 0.0 and h24 > 0.0 and h25 > 0.0) else "NULL"
            )
            out[clk] = {
                "n_days": len(common_d),
                "mean_clock": round(float(np.mean([day_means[(d, clk)] for d in common_d])), 4),
                "mean_ref": round(float(np.mean([day_means[(d, ref_clock)] for d in common_d])), 4),
                "mean_diff": round(float(diffs.mean()), 4),
                "ci_bonf": [round(lo, 4), round(hi, 4)],
                "halves": {"2024": round(h24, 4), "2025": round(h25, 4)},
                "verdict": verdict,
            }
        return out

    # H4-C: Multi-ETF O1 friction audit at 15:45 ET
    h4_c: Dict[str, Any] = {}
    for sym in ETFS:
        sub = [r for r in o1_records if r["symbol"] == sym and r["clock"] == "15:45"]
        if not sub:
            h4_c[sym] = {"n": 0}
            continue
        fc = np.array([r["fric_cross"] for r in sub], dtype=float)
        dp = [r["min_depth"] for r in sub if r["min_depth"] is not None]
        h4_c[sym] = {
            "n": len(sub),
            "med_fric_cross": round(float(np.median(fc)), 4),
            "p90_fric_cross": round(float(np.percentile(fc, 90)), 4),
            "pass_25pct_rate": round(float(np.mean([r["pass_25pct"] for r in sub])), 3),
            "med_min_depth": round(float(np.median(dp)), 1) if dp else None,
        }

    # Depth by clock (O1 spreads)
    depth_by_clock = {}
    for clk in CLOCKS:
        dps = [r["min_depth"] for r in o1_records if r["clock"] == clk and r["min_depth"] is not None]
        frcs = [r["fric_mid"] for r in o1_records if r["clock"] == clk and math.isfinite(r["fric_mid"])]
        depth_by_clock[clk] = {
            "n_obs": len(frcs),
            "med_fric_mid": round(float(np.median(frcs)), 4) if frcs else None,
            "med_min_depth": round(float(np.median(dps)), 1) if dps else None,
        }

    return {
        "registered": "docs/PREREG_H4_INTRADAY_OPTIONS_EXEC_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "o1_obs": len(o1_records),
        "atm_obs": len(atm_records),
        "H4_A_o1_vs_0945": _paired_table(o1_records, "fric_mid", "09:45", alpha=0.05 / 4.0),
        "H4_A_atm_vs_0945": _paired_table(atm_records, "rel_spread", "09:45", alpha=0.05 / 4.0),
        "H4_B_o1_vs_1545": _paired_table(o1_records, "fric_mid", "15:45", alpha=0.05 / 3.0),
        "H4_B_atm_vs_1545": _paired_table(atm_records, "rel_spread", "15:45", alpha=0.05 / 3.0),
        "H4_C_etf_o1_1545": h4_c,
        "depth_and_spread_by_clock": depth_by_clock,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_project_env()
    ap = argparse.ArgumentParser(description="Study H4: Databento OPRA intraday execution clock")
    ap.add_argument("--download", action="store_true", help="Download OPRA snapshots if under --max-usd")
    ap.add_argument("--evaluate", action="store_true", help="Run H4 evaluation on local databento_opra.db")
    ap.add_argument("--max-usd", type=float, default=8.0)
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args(list(argv) if argv is not None else None)

    if not a.evaluate:
        dl = download_or_estimate(download=a.download, max_usd=a.max_usd, db_path=a.db)
        print(json.dumps(dl, indent=2))
        if not a.download:
            return 0

    res = evaluate(db_path=a.db)
    print(json.dumps(res, indent=2))
    out = REPO / "reports" / f"h4_opra_intraday_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
