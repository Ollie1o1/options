"""V1: does intraday data forecast the sleeve's basket volatility better than RV20?

Registration: docs/PREREG_SLEEVE_INTRADAY_VOL_20261127.md (frozen before the 5-minute
sleeve bars were downloaded). Data: `python -m src.intraday.alpaca_bars --timeframe 5Min
--adjustment all --out data/intraday_5m --symbols <16 sleeve ETFs> --start 2016-01`.

    python -m scripts.sleeve_vol_forecast_test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.backtest_vol_managed_outlook import (MAX_SCALE, MIN_SCALE, STEP, TARGET_VOL, TURNOVER_COST,
                                                  _align_matrix, _metrics_from_monthly, _simulate_arms,
                                                  _trailing_basket_rv)
from src.equity_playbooks.backtest import load_equity_ohlcv
from src.intraday import realized as rz
from src.intraday.alpaca_bars import fetch_calendar, load_symbol
from src.intraday.sessions import full_sessions, with_clock
from src.intraday.stats import block_bootstrap_means
from src.outlook.backtest import BENCH, DEFAULT_UNIVERSE, _features
from src.outlook.engine import DEFAULT_OUTLOOK_CONFIG, rank_universe

REPO = Path(__file__).resolve().parents[1]
DATA_5M = REPO / "data" / "intraday_5m"
SEED, DRAWS, BLOCK = 20261127, 10_000, 3
ALPHA = 0.05 / 2          # Bonferroni over E1, E2 -> 97.5% CI
MIN_N = 100
HALVES = [("2016-2020", 2016, 2020), ("2021-2026", 2021, 2026)]
ESTIMATORS = ("E0", "E1", "E2", "E3")


def grid(dates: List[str], cols: Dict[str, np.ndarray], universe=DEFAULT_UNIVERSE) -> List[Dict[str, Any]]:
    """The extended dynamic run's rebalance months, baskets and holding windows (copied logic)."""
    n = len(dates)
    rebal = [t for t in range(252, n - STEP - 1, STEP)
             if len([tk for tk in universe if tk in cols and np.all(np.isfinite(cols[tk][t - 252:t + 1]))]) >= 6]
    out = []
    bench = cols[BENCH]
    for k, t in enumerate(rebal):
        exec_in = t + 1
        exec_out = (rebal[k + 1] + 1) if k + 1 < len(rebal) else (n - 1)
        if exec_out <= exec_in:
            continue
        avail = [tk for tk in universe if tk in cols and np.all(np.isfinite(cols[tk][t - 252:t + 1]))
                 and np.all(np.isfinite(cols[tk][exec_in:exec_out + 1]))]
        if len(avail) < 3:
            continue
        feats = {tk: _features(cols[tk][:t + 1].tolist(), bench[:t + 1].tolist(), t) for tk in avail}
        ranked = rank_universe(feats, DEFAULT_OUTLOOK_CONFIG)
        w_a = {r["ticker"]: 1.0 / 3.0 for r in ranked[:3] if r["score"] > 0.0}
        out.append({"t": t, "exec_in": exec_in, "exec_out": exec_out, "date": dates[t],
                    "exec_date": dates[exec_in], "w_a": w_a})
    return out


def basket_daily(cols, w, lo: int, hi: int) -> np.ndarray:
    """Basket daily close-to-close returns for sessions lo..hi (inclusive), weights normalised."""
    ws = sum(w.values())
    r = np.zeros(hi - lo + 1)
    for tk, wt in w.items():
        seg = cols[tk][lo - 1:hi + 1]
        r += (wt / ws) * (seg[1:] / seg[:-1] - 1.0)
    return r


def simulate_b(months, cols, sigmas: List[Optional[float]], fallback: List[Optional[float]]):
    """Arm B with the given sigma per month (fallback to E0 where None). Mirrors _simulate_arms."""
    prev: Dict[str, float] = {}
    rets, scales, used_fallback = [], [], 0
    for m, s, s0 in zip(months, sigmas, fallback):
        w_a = m["w_a"]
        if s is None:
            s, used_fallback = s0, used_fallback + 1
        scale = float(np.clip(TARGET_VOL / s, MIN_SCALE, MAX_SCALE)) if (s and s > 0 and w_a) else 1.0
        w = {tk: v * scale for tk, v in w_a.items()}
        to = sum(abs(w.get(k, 0.0) - prev.get(k, 0.0)) for k in set(w) | set(prev))
        eq = 1.0 - to * TURNOVER_COST
        for d in range(m["exec_in"] + 1, m["exec_out"] + 1):
            eq *= 1.0 + sum(wt * (cols[tk][d] / cols[tk][d - 1] - 1.0) for tk, wt in w.items())
        rets.append(eq - 1.0)
        scales.append(scale)
        prev = w
    return np.asarray(rets), np.asarray(scales), used_fallback


def paired_test(d: np.ndarray, years: np.ndarray) -> Dict[str, Any]:
    means = block_bootstrap_means(d, block=BLOCK, draws=DRAWS, seed=SEED, circular=True)
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    halves = {lab: float(d[(years >= a) & (years <= b)].mean()) for lab, a, b in HALVES}
    p = float(np.mean(np.abs(means - d.mean()) >= abs(d.mean())))
    verdict = ("INSUFFICIENT" if len(d) < MIN_N else
               "PASS" if (lo > 0 and all(v > 0 for v in halves.values())) else "FAIL")
    return {"n": int(len(d)), "mean_D": float(d.mean()), "ci975": [float(lo), float(hi)], "p": max(p, 1 / DRAWS),
            "halves": halves, "share_months_better": float(np.mean(d > 0)), "verdict": verdict}


def run() -> Dict[str, Any]:
    ohlcv = load_equity_ohlcv(str(REPO / "data" / "equity_ohlcv.db"), [BENCH] + list(DEFAULT_UNIVERSE))
    dates, cols = _align_matrix(ohlcv, DEFAULT_UNIVERSE, strict_common=False, min_date="2005-01-01")
    months = grid(dates, cols)

    # intraday inputs
    cal = fetch_calendar()
    bars = {tk: load_symbol(tk, DATA_5M) for tk in DEFAULT_UNIVERSE}
    ref = with_clock(bars["QQQ"])
    sess = full_sessions(ref, cal)
    rets = {tk: rz.interval_returns(b, sess) for tk, b in bars.items() if not b.empty}
    frames = [rz.har_frame(rz.daily_rv(r)) for r in rets.values()]

    rows = []
    for m in months:
        w = m["w_a"]
        if not w:
            continue
        t = m["t"]
        d_t = dt.date.fromisoformat(m["date"])
        e0 = _trailing_basket_rv(cols, w, t, window=20)
        e1 = e2 = None
        if all(tk in rets for tk in w):
            brv = rz.basket_daily_rv(rets, w)
            e1 = rz.e1_intraday_rv20(brv, d_t)
            e2 = rz.e2_har(rz.fit_har(frames, d_t), brv, d_t)
        e3 = rz.e3_ewma(basket_daily(cols, w, max(1, t - 249), t))
        hold = basket_daily(cols, w, m["exec_in"] + 1, m["exec_out"])
        real_var = float(rz.ANN * np.mean(hold ** 2))
        rows.append({**m, "E0": e0, "E1": e1, "E2": e2, "E3": e3, "real_var": real_var,
                     "real_vol": math.sqrt(real_var)})
    first = next(i for i, r in enumerate(rows) if r["E1"] is not None)
    win = rows[first:]
    years = np.array([int(r["date"][:4]) for r in win])

    res: Dict[str, Any] = {"registered": "docs/PREREG_SLEEVE_INTRADAY_VOL_20261127.md",
                           "generated": dt.datetime.now().isoformat(timespec="seconds"),
                           "window": [win[0]["date"], win[-1]["date"]], "months": len(win),
                           "defined": {e: int(sum(r[e] is not None for r in win)) for e in ESTIMATORS}}

    # parity: E0 sleeve must reproduce the validated backtest's Arm B on these months
    sim = _simulate_arms(dates, cols, DEFAULT_UNIVERSE)
    ref_by_date = dict(zip(sim["period_dates"], sim["period_rets"]["VM_OUTLOOK_TOP3"]))
    all_months = [m for m in months]
    r0_all, _, _ = simulate_b(all_months, cols, [_trailing_basket_rv(cols, m["w_a"], m["t"], 20) if m["w_a"] else None
                                                 for m in all_months], [None] * len(all_months))
    diffs = [abs(r - ref_by_date[m["exec_date"]]) for m, r in zip(all_months, r0_all) if m["exec_date"] in ref_by_date]
    res["parity_max_abs_diff"] = float(max(diffs)) if diffs else None

    # primary: QLIKE paired differences
    loss = {e: np.array([rz.qlike(r["real_var"], r[e] ** 2) if r[e] else np.nan for r in win]) for e in ESTIMATORS}
    res["mean_qlike"] = {}
    res["tests"] = {}
    for k in ("E1", "E2", "E3"):
        ok = ~np.isnan(loss["E0"]) & ~np.isnan(loss[k])
        d = loss["E0"][ok] - loss[k][ok]
        res["tests"][k] = paired_test(d, years[ok]) | {"mean_qlike_E0": float(loss["E0"][ok].mean()),
                                                       "mean_qlike_k": float(loss[k][ok].mean())}
        if k == "E3":
            res["tests"][k]["verdict"] = "diagnostic (no authority)"
    # log-vol forecast error diagnostics
    res["bias_log_vol"] = {e: float(np.nanmean([math.log(r[e] / r["real_vol"]) if r[e] else np.nan for r in win]))
                           for e in ESTIMATORS}
    res["corr_log_vol"] = {e: float(pd.Series([math.log(r[e]) if r[e] else np.nan for r in win]).corr(
        pd.Series([math.log(r["real_vol"]) for r in win]))) for e in ESTIMATORS}

    # consequences: sleeve re-simulated per estimator (fallback E0 where undefined)
    e0s = [r["E0"] for r in win]
    sleeves = {}
    for e in ESTIMATORS:
        rr, sc, fb = simulate_b(win, cols, [r[e] for r in win], e0s)
        mt = _metrics_from_monthly(rr)
        capped = sc < 1.0 - 1e-12
        err = np.abs(sc * np.array([r["real_vol"] for r in win]) - TARGET_VOL)[capped]
        sleeves[e] = mt | {"fallback_months": int(fb), "months_scaled": int(capped.sum()),
                           "mean_abs_err_vs_target_on_scaled": float(err.mean()) if err.size else None,
                           "avg_scale": float(sc.mean())}
    res["sleeve"] = sleeves
    base = sleeves["E0"]
    adopt = []
    for k in ("E1", "E2"):
        s = sleeves[k]
        qualifies = (res["tests"][k]["verdict"] == "PASS" and s["sharpe"] >= base["sharpe"] - 0.05
                     and s["max_dd"] >= base["max_dd"] - 0.02)
        res["tests"][k]["adoption_qualifies"] = bool(qualifies)
        if qualifies:
            adopt.append((res["tests"][k]["mean_D"], k))
    res["adopt"] = max(adopt)[1] if adopt else None
    return res


def print_report(r: Dict[str, Any]) -> None:
    print(f"V1 sleeve vol forecast — {r['months']} months {r['window'][0]}..{r['window'][1]}; defined {r['defined']}")
    print(f"parity vs validated Arm B: max |diff| = {r['parity_max_abs_diff']:.2e}")
    for k, t in r["tests"].items():
        print(f"  {k} vs E0: n={t['n']} mean D={t['mean_D']:+.4f} 97.5% CI [{t['ci975'][0]:+.4f}, {t['ci975'][1]:+.4f}]"
              f" p={t['p']:.4f} better in {t['share_months_better']:.0%} of months | halves "
              + ", ".join(f"{a} {v:+.4f}" for a, v in t["halves"].items())
              + f" | QLIKE E0 {t['mean_qlike_E0']:.4f} -> {t['mean_qlike_k']:.4f} | {t['verdict']}")
    print("  log-vol bias:", {k: round(v, 3) for k, v in r["bias_log_vol"].items()},
          " corr:", {k: round(v, 3) for k, v in r["corr_log_vol"].items()})
    print("  sleeve:")
    for e, s in r["sleeve"].items():
        err = s["mean_abs_err_vs_target_on_scaled"]
        print(f"    {e}: CAGR {s['cagr']:+.2%} vol {s['ann_vol']:.2%} Sharpe {s['sharpe']:.3f} MaxDD {s['max_dd']:+.2%}"
              f" avg scale {s['avg_scale']:.2f} | scaled months {s['months_scaled']}, |vol-14%| "
              f"{(err or float('nan')):.2%} | fallback {s['fallback_months']}")
    print(f"  ADOPT: {r['adopt']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="V1 sleeve intraday vol forecast test")
    ap.parse_args(argv)
    r = run()
    print_report(r)
    out = REPO / "reports" / "intraday" / f"sleeve_vol_forecast_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
