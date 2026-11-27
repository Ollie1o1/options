"""Run the preregistered intraday tests I1 + S1..S7 (shares, index ETFs).

Registration: docs/PREREG_INTRADAY_LATE_DAY_AND_OPEN_20261127.md (frozen before
the 2016-2023 data was downloaded). Data: `python -m src.intraday.alpaca_bars`.

    python -m scripts.intraday_tests                # prints the report, writes JSON
    python -m scripts.intraday_tests --json-out PATH
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.intraday.alpaca_bars import DEFAULT_OUT, fetch_calendar, load_symbol
from src.intraday.sessions import build_tables, hhmm, stack
from src.intraday.stats import holm, summarise

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TARGETS = {name: hhmm(clock) for name, clock in [
    ("p0930", "09:30"), ("p0931", "09:31"), ("p0935", "09:35"), ("p1000", "10:00"),
    ("p1030", "10:30"), ("p1530", "15:30"), ("p1545", "15:45"), ("p1600", "16:00")]}
BINDING = (dt.date(2016, 1, 4), dt.date(2023, 12, 29))
HALVES = [("2016-2019", dt.date(2016, 1, 1), dt.date(2019, 12, 31)),
          ("2020-2023", dt.date(2020, 1, 1), dt.date(2023, 12, 31))]
SEEN_START = dt.date(2024, 1, 2)
MIN_N = 1500
BLOCK, DRAWS, SEED = 10, 10_000, 20261127
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- rules
def _trade(df: pd.DataFrame, signal: pd.Series, entry: str, exit_: str, cents: float,
           sign: int, mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """Rows for trades: position = sign * sign(signal), entry -> exit, cents each way."""
    pos = sign * np.sign(signal)
    ok = signal.notna() & (signal != 0) & df[entry].notna() & df[exit_].notna()
    if mask is not None:
        ok &= mask.fillna(False)
    d = df[ok]
    gross = pos[ok] * (d[exit_] / d[entry] - 1.0)
    return pd.DataFrame({"date": d["date"], "symbol": d["symbol"], "gross": gross,
                         "inv_px": 1.0 / d[entry], "cents": cents})


def late_fade_move(df: pd.DataFrame, end: str = "p1530") -> pd.Series:
    return df[end] / df["p1030"] - 1.0


def big_move_mask(df: pd.DataFrame, move: pd.Series) -> pd.Series:
    """|move| above the 80th pct of that symbol's |move| over the prior 250 sessions (>=120)."""
    absm = move.abs()
    out = pd.Series(False, index=df.index)
    for _, idx in df.groupby("symbol").groups.items():
        s = absm.loc[idx]
        order = df.loc[idx, "date"].argsort()
        s = s.iloc[order]
        thr = s.rolling(250, min_periods=120).quantile(0.8).shift(1)
        out.loc[s.index] = (s > thr).values
    return out


RuleFn = Callable[[pd.DataFrame], pd.DataFrame]


def rule_i1(df: pd.DataFrame, cents: float = 0.01) -> pd.DataFrame:
    return _trade(df, late_fade_move(df), "p1530", "p1600", cents, sign=-1)


def rule_s2(df: pd.DataFrame) -> pd.DataFrame:
    m = late_fade_move(df)
    return _trade(df, m, "p1530", "p1600", 0.01, sign=-1, mask=big_move_mask(df, m))


def rule_s3(df: pd.DataFrame) -> pd.DataFrame:
    return _trade(df, late_fade_move(df, "p1545"), "p1545", "p1600", 0.01, sign=-1)


def rule_s4(df: pd.DataFrame) -> pd.DataFrame:
    return _trade(df, df["p1000"] / df["prev_close"] - 1.0, "p1530", "p1600", 0.01, sign=+1)


def rule_s5(df: pd.DataFrame) -> pd.DataFrame:
    g = df["open"] / df["prev_close"] - 1.0
    return _trade(df, g, "p0935", "p1600", 0.02, sign=+1, mask=g.abs() > 0.005)


def rule_s6(df: pd.DataFrame) -> pd.DataFrame:
    pm = df["p0930"] / df["prev_close"] - 1.0
    return _trade(df, pm, "p0931", "p1000", 0.02, sign=+1, mask=pm.abs() > 0.003)


def rule_s7(df: pd.DataFrame) -> pd.DataFrame:
    return _trade(df, df["p1000"] / df["open"] - 1.0, "p1000", "p1600", 0.01, sign=+1)


# ---------------------------------------------------------------- aggregation
def daily(trades: pd.DataFrame, min_count: int, cents_override: Optional[float] = None) -> pd.Series:
    """Equal-weight daily net return over symbols; days with < min_count trades dropped."""
    if trades.empty:
        return pd.Series(dtype=float)
    cents = trades["cents"] if cents_override is None else cents_override
    net = trades["gross"] - 2.0 * cents * trades["inv_px"]
    g = trades.assign(net=net).groupby("date")["net"]
    out = g.mean()[g.count() >= min_count]
    return out.sort_index()


def window(s: pd.Series, start: dt.date, end: Optional[dt.date] = None) -> pd.Series:
    idx = np.array(s.index)
    keep = idx >= start
    if end is not None:
        keep &= idx <= end
    return s[keep]


def bps(x: float) -> float:
    return x * 1e4


def break_even_cents(trades: pd.DataFrame, min_count: int) -> float:
    """Cents/share each way at which the daily mean net return is zero (binding window)."""
    t = trades[(trades["date"] >= BINDING[0]) & (trades["date"] <= BINDING[1])]
    g = t.groupby("date")
    keep = g["gross"].count() >= min_count
    gross = g["gross"].mean()[keep]
    inv = g["inv_px"].mean()[keep]
    return float(gross.mean() / (2.0 * inv.mean())) * 100.0  # dollars -> cents


# ---------------------------------------------------------------- report
def run(data_dir: Path = DEFAULT_OUT) -> Dict:
    bars = {s: load_symbol(s, data_dir) for s in SYMBOLS}
    missing = [s for s, b in bars.items() if b.empty]
    if missing:
        raise SystemExit(f"no bars for {missing}; run python -m src.intraday.alpaca_bars first")
    calendar = fetch_calendar(out_dir=data_dir)  # amendment A1: early closes from the calendar
    long = stack(build_tables(bars, TARGETS, calendar=calendar))
    sessions = sorted(long["date"].unique())
    res: Dict = {"registered": "docs/PREREG_INTRADAY_LATE_DAY_AND_OPEN_20261127.md",
                 "generated": dt.datetime.now().isoformat(timespec="seconds"),
                 "data": {"symbols": SYMBOLS, "full_sessions": len(sessions),
                          "first": str(sessions[0]), "last": str(sessions[-1])}}

    # ---- I1 primary
    t1 = rule_i1(long)
    d1 = daily(t1, 3)
    d1_stress = daily(t1, 3, cents_override=0.02)
    b = window(d1, *BINDING)
    prim = summarise(b.values, BLOCK, DRAWS, SEED)
    halves = {lab: float(window(d1, s, e).mean()) for lab, s, e in HALVES}
    stress = float(window(d1_stress, *BINDING).mean())
    if prim["n"] < MIN_N:
        verdict = "INSUFFICIENT"
    elif prim["lo"] > 0 and all(v > 0 for v in halves.values()) and stress > 0:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    per_year = {int(y): float(v) for y, v in d1.groupby([d.year for d in d1.index]).mean().items()}
    res["I1"] = {"binding": prim, "halves": halves, "stress_mean": stress, "verdict": verdict,
                 "per_year_mean": per_year, "break_even_cents_each_way": break_even_cents(t1, 3),
                 "seen": summarise(window(d1, SEEN_START).values, BLOCK, DRAWS, SEED),
                 "gross_binding_mean": float(window(daily(t1, 3, cents_override=0.0), *BINDING).mean())}

    # ---- secondaries
    secondaries: Dict[str, pd.Series] = {}
    for sym in SYMBOLS:
        secondaries[f"S1_{sym}"] = daily(t1[t1["symbol"] == sym], 1)
    for key, fn, mc in [("S2_big_move_fade", rule_s2, 3), ("S3_fade_1545", rule_s3, 3),
                        ("S4_intraday_momentum", rule_s4, 3), ("S5_gap_continuation", rule_s5, 1),
                        ("S6_premarket_into_open", rule_s6, 1), ("S7_opening_range", rule_s7, 3)]:
        secondaries[key] = daily(fn(long), mc)
    sec_bind = {k: summarise(window(v, *BINDING).values, BLOCK, DRAWS, SEED) for k, v in secondaries.items()}
    adj = holm({k: v["p"] for k, v in sec_bind.items() if v["n"] >= 2})
    res["secondary"] = {}
    for k, v in secondaries.items():
        res["secondary"][k] = {"binding": sec_bind[k], "holm": adj.get(k),
                               "seen": summarise(window(v, SEEN_START).values, BLOCK, DRAWS, SEED),
                               "halves": {lab: float(window(v, s, e).mean()) for lab, s, e in HALVES}}
    return res


def fmt(s: Dict) -> str:
    return (f"n={s['n']:>5}  mean {bps(s['mean']):+6.2f} bps/day  "
            f"95% CI [{bps(s['lo']):+6.2f}, {bps(s['hi']):+6.2f}]  p={s['p']:.4f}  hit {s['hit']:.1%}")


def print_report(res: Dict) -> None:
    d = res["data"]
    print(f"Intraday tests — {d['full_sessions']} full sessions {d['first']}..{d['last']}  ({', '.join(d['symbols'])})")
    i1 = res["I1"]
    print("\nI1 late-day fade (PRIMARY), binding 2016-01-04..2023-12-29, 1c/share each way")
    print("   " + fmt(i1["binding"]))
    print("   gross mean {:+.2f} bps/day | halves: {} | stress (2c) mean {:+.2f} bps".format(
        bps(i1["gross_binding_mean"]),
        ", ".join(f"{k} {bps(v):+.2f}" for k, v in i1["halves"].items()), bps(i1["stress_mean"])))
    print(f"   break-even cost: {i1['break_even_cents_each_way']:.2f} c/share each way")
    print("   per year: " + "  ".join(f"{y}:{bps(v):+.1f}" for y, v in i1["per_year_mean"].items()))
    print(f"   VERDICT: {i1['verdict']}")
    print("   seen window 2024+ (no authority): " + fmt(i1["seen"]))
    print("\nSecondaries (binding window; Holm across 10; no authority)")
    for k, v in res["secondary"].items():
        h = v["holm"] or {}
        flag = "SURVIVES Holm" if h.get("reject") else ""
        print(f"   {k:<24} {fmt(v['binding'])}  p_holm={h.get('p_holm', float('nan')):.3f} {flag}")
        print(f"   {'':<24} halves: " + ", ".join(f"{a} {bps(x):+.2f}" for a, x in v["halves"].items())
              + f" | seen 2024+: {bps(v['seen']['mean']):+.2f} bps (n={v['seen']['n']}, p={v['seen']['p']:.3f})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Preregistered intraday tests I1/S1-S7")
    ap.add_argument("--data", default=str(DEFAULT_OUT))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    res = run(Path(a.data))
    print_report(res)
    out = Path(a.json_out) if a.json_out else REPO / "reports" / "intraday" / (
        f"intraday_tests_{dt.datetime.now():%Y%m%dT%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
