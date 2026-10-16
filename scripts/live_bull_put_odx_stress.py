"""The live Bull Put rule on optionsDX SPY 2010-2023 — the registered estimator.

Registration: docs/PREREG_LIVE_BULL_PUT_ODX_STRESS_20261015.md. Change nothing
here without amending it. Read-only on data/optionsdx.db.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -u -m scripts.live_bull_put_odx_stress \
        --csv /path/to/trades.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.alloc.engine import crossing_cost, replay
from src.alloc.fills import quotes_from_chain, _key as _fkey
from src.alloc.settle import implied_spot_any
from src.optionsdx import OdxChainSource, available_dates, terminal_date
from src.strategies.spec import StrategySpec

DB = "data/optionsdx.db"
SYM = "SPY"
START, END = "2010-01-01", "2023-12-31"
N_BOOT = 10_000
SEED = 20261015
R_RATE, Q_DIV = 0.02, 0.015
STRESS_YEARS = (2011, 2015, 2018, 2020, 2022)
NOTIONAL, RISK_PER_TRADE = 100_000.0, 0.02

LIVE_ENTRY = {"dte": [14, 31], "short_delta": 0.32, "width": 5.0,
              "width_pct_of_spot": 0.015, "min_credit_to_width": 0.20,
              "max_friction_pct_of_credit": 0.25}
LIVE_EXIT = {"profit_target": 0.5, "stop": 1.0, "time_exit_dte": 21,
             "min_days_held": 3, "hold_to_expiry": False}


class SpotRecordingSource:
    """Pass-through source that remembers each date's implied spot.

    The engine reads every date's chain anyway, so the spot series costs no
    extra reads and is the same parity-implied spot the engine settles on.
    """

    def __init__(self, inner: OdxChainSource):
        self.inner = inner
        self.spot: Dict[str, Optional[float]] = {}

    def chain(self, symbol: str, date: str) -> List[Dict[str, Any]]:
        rows = self.inner.chain(symbol, date)
        if date not in self.spot:
            self.spot[date] = implied_spot_any(rows) if rows else None
        return rows


def make_spec(entry: Dict, exit_: Dict, label: str) -> StrategySpec:
    return StrategySpec(id=label, version=1, structure="bull_put", universe={},
                        entry=entry, exit=exit_,
                        sizing={"max_capital_at_risk": 1e9, "max_concurrent": 1000},
                        created="2026-10-15", trial_count=1)


def run(entry: Dict, exit_: Dict, label: str, dates: List[str], term: Dict
        ) -> Tuple[List[Any], Dict[str, Optional[float]], Dict[str, int]]:
    src = SpotRecordingSource(OdxChainSource(DB))
    try:
        trades, stats = replay(make_spec(entry, exit_, label), [SYM], dates, src,
                               terminal=term, splits={}, stratum_of={SYM: "mega"})
    finally:
        src.inner.close()
    return trades, src.spot, stats


# ── realised vol, the gate's basis ──────────────────────────────────────────
def rv_basis(spot: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    s = pd.Series({d: v for d, v in spot.items() if v}).sort_index()
    r = np.log(s).diff().dropna()
    out: Dict[str, Optional[float]] = {}
    for i, d in enumerate(r.index):
        past = r.iloc[: i + 1]
        h30 = past.iloc[-30:].std(ddof=1) * np.sqrt(252) if len(past) >= 30 else None
        h252 = past.iloc[-252:].std(ddof=1) * np.sqrt(252) if len(past) >= 252 else None
        if h30 is not None and h252 is not None:
            out[d] = 0.5 * h30 + 0.5 * h252
        else:
            out[d] = h252 if h252 is not None else h30
    return out


def bs_put(S, K, s, T):
    d1 = (np.log(S / K) + (R_RATE - Q_DIV + s * s / 2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    return K * np.exp(-R_RATE * T) * norm.cdf(-d2) - S * np.exp(-Q_DIV * T) * norm.cdf(-d1)


def frame(trades: List[Any], spot: Dict[str, Optional[float]],
          with_gate: bool) -> pd.DataFrame:
    """One row per closed trade, with excess over delta and (optionally) the gate."""
    rv = rv_basis(spot) if with_gate else {}
    reader = OdxChainSource(DB, cache_dates=2)
    rows = []
    try:
        for t in trades:
            if not t.exit_date or t.exit_reason == "ticker_ended" or not t.capital_at_risk:
                continue
            f = t.features
            s0, s1 = spot.get(t.entry_date), spot.get(t.exit_date)
            sd, ld = f.get("short_delta"), f.get("long_delta")
            if s0 is None or s1 is None or sd is None or ld is None:
                continue
            car = float(t.capital_at_risk)
            H = (sd - ld) * 100.0 * (s1 - s0)
            row = dict(entry=t.entry_date, exit=t.exit_date, reason=t.exit_reason,
                       expiration=t.expiration,
                       dte=f.get("dte"), short_delta=sd, long_delta=ld,
                       width=f.get("width"), credit=t.entry_price, car=car,
                       pnl=float(t.pnl), H=H, roc=float(t.pnl) / car,
                       x=(float(t.pnl) - H) / car, s0=s0, s1=s1,
                       fric_entry=f.get("friction"))
            # exit crossing, for the mid-fill sensitivity: none at expiry
            fric_exit = 0.0
            if t.exit_reason != "expiry":
                q = quotes_from_chain(reader.chain(SYM, t.exit_date))
                fx = crossing_cost(t.legs, q)
                fric_exit = fx if fx is not None else np.nan
            fe = row["fric_entry"] if row["fric_entry"] is not None else np.nan
            row["roc_mid"] = (float(t.pnl) + 100.0 * (fe + fric_exit)) / car
            row["x_mid"] = row["roc_mid"] - H / car
            if with_gate:
                row.update(gate_fields(t, reader.chain(SYM, t.entry_date), s0,
                                       rv.get(t.entry_date)))
            rows.append(row)
    finally:
        reader.close()
    df = pd.DataFrame(rows)
    df["month"] = df.entry.str[:7]
    df["year"] = df.entry.str[:4].astype(int)
    return df


def gate_fields(t, chain, S, sigma) -> Dict[str, Any]:
    by = {_fkey(c["expiration"], c["strike"], c["type"]): c for c in chain}
    short = next(l for l in t.legs if l.action == "sell")
    long_ = next(l for l in t.legs if l.action == "buy")
    cs = by.get(_fkey(short.expiration, short.strike, short.type))
    cl = by.get(_fkey(long_.expiration, long_.strike, long_.type))
    out: Dict[str, Any] = {"take": np.nan, "skew_pen": np.nan, "gross": np.nan}
    if cs is None or cl is None or not sigma:
        return out
    T = max((dt.date.fromisoformat(short.expiration[:10])
             - dt.date.fromisoformat(t.entry_date)).days, 1) / 365.0
    mid_s = (cs["bid"] + cs["ask"]) / 2
    mid_l = (cl["bid"] + cl["ask"]) / 2
    g_s = 100 * (bs_put(S, short.strike, sigma, T) - mid_s)
    g_l = 100 * (bs_put(S, long_.strike, sigma, T) - mid_l)
    gross = g_l - g_s
    cost = 100 * ((cs["ask"] - cs["bid"]) + (cl["ask"] - cl["bid"]))
    net = gross - cost
    out["gross"], out["net"], out["cost"] = gross, net, cost
    out["take"] = float(gross > 0 and net > 0.25 * cost)
    iv = cs.get("iv")
    w = short.strike - long_.strike
    if iv and iv > 0.02 and w > 0:
        flat = bs_put(S, short.strike, iv, T) - bs_put(S, long_.strike, iv, T)
        out["skew_pen"] = (flat - (mid_s - mid_l)) / w
    out["sigma_rv"], out["short_iv"] = sigma, iv
    return out


# ── statistics ──────────────────────────────────────────────────────────────
def pf(v: np.ndarray) -> float:
    neg = -v[v < 0].sum()
    return float(v[v > 0].sum() / neg) if neg > 0 else float("inf")


def month_boot(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float]) -> Dict:
    groups = {k: g for k, g in df.groupby("month")}
    keys = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = stat(pd.concat([groups[k] for k in rng.choice(keys, len(keys))]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"est": stat(df), "lo": float(lo), "hi": float(hi), "k": len(keys)}


def fmt(r: Dict, pct: bool = True) -> str:
    f = (lambda v: f"{100 * v:+.2f}%") if pct else (lambda v: f"{v:.3f}")
    return f"{f(r['est'])} [{f(r['lo'])}, {f(r['hi'])}]  months={r['k']}"


def ha_verdict(r: Dict) -> str:
    return ("index premium exists" if r["lo"] > 0 else
            "costly beta" if r["hi"] < 0 else "inconclusive")


def hb_verdict(r: Dict) -> str:
    return ("gate separates correctly" if r["hi"] < 0 else
            "gate inverted" if r["lo"] > 0 else "no separation")


def drawdown(df: pd.DataFrame) -> Dict[str, Any]:
    pnl = (df.roc * NOTIONAL * RISK_PER_TRADE).groupby(df.exit).sum().sort_index()
    eq = NOTIONAL + pnl.cumsum()
    peak = eq.cummax()
    dd = (peak - eq) / NOTIONAL
    trough = dd.idxmax()
    peak_date = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    rec = after[after >= eq.loc[peak_date]]
    rec_date = rec.index[0] if len(rec) else None
    days = ((dt.date.fromisoformat(rec_date) - dt.date.fromisoformat(peak_date)).days
            if rec_date else None)
    return {"max_dd": float(dd.max()), "peak": peak_date, "trough": trough,
            "recovered": rec_date, "days": days, "final": float(eq.iloc[-1] / NOTIONAL - 1),
            "series": eq}


def year_table(df: pd.DataFrame, spot: Dict[str, Optional[float]]) -> pd.DataFrame:
    s = pd.Series({d: v for d, v in spot.items() if v}).sort_index()
    eq = drawdown(df)["series"]
    out = []
    for y, g in df.groupby("year"):
        sy = s[s.index.str.startswith(str(y))]
        ey = eq[eq.index.str.startswith(str(y))]
        ydd = float(((ey.cummax() - ey) / NOTIONAL).max()) if len(ey) else np.nan
        out.append(dict(year=y, n=len(g), win=(g.pnl > 0).mean(), roc=g.roc.mean(),
                        x=g.x.mean(), pf=pf(g.roc.values),
                        book_pct=(g.roc * RISK_PER_TRADE).sum(),
                        intra_dd=ydd, spy=sy.iloc[-1] / sy.iloc[0] - 1))
    return pd.DataFrame(out).set_index("year")


def headline(df: pd.DataFrame, label: str) -> Dict:
    r = month_boot(df, lambda d: d.x.mean())
    roc = month_boot(df, lambda d: d.roc.mean())
    print(f"  {label:34s} n={len(df):5d}  RoC {fmt(roc)}")
    print(f"  {'':34s} excess {fmt(r)} -> {ha_verdict(r)}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--skip-sens", action="store_true")
    a = ap.parse_args()
    dates = available_dates(DB, SYM, START, END)
    last = terminal_date(DB, SYM)
    term = {SYM: last} if last else {}
    print(f"optionsDX {SYM} {dates[0]}..{dates[-1]} ({len(dates)} dates)")

    trades, spot, stats = run(LIVE_ENTRY, LIVE_EXIT, "live", dates, term)
    print("engine stats:", {k: v for k, v in stats.items() if v})
    df = frame(trades, spot, with_gate=True)
    if a.csv:
        df.to_csv(a.csv, index=False)
    print(f"closed trades in the return series: {len(df)} | months {df.month.nunique()}"
          f" | median DTE {df.dte.median():.0f} | median hold "
          f"{(pd.to_datetime(df.exit) - pd.to_datetime(df.entry)).dt.days.median():.0f}d"
          f" | median width/spot {(df.width / df.s0).median():.4f}"
          f" | median short delta {df.short_delta.median():.3f}")
    print("exit reasons:", df.reason.value_counts().to_dict())

    print("\n== H-A (primary): live rule on SPY, excess over its own entry delta")
    ha = month_boot(df, lambda d: d.x.mean())
    print(f"  mean excess x      {fmt(ha)}")
    print(f"  VERDICT: {ha_verdict(ha)}")
    print(f"  mean RoC           {fmt(month_boot(df, lambda d: d.roc.mean()))}")
    print(f"  mean delta-equiv   {fmt(month_boot(df, lambda d: (d.H / d.car).mean()))}")
    print(f"  PF on CaR          {fmt(month_boot(df, lambda d: pf(d.roc.values)), pct=False)}")
    print(f"  win rate           {(df.pnl > 0).mean():.3f}")
    dd = drawdown(df)
    print(f"  fixed-fractional book (2% of $100k per trade, not compounded): "
          f"total {100 * dd['final']:+.1f}% | max DD {100 * dd['max_dd']:.1f}% of notional "
          f"(peak {dd['peak']}, trough {dd['trough']}, recovered {dd['recovered']}, "
          f"{dd['days']} days)")

    yt = year_table(df, spot)
    print("\n  per year:")
    print(yt.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("  stress years:", ", ".join(
        f"{y}: RoC {yt.loc[y, 'roc']:+.3f} x {yt.loc[y, 'x']:+.3f} intraDD {yt.loc[y, 'intra_dd']:.3f}"
        for y in STRESS_YEARS if y in yt.index))

    print("\n== H-B: the gate's flat-RV EV verdict on these trades")
    g = df[df["take"].notna()]
    print(f"  gated rows {len(g)} | TAKE share {g['take'].mean():.3f} "
          f"| gross<=0 share {(g.gross <= 0).mean():.3f}")
    for arm, sub in (("TAKE", g[g["take"] == 1]), ("refused", g[g["take"] == 0])):
        if len(sub):
            print(f"  {arm:8s} n={len(sub):5d}  RoC {fmt(month_boot(sub, lambda d: d.roc.mean()))}"
                  f"\n  {'':8s}           x   {fmt(month_boot(sub, lambda d: d.x.mean()))}")

    def _diff(d: pd.DataFrame) -> float:
        a, b = d[d["take"] == 0].x, d[d["take"] == 1].x
        return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
    hb = month_boot(g, _diff)
    print(f"  refused − TAKE (x) {fmt(hb)}")
    print(f"  VERDICT: {hb_verdict(hb)}")

    print("\n  mechanism (no authority): mean x by skew-penalty quintile")
    m = g[g.skew_pen.notna()].copy()
    m["q"] = pd.qcut(m.skew_pen, 5, labels=False) + 1
    for q, sub in m.groupby("q"):
        print(f"    Q{q} pen {sub.skew_pen.median():+.4f}  TAKE {sub['take'].mean():.2f}  "
              f"x {fmt(month_boot(sub, lambda d: d.x.mean()))}")

    print("\n== Sensitivities (no authority)")
    mid = df[df.x_mid.notna()].copy()
    mid["x"], mid["roc"] = mid.x_mid, mid.roc_mid
    headline(mid, "mid fills (entry+exit crossing back)")
    if not a.skip_sens:
        for label, entry, exit_ in (
                ("DTE [8,31]", {**LIVE_ENTRY, "dte": [8, 31]}, LIVE_EXIT),
                ("hold to expiry", LIVE_ENTRY, {"hold_to_expiry": True}),
                ("25 delta", {**LIVE_ENTRY, "short_delta": 0.25}, LIVE_EXIT)):
            tr, sp, _ = run(entry, exit_, label, dates, term)
            headline(frame(tr, sp, with_gate=False), label)


if __name__ == "__main__":
    main()
