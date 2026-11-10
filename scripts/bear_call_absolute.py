"""B1: gate-passed Bear Calls in the shadow corpus — absolute and delta-adjusted.

Registration: docs/PREREG_EARNINGS_AND_BEAR_CALL_20261109.md (frozen before running).
Read-only on data/candidates.db; daily closes from Yahoo, cached to --cache.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -u -m scripts.bear_call_absolute --cache DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

DB = "data/candidates.db"
R = 0.04
SEED = 20261109
N_BOOT = 10_000
EXP_THROUGH = "2026-11-06"
ENTRY_FROM = "2000-01-01"  # B1-F (Amendment 2): --forward sets 2026-10-30 / 2027-01-15
INDEX = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLU", "XLV", "XLI",
         "XLP", "XLY", "XLB", "XLRE", "XLC", "EEM", "EFA", "GLD", "SLV", "TLT",
         "HYG", "SMH", "XBI", "KRE", "GDX", "USO", "ARKK"}


def bs_call(S, K, s, T):
    d1 = (np.log(S / K) + (R + s * s / 2) * T) / (s * np.sqrt(T))
    return S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d1 - s * np.sqrt(T))


def call_delta(S, K, s, T):
    return norm.cdf((np.log(S / K) + (R + s * s / 2) * T) / (s * np.sqrt(T)))


def load(arm: str) -> pd.DataFrame:
    where = {"passed": "c.gate_passed = 1",
             "friction": "c.refused_by = 'friction'",
             "negative_ev": "c.refused_by = 'negative_ev'"}[arm]
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql(
        f"""SELECT c.ts, c.symbol, c.expiration, c.contract_key, c.features_json,
                   p.entry_date, p.entry_price, p.exit_date, p.pnl_pct, p.status
            FROM candidates c JOIN candidate_positions p USING (scan_id, board, contract_key)
            WHERE c.strategy_name = 'Bear Call' AND {where}
              AND c.expiration <= ? AND p.entry_date >= ?""", con,
        params=(EXP_THROUGH, ENTRY_FROM))
    con.close()
    n_open = int((df.status != "CLOSED").sum())
    df = df[(df.status == "CLOSED") & df.pnl_pct.notna()]
    df = df[~df.exit_date.str[:10].isin(["2026-08-29", "2026-08-30"])]
    df = df.sort_values("ts").groupby(["contract_key", "entry_date"], as_index=False).first()
    f = df.features_json.map(json.loads)
    for k in ("short_strike", "long_strike", "spread_width", "entry_iv",
              "short_bid", "short_ask", "long_bid", "long_ask"):
        df[k] = f.map(lambda d, k=k: d.get(k))
    df.attrs["open"] = n_open
    return df.drop(columns="features_json")


def fetch(tickers, cache: str) -> pd.DataFrame:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"bear_call_px_1d_{pd.Timestamp.today().date()}.pkl")
    # The pickle is this script's own output in a cache dir the caller chose;
    # never point --cache at a directory someone else can write.
    if os.path.exists(path):
        px = pd.read_pickle(path)
        if set(tickers) <= set(px.columns):
            return px
    import yfinance as yf
    frames = {}
    for t in sorted(tickers):
        h = yf.Ticker(t).history(start="2026-08-01", interval="1d", auto_adjust=False)
        if h is not None and not h.empty:
            s = h["Close"]
            s.index = [x.date().isoformat() for x in s.index]
            frames[t] = s
    px = pd.DataFrame(frames)
    px.to_pickle(path)
    return px


def close_on(px: pd.DataFrame, t: str, day: str) -> Optional[float]:
    if t not in px.columns:
        return None
    s = px[t].dropna()
    s = s[s.index <= day]
    return float(s.iloc[-1]) if len(s) else None


def per_row(df: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in df.itertuples():
        credit, width = float(r.entry_price), float(r.spread_width or np.nan)
        if not (width > credit > 0):
            continue
        car = width - credit
        Ks, Kl, iv = float(r.short_strike), float(r.long_strike), float(r.entry_iv or np.nan)
        T = max((pd.Timestamp(r.expiration) - pd.Timestamp(r.entry_date[:10])).days, 1) / 365
        c0 = close_on(px, r.symbol, r.entry_date[:10])
        c1 = close_on(px, r.symbol, r.exit_date[:10])
        if c0 is None or c1 is None or not np.isfinite(iv):
            continue
        S0, iv_l, implied = c0, iv, False
        try:
            ms = (r.short_bid + r.short_ask) / 2
            S = brentq(lambda S: bs_call(S, Ks, iv, T) - ms, Ks * 0.3, Ks * 3)
            if abs(S / c0 - 1) <= 0.03:
                S0, implied = S, True
                ml = (r.long_bid + r.long_ask) / 2
                iv_l = brentq(lambda s: bs_call(S0, Kl, s, T) - ml, 0.01, 5.0)
        except (ValueError, TypeError):
            pass
        d = call_delta(S0, Ks, iv, T) - call_delta(S0, Kl, iv_l, T)
        rr = float(r.pnl_pct) * credit / car
        h = -d * (c1 - S0) / car
        wk = pd.Timestamp(r.entry_date[:10]).isocalendar()
        r_cross, x_cross = cross_fills(r, credit, width, h * car)
        out.append({"symbol": r.symbol, "entry": r.entry_date[:10], "exit": r.exit_date[:10],
                    "r": rr, "h": h, "x": rr - h, "delta": d, "implied": implied,
                    "index": r.symbol in INDEX, "week": f"{wk[0]}-W{wk[1]:02d}",
                    "r_cross": r_cross, "x_cross": x_cross})
    return pd.DataFrame(out)


def cross_fills(r, credit: float, width: float,
                hedge_per_share: float) -> Tuple[float, float]:
    """(r, x) with the bid-ask crossed at BOTH ends. Amendment 3, no authority.

    The registered r is the shadow position's `pnl_pct`: entry at the `limit`
    policy (35% of the half-spread conceded) and exit at the MID, so it charges
    about a sixth of a round trip. Here:

        credit_cross = short_bid - long_ask                  (scan-time quotes)
        exit_cost    = exit_mid + (short and long half-spreads AT ENTRY)
        exit_mid     = credit * (1 - pnl_pct)                (inverts pnl_pct)
        r_cross      = (credit_cross - exit_cost) / (width - credit_cross)

    Per-leg quotes at exit are not stored, so the entry half-spreads stand in
    for them. NaN when a leg quote is missing or the crossed credit leaves no
    capital at risk to divide by.
    """
    try:
        hs = ((r.short_ask - r.short_bid) + (r.long_ask - r.long_bid)) / 2
        c_cross = float(r.short_bid - r.long_ask)
    except TypeError:
        return np.nan, np.nan
    if not np.isfinite(hs) or not np.isfinite(c_cross) or not width > c_cross:
        return np.nan, np.nan
    car = width - c_cross
    exit_mid = credit * (1 - float(r.pnl_pct))
    r_cross = (c_cross - (exit_mid + hs)) / car
    return r_cross, r_cross - hedge_per_share / car


def pf(v: np.ndarray) -> float:
    loss = -v[v < 0].sum()
    return float(v[v > 0].sum() / loss) if loss > 0 else np.inf


def cboot(df, col, by, f=np.mean) -> Dict:
    g = [x[col].values for _, x in df.groupby(by)]
    rng = np.random.default_rng(SEED)
    v = np.array([f(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]))
                  for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return {"est": float(f(df[col].values)), "lo": float(lo), "hi": float(hi), "k": len(g)}


def show(label: str, d: pd.DataFrame, by="week", claim=False) -> None:
    a, b = cboot(d, "x", by), cboot(d, "r", by, pf)
    m, hh = cboot(d, "r", by), cboot(d, "h", by)
    print(f"  {label}: n={len(d)} sym={d.symbol.nunique()} clusters={a['k']} "
          f"win {np.mean(d.r > 0):.2f}")
    print(f"    r (on CAR)        {m['est']:+.4f} [{m['lo']:+.4f},{m['hi']:+.4f}]")
    print(f"    delta-equivalent  {hh['est']:+.4f} [{hh['lo']:+.4f},{hh['hi']:+.4f}]")
    print(f"    A excess x        {a['est']:+.4f} [{a['lo']:+.4f},{a['hi']:+.4f}]")
    print(f"    B PF on r         {b['est']:.3f} [{b['lo']:.3f},{b['hi']:.3f}]")
    if claim:
        ok = a["lo"] > 0 and b["lo"] > 1
        print(f"    VERDICT: {'PASS' if ok else 'fail'} (A {'pass' if a['lo'] > 0 else 'fail'}, "
              f"B {'pass' if b['lo'] > 1 else 'fail'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--forward", action="store_true",
                    help="B1-F: entries >= 2026-10-30, expiry <= 2027-01-15; judge >= 2027-01-20")
    ap.add_argument("--split-at", metavar="YYYY-MM-DD",
                    help="also report entries before / on-or-after this date "
                         "(Amendment 3: the day the entry-date and fresh-mark "
                         "fixes merged); no authority")
    a = ap.parse_args()
    global EXP_THROUGH, ENTRY_FROM
    if a.forward:
        EXP_THROUGH, ENTRY_FROM = "2027-01-15", "2026-10-30"
    arms = {k: load(k) for k in ("passed", "friction", "negative_ev")}
    syms = set().union(*(set(d.symbol) for d in arms.values()))
    px = fetch(syms, a.cache)
    res = {}
    for k, d in arms.items():
        res[k] = per_row(d, px)
        print(f"{k}: {len(d)} obs loaded, {len(res[k])} priced, still OPEN {d.attrs['open']}, "
              f"spot implied {res[k].implied.mean():.0%}")
    p = res["passed"]
    print(f"\n== B1 PRIMARY (gate-passed Bear Calls, expiry <= {EXP_THROUGH}, "
          f"entries {p.entry.min()}..{p.entry.max()})")
    show("passed", p, claim=True)
    print("\n== Reference arms (no authority)")
    show("refused friction", res["friction"])
    show("refused negative_ev", res["negative_ev"])
    print("\n== Sensitivities (no authority)")
    show("passed, cluster by symbol", p, by="symbol")
    show("passed, drop exits 2026-10-12", p[p.exit != "2026-10-12"])
    show("passed, index ETFs", p[p["index"]])
    show("passed, single names", p[~p["index"]])
    if a.split_at:
        for label, d in ((f"passed, entries < {a.split_at}", p[p.entry < a.split_at]),
                         (f"passed, entries >= {a.split_at}", p[p.entry >= a.split_at])):
            if len(d):
                show(label, d)
            else:
                print(f"  {label}: no rows")
    print("\n== Amendment 3 (no authority): crossing the spread at both ends")
    for label, d in (("passed", p), ("passed, index ETFs", p[p["index"]]),
                     ("passed, single names", p[~p["index"]])):
        d = d.dropna(subset=["r_cross", "x_cross"])
        if not len(d):
            print(f"  {label}: no rows with both legs quoted")
            continue
        a, b = cboot(d, "x_cross", "week"), cboot(d, "r_cross", "week", pf)
        m = cboot(d, "r_cross", "week")
        print(f"  {label}: n={len(d)}")
        print(f"    r (on CAR)        {m['est']:+.4f} [{m['lo']:+.4f},{m['hi']:+.4f}]")
        print(f"    A excess x        {a['est']:+.4f} [{a['lo']:+.4f},{a['hi']:+.4f}]")
        print(f"    B PF on r         {b['est']:.3f} [{b['lo']:.3f},{b['hi']:.3f}]")
        print(f"    would pass at the touch: "
              f"{'yes' if a['lo'] > 0 and b['lo'] > 1 else 'no'}")


if __name__ == "__main__":
    main()
