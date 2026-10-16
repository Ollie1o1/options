"""Bull Put book vs holding its own entry delta — the registered estimator.

Registration: docs/PREREG_DELTA_MATCHED_BENCHMARK_20261015.md. Read-only on the
ledger; underlying prices from Yahoo (daily + 1-hour bars), cached to --cache.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.delta_matched_benchmark
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

LEDGER = "paper_trades.db"
TZ = "America/New_York"
R_RATE = 0.04
N_BOOT = 10_000
SEED = 20261015


def _d1(S, K, s, T, q):
    return (np.log(S / K) + (R_RATE - q + s * s / 2) * T) / (s * np.sqrt(T))


def bs_put(S, K, s, T, q=0.0):
    d1 = _d1(S, K, s, T, q)
    d2 = d1 - s * np.sqrt(T)
    return K * np.exp(-R_RATE * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def put_delta(S, K, s, T, q=0.0):
    return -np.exp(-q * T) * norm.cdf(-_d1(S, K, s, T, q))


def load_trades(path: str, exit_through: Optional[str] = None) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    df = pd.read_sql(
        """SELECT entry_id, date, ticker, expiration, strike, long_strike,
                  entry_iv, dividend_yield, quantity, capital_at_risk, pnl_usd,
                  exit_date, short_bid_entry, short_ask_entry,
                  long_bid_entry, long_ask_entry
           FROM trades WHERE strategy_name='Bull Put' AND status='CLOSED'
             AND date >= '2026-08-20' AND pnl_usd IS NOT NULL
             AND (? IS NULL OR substr(exit_date, 1, 10) <= ?)""",
        con, params=(exit_through, exit_through))
    con.close()
    return df


def fetch_prices(tickers, start: str, cache: str) -> Dict[str, pd.DataFrame]:
    os.makedirs(cache, exist_ok=True)
    out = {}
    import yfinance as yf
    for interval in ("1d", "1h"):
        # The pickle is this script's own output in a cache dir the caller
        # chose; never point --cache at a directory someone else can write.
        f = os.path.join(cache, f"px_{interval}.pkl")
        if os.path.exists(f):
            out[interval] = pd.read_pickle(f)
            continue
        frames = {}
        for t in tickers:
            h = yf.Ticker(t).history(start=start, interval=interval, auto_adjust=False)
            if h is None or h.empty:
                raise RuntimeError(f"no {interval} bars for {t}")
            frames[t] = h["Close"]
        px = pd.DataFrame(frames)
        px.to_pickle(f)
        out[interval] = px
    return out


def spot_at_exit(px1h: pd.DataFrame, px1d: pd.DataFrame, t: str, ts: str) -> Optional[float]:
    when = pd.Timestamp(ts).tz_localize(TZ)
    s = px1h[t].dropna()
    s.index = s.index.tz_convert(TZ)
    # A bar is stamped at its open; the bar containing `when` opened at or before it.
    s = s[s.index <= when]
    if len(s) and s.index[-1].date() == when.date():
        return float(s.iloc[-1])
    d = px1d[t].dropna()
    d.index = d.index.tz_convert(TZ).date
    hit = d[[x <= when.date() for x in d.index]]
    return float(hit.iloc[-1]) if len(hit) else None


def daily_close(px1d: pd.DataFrame, t: str, day: str) -> Optional[float]:
    d = px1d[t].dropna()
    d.index = [x.date().isoformat() for x in d.index.tz_convert(TZ)]
    return float(d[day]) if day in d.index else None


def per_trade(df: pd.DataFrame, px: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        q = float(r.dividend_yield or 0.0)
        T = ((pd.Timestamp(r.expiration) + pd.Timedelta(hours=16))
             - (pd.Timestamp(r.date) + pd.Timedelta(hours=12))).total_seconds() / (365 * 86400)
        Ks, Kl, iv = float(r.strike), float(r.long_strike), float(r.entry_iv)
        close0 = daily_close(px["1d"], r.ticker, r.date)
        fallback = r.short_bid_entry is None or pd.isna(r.short_bid_entry) \
            or r.long_bid_entry is None or pd.isna(r.long_bid_entry)
        S0 = None
        iv_l = iv
        if not fallback:
            ms = (r.short_bid_entry + r.short_ask_entry) / 2
            ml = (r.long_bid_entry + r.long_ask_entry) / 2
            try:
                S0 = brentq(lambda S: bs_put(S, Ks, iv, T, q) - ms, Ks * 0.3, Ks * 3)
                if close0 is not None and abs(S0 / close0 - 1) > 0.03:
                    S0 = None
                else:
                    iv_l = brentq(lambda s: bs_put(S0, Kl, s, T, q) - ml, 0.01, 5.0)
            except ValueError:
                S0 = None
            if S0 is None:
                fallback = True
                iv_l = iv
        if S0 is None:
            S0 = close0
        S1 = spot_at_exit(px["1h"], px["1d"], r.ticker, r.exit_date)
        if S0 is None or S1 is None:
            raise RuntimeError(f"no spot for trade {r.entry_id} {r.ticker}")
        delta = abs(put_delta(S0, Ks, iv, T, q)) - abs(put_delta(S0, Kl, iv_l, T, q))
        H = delta * 100 * float(r.quantity) * (S1 - S0)
        X = float(r.pnl_usd) - H
        wk = pd.Timestamp(r.date).isocalendar()
        rows.append(dict(entry_id=r.entry_id, ticker=r.ticker, date=r.date,
                         exit=r.exit_date[:10], qty=r.quantity, S0=S0, S1=S1,
                         move=S1 / S0 - 1, delta=delta, pnl=float(r.pnl_usd), H=H, X=X,
                         car=float(r.capital_at_risk), x=X / float(r.capital_at_risk),
                         p=float(r.pnl_usd) / float(r.capital_at_risk),
                         h=H / float(r.capital_at_risk), s0_fallback=fallback,
                         week=f"{wk[0]}-W{wk[1]:02d}"))
    return pd.DataFrame(rows)


def pf(v: np.ndarray) -> float:
    """Profit factor on capital at risk: sum of gains over sum of losses.

    docs/DECISION_RULE_20261015.md §2 condition B, on r_i = pnl_usd /
    capital_at_risk. Infinite with no losses; zero with no gains.
    """
    v = np.asarray(v, dtype=float)
    loss = -v[v < 0].sum()
    gain = v[v > 0].sum()
    if loss <= 0:
        return float("inf") if gain > 0 else float("nan")
    return float(gain / loss)


def cluster_boot(df: pd.DataFrame, col: str, by: str,
                 stat: Callable[[np.ndarray], float] = np.mean) -> Dict:
    """Cluster bootstrap of `stat` over `col`. The default (mean) draws exactly
    as the registered excess estimator always has, so its numbers cannot move."""
    groups = {k: g[col].values for k, g in df.groupby(by)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = stat(np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"est": float(stat(df[col].values)), "lo": float(lo), "hi": float(hi),
            "k": len(keys)}


def pf_verdict(r: Dict) -> str:
    """Condition B: the CI's lower bound must clear 1."""
    return "profitable" if r["lo"] > 1 else "unprofitable" if r["hi"] < 1 else "inconclusive"


def verdict(r: Dict) -> str:
    return "beats its delta" if r["lo"] > 0 else "costly beta" if r["hi"] < 0 else "inconclusive"


def fmt(r: Dict) -> str:
    return f"{r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] clusters={r['k']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--exit-through", metavar="YYYY-MM-DD",
                    help="only trades closed on or before this date; the "
                         "decision rule runs with 2027-01-15")
    a = ap.parse_args()
    trades = load_trades(a.ledger, a.exit_through)
    px = fetch_prices(sorted(trades.ticker.unique()), "2026-08-10", a.cache)
    t = per_trade(trades, px)
    if a.csv:
        t.to_csv(a.csv, index=False)
    print(f"trades {len(t)} | symbols {t.ticker.nunique()} | weeks {t.week.nunique()}"
          f" | S0 fallback {int(t.s0_fallback.sum())}")
    print(f"median delta/share {t.delta.median():.3f} | median underlying move {t.move.median():+.4f}")
    print(f"\nSum pnl_usd ${t.pnl.sum():+,.2f} | Sum H (delta-equivalent) ${t.H.sum():+,.2f}"
          f" | Sum X (excess) ${t.X.sum():+,.2f} | capital at risk ${t.car.sum():,.0f}")
    print("\nmean per trade on capital at risk, week-clustered:")
    for col, name in (("p", "book"), ("h", "delta-equivalent"), ("x", "EXCESS")):
        print(f"  {name:17s} {fmt(cluster_boot(t, col, 'week'))}")
    prim = cluster_boot(t, "x", "week")
    print(f"\nVERDICT: {verdict(prim)}")
    pfr = cluster_boot(t, "p", "week", stat=pf)
    print(f"PF on capital at risk  {pfr['est']:.3f} [{pfr['lo']:.3f}, {pfr['hi']:.3f}]"
          f" clusters={pfr['k']} -> {pf_verdict(pfr)}")
    print(f"DECISION RULE §2: A {'PASS' if prim['lo'] > 0 else 'FAIL'} "
          f"(excess CI lower > 0) | B {'PASS' if pfr['lo'] > 1 else 'FAIL'} "
          f"(PF CI lower > 1)")
    print("\nsensitivities (no authority):")
    r = cluster_boot(t, "x", "ticker")
    print(f"  cluster by symbol      {fmt(r)} -> {verdict(r)}")
    sub = t[~t.s0_fallback]
    r = cluster_boot(sub, "x", "week")
    print(f"  drop S0 fallback (n={len(sub)}) {fmt(r)} -> {verdict(r)}")
    print(f"\ncorr(book, delta-equiv) per trade: {t[['p', 'h']].corr().iloc[0, 1]:.3f}"
          f" | sd(book) {t.p.std():.4f} sd(excess) {t.x.std():.4f}")


if __name__ == "__main__":
    main()
