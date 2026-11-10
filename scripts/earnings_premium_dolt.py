"""E1: selling earnings vol with defined risk, on Dolt EOD chains 2020-2026.

Registration: docs/PREREG_EARNINGS_AND_BEAR_CALL_20261109.md (frozen before running).

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -u -m scripts.earnings_premium_dolt

E2 (Amendment 3) is the same rule on our own chain archive, judged on or after
2027-03-15:

    ... -m scripts.earnings_premium_dolt --forward             # the E2 run
    ... -m scripts.earnings_premium_dolt --forward --coverage  # event counts only
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DB = "data/dolt_options.db"
# E2: chains from our own daily archive; the calendar stays the Dolt cache.
ARCHIVE_DB = "data/chain_archive.db"
E2_FROM, E2_TO = "2026-10-01", "2027-02-28"
E2_JUDGE_FROM = "2027-03-15"
SEED = 20261109
N_BOOT = 10_000
WINDOW = 4
LVL = 1 - 0.05 / 3


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def events(con, snaps: Optional[pd.DataFrame] = None,
           date_from: Optional[str] = None,
           date_to: Optional[str] = None) -> pd.DataFrame:
    """Earnings events with a chain snapshot on each side.

    `snaps` (columns symbol, date) names the days a chain exists; by default
    the Dolt chains in `con`. `date_from`/`date_to` bound the EVENT date,
    inclusive. E1 passes none of the three and is unchanged.
    """
    ec = pd.read_sql("SELECT symbol, date, whn FROM earnings_cal", con)
    ec = ec[ec.whn.isin(["Before market open", "After market close"])]
    if date_from:
        ec = ec[ec.date >= date_from]
    if date_to:
        ec = ec[ec.date <= date_to]
    if snaps is None:
        snaps = pd.read_sql("SELECT DISTINCT symbol, date FROM dolt_chain", con)
    days = {s: sorted(_d(x) for x in g.date) for s, g in snaps.groupby("symbol")}
    rows = []
    for e in ec.itertuples():
        if e.symbol not in days:
            continue
        ed = _d(e.date)
        amc = e.whn.startswith("After")
        pre_lim = ed if amc else ed - dt.timedelta(1)
        post_from = ed + dt.timedelta(1) if amc else ed
        ds = days[e.symbol]
        pre = [x for x in ds if pre_lim - dt.timedelta(WINDOW) <= x <= pre_lim]
        post = [x for x in ds if post_from <= x <= post_from + dt.timedelta(WINDOW)]
        if pre and post:
            rows.append({"symbol": e.symbol, "event": e.date, "amc": amc,
                         "D": post_from.isoformat(), "pre": pre[-1].isoformat(),
                         "post": post[0].isoformat()})
    return pd.DataFrame(rows, columns=["symbol", "event", "amc", "D", "pre", "post"])


def chain(con, sym: str, day: str) -> pd.DataFrame:
    return pd.read_sql("SELECT expiration, strike, type, bid, ask, mid, delta "
                       "FROM dolt_chain WHERE symbol=? AND date=?", con, params=(sym, day))


def archive_snaps(acon) -> pd.DataFrame:
    return pd.read_sql("SELECT DISTINCT symbol, snap_date AS date "
                       "FROM chain_snapshots", acon)


def archive_chain(acon, sym: str, day: str) -> pd.DataFrame:
    """One day's chain from `chain_snapshots`, in `chain`'s columns.

    The archive stores no mid; it is the quote midpoint, which is what Dolt's
    `mid` column is. Type is lower-cased so 'call'/'put' compare as in E1.
    """
    df = pd.read_sql("SELECT expiration, strike, LOWER(type) AS type, bid, ask, "
                     "(bid + ask) / 2.0 AS mid, delta FROM chain_snapshots "
                     "WHERE symbol=? AND snap_date=?", acon, params=(sym, day))
    return df.dropna(subset=["bid", "ask"])


def _ok(r) -> bool:
    return r.bid > 0 and r.ask >= r.bid and r.mid > 0 and (r.ask - r.bid) / r.mid <= 0.5


def _near(df: pd.DataFrame, target: float) -> Optional[pd.Series]:
    df = df.dropna(subset=["delta"])
    df = df[[_ok(r) for r in df.itertuples()]] if len(df) else df
    if df.empty:
        return None
    return df.loc[(df.delta - target).abs().idxmin()]


def legs_for(c: pd.DataFrame, kind: str) -> Optional[Dict[str, tuple]]:
    """Return {name: (strike, type, sign)}; sign -1 short, +1 long."""
    calls, puts = c[c.type == "call"], c[c.type == "put"]
    if kind == "condor":
        sc, sp = _near(calls, 0.25), _near(puts, -0.25)
        if sc is None or sp is None:
            return None
        ks, kp = sc.strike, sp.strike
    else:
        atm = _near(calls, 0.50)
        if atm is None:
            return None
        ks = kp = atm.strike
        p = puts[puts.strike == ks]
        if p.empty or not _ok(p.iloc[0]):
            return None
        if kind == "straddle":
            return {"c": (ks, "call", +1), "p": (kp, "put", +1)}
    lc = _near(calls[calls.strike > ks], 0.10)
    lp = _near(puts[puts.strike < kp], -0.10)
    if lc is None or lp is None:
        return None
    return {"sc": (ks, "call", -1), "sp": (kp, "put", -1),
            "lc": (lc.strike, "call", +1), "lp": (lp.strike, "put", +1)}


def trade(pre: pd.DataFrame, post: pd.DataFrame, kind: str, mid: bool) -> Dict:
    """Outcome dict, or {'skip': reason}."""
    exps = sorted(e for e in pre.expiration.unique() if e > post.attrs["day"])
    if not exps:
        return {"skip": "no_expiry"}
    exp = exps[0]
    c0 = pre[pre.expiration == exp]
    legs = legs_for(c0, kind)
    if legs is None:
        return {"skip": "no_legs"}
    q0 = c0.set_index(["strike", "type"])
    p1 = post[post.expiration == exp]
    q1 = p1.set_index(["strike", "type"])
    cash0 = cash1 = 0.0          # cash received (+) / paid (−) per share
    interp = False
    for k, t, sgn in legs.values():
        r0 = q0.loc[(k, t)]
        if (k, t) in q1.index:
            r1 = q1.loc[(k, t)]
        else:
            r1 = _interp(p1[p1.type == t], k)   # Amendment 1
            if r1 is None:
                return {"skip": "unbracketed_at_exit"}
            interp = True
        if not (r1.ask > 0):
            return {"skip": "unquoted_at_exit"}
        if mid:
            cash0 -= sgn * r0.mid
            cash1 += sgn * r1.mid
        else:
            cash0 -= sgn * (r0.ask if sgn > 0 else r0.bid)
            cash1 += sgn * (r1.bid if sgn > 0 else r1.ask)
    pnl = 100.0 * (cash0 + cash1)
    if kind == "straddle":
        car = -100.0 * cash0
    else:
        w = max(legs["lc"][0] - legs["sc"][0], legs["sp"][0] - legs["lp"][0])
        car = 100.0 * (w - cash0)
    if car <= 0 or (kind != "straddle" and cash0 <= 0):
        return {"skip": "bad_price"}
    return {"exp": exp, "pnl": pnl, "car": car, "r": pnl / car, "interp": interp}


def _interp(side: pd.DataFrame, k: float) -> Optional[pd.Series]:
    """Linear interpolation of bid/ask/mid in strike between stored neighbours."""
    lo, hi = side[side.strike < k], side[side.strike > k]
    if lo.empty or hi.empty:
        return None
    a, b = lo.loc[lo.strike.idxmax()], hi.loc[hi.strike.idxmin()]
    w = (k - a.strike) / (b.strike - a.strike)
    return pd.Series({c: (1 - w) * a[c] + w * b[c] for c in ("bid", "ask", "mid")})


def pf(r: np.ndarray) -> float:
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else np.inf


def cboot(df: pd.DataFrame, f, lvl: float) -> Dict[str, float]:
    g = [x.r.values for _, x in df.groupby("week")]
    rng = np.random.default_rng(SEED)
    v = np.array([f(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]))
                  for _ in range(N_BOOT)])
    a = (1 - lvl) / 2
    return {"est": f(df.r.values), "lo": float(np.quantile(v, a)),
            "hi": float(np.quantile(v, 1 - a)), "k": len(g)}


def judge(label: str, df: pd.DataFrame, claim=True) -> Dict:
    m, p = cboot(df, np.mean, 0.95), cboot(df, pf, 0.95)
    mb, pb = cboot(df, np.mean, LVL), cboot(df, pf, LVL)
    ok = mb["lo"] > 0 and pb["lo"] > 1
    print(f"  {label}: n={len(df)} sym={df.symbol.nunique()} weeks={m['k']} "
          f"win {np.mean(df.r > 0):.2f}")
    print(f"    mean r {m['est']:+.4f} 95% [{m['lo']:+.4f},{m['hi']:+.4f}] "
          f"98.33% [{mb['lo']:+.4f},{mb['hi']:+.4f}]")
    print(f"    PF {p['est']:.3f} 95% [{p['lo']:.3f},{p['hi']:.3f}] 98.33% lo {pb['lo']:.3f}"
          f" | median r {df.r.median():+.4f} | worst {df.r.min():+.3f}")
    if claim:
        print(f"    VERDICT: {'PASS' if ok else 'fail'}")
    return {"pass": ok}


def run(con, ev: pd.DataFrame, kind: str, mid=False, get_chain=chain) -> pd.DataFrame:
    out: List[Dict] = []
    skips: Dict[str, int] = {}
    cache: Dict[tuple, pd.DataFrame] = {}

    def get(sym, day):
        if (sym, day) not in cache:
            cache[(sym, day)] = get_chain(con, sym, day)
        return cache[(sym, day)]

    for e in ev.itertuples():
        pre, post = get(e.symbol, e.pre), get(e.symbol, e.post)
        post.attrs["day"] = e.post
        t = trade(pre, post, kind, mid)
        if "skip" in t:
            skips[t["skip"]] = skips.get(t["skip"], 0) + 1
            continue
        wk = _d(e.D).isocalendar()
        out.append({**e._asdict(), **t, "week": f"{wk[0]}-W{wk[1]:02d}",
                    "year": int(e.D[:4])})
    df = pd.DataFrame(out, columns=None if out else
                      ["symbol", "event", "amc", "D", "pre", "post", "exp", "pnl",
                       "car", "r", "interp", "week", "year"])
    tot = max(len(ev), 1)
    print(f"  [{kind}{' mid' if mid else ''}] traded {len(df)}/{tot}; skipped {skips} "
          f"(unquoted at exit {skips.get('unquoted_at_exit', 0) / tot:.1%})")
    return df


def forward(a) -> None:
    """E2: E1-S and E1-F on the chain archive, earnings 2026-10-01..2027-02-28.

    No claimant is judged for a pass: the registration gives E2 the job of
    confirming sign and cost on ~30-60 events, not of reaching significance.
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    acon = sqlite3.connect(f"file:{ARCHIVE_DB}?mode=ro", uri=True)
    snaps = archive_snaps(acon)
    cal = pd.read_sql("SELECT symbol, date, whn FROM earnings_cal "
                      "WHERE date BETWEEN ? AND ?", con, params=(E2_FROM, E2_TO))
    cal = cal[cal.symbol.isin(set(snaps.symbol))]
    ev = events(con, snaps, E2_FROM, E2_TO)
    timed = cal.whn.isin(["Before market open", "After market close"])
    print(f"E2 window {E2_FROM}..{E2_TO} | archive {snaps.date.min()}..{snaps.date.max()}, "
          f"{snaps.date.nunique()} days, {snaps.symbol.nunique()} symbols")
    print(f"calendar events on archived symbols: {len(cal)} "
          f"({int(timed.sum())} with a known session) | "
          f"with a snapshot within {WINDOW} days on BOTH sides: {len(ev)}")
    lost = sorted(set(cal[timed].symbol + " " + cal[timed].date)
                  - set(ev.symbol + " " + ev.event))
    if lost:
        print(f"  no usable snapshot pair ({len(lost)}): {', '.join(lost)}")
    if a.coverage:
        return
    today = dt.date.today().isoformat()
    if today < E2_JUDGE_FROM and not a.early:
        raise SystemExit(f"E2 is judged on or after {E2_JUDGE_FROM}; today is "
                         f"{today}. --coverage prints counts with no outcomes; "
                         "--early overrides and must be disclosed in the result.")
    frames = {}
    for kind, label in (("condor", "E2-S short 25/10 condor"),
                        ("fly", "E2-F short ATM iron fly")):
        print(f"\n== {label}")
        df = run(acon, ev, kind, get_chain=archive_chain)
        frames[kind] = df
        if len(df):
            judge(label, df, claim=False)
            judge(f"{kind} mid fills (no authority)",
                  run(acon, ev, kind, mid=True, get_chain=archive_chain), claim=False)
        else:
            print("  no tradable events")
    if a.csv:
        pd.concat([f.assign(kind=k) for k, f in frames.items()]).to_csv(a.csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--forward", action="store_true",
                    help=f"E2: the chain archive, judged >= {E2_JUDGE_FROM}")
    ap.add_argument("--coverage", action="store_true",
                    help="with --forward: event counts only, no outcomes")
    ap.add_argument("--early", action="store_true",
                    help="with --forward: run before the judge date (disclose it)")
    a = ap.parse_args()
    if a.forward:
        forward(a)
        return
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ev = events(con)
    print(f"Dolt earnings events with pre+post snapshots: {len(ev)}, "
          f"{ev.symbol.nunique()} symbols, {ev.D.min()}..{ev.D.max()}")
    res = {}
    frames = {}
    for kind, label in (("condor", "E1-S short 25/10 condor"),
                        ("fly", "E1-F short ATM iron fly"),
                        ("straddle", "E1-L long ATM straddle")):
        print(f"\n== {label}")
        df = run(con, ev, kind)
        frames[kind] = df
        res[kind] = judge(label, df)
    if a.csv:
        pd.concat([f.assign(kind=k) for k, f in frames.items()]).to_csv(a.csv, index=False)

    print("\n== Sensitivities (no authority)")
    for kind in ("condor", "fly", "straddle"):
        df = frames[kind]
        judge(f"{kind} mid fills", run(con, ev, kind, mid=True), claim=False)
        print(f"  {kind}: {df.interp.mean():.1%} of trades used an interpolated exit leg")
        judge(f"{kind} exact-quoted only", df[~df.interp], claim=False)
        judge(f"{kind} excl. 2020", df[df.year != 2020], claim=False)
        judge(f"{kind} before-open", df[~df.amc], claim=False)
        judge(f"{kind} after-close", df[df.amc], claim=False)
        y = df.groupby("year").r.agg(["size", "mean", "median"])
        print(f"  {kind} by year:\n" + y.to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"\nFAMILY: {sum(r['pass'] for r in res.values())}/3 claimants pass (98.33%)")


if __name__ == "__main__":
    main()
