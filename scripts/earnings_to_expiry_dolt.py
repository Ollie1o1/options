"""E3: earnings premium held to the front expiry (one crossing), Dolt 2020-2026.

Registration: docs/PREREG_EARNINGS_TO_EXPIRY_20261109.md (frozen before running).
Reuses E1's events, legs and entry fills unchanged.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -u -m scripts.earnings_to_expiry_dolt
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List

import pandas as pd

from scripts import earnings_premium_dolt as E


def closes(con) -> Dict[str, pd.Series]:
    df = pd.read_sql("SELECT symbol, date, close FROM stocks_close", con)
    return {s: g.set_index("date")["close"].sort_index() for s, g in df.groupby("symbol")}


def last_close(px: Dict[str, pd.Series], sym: str, day: str):
    s = px.get(sym)
    if s is None:
        return None
    s = s[s.index <= day]
    return float(s.iloc[-1]) if len(s) else None


def trade(pre: pd.DataFrame, post_day: str, kind: str, mid: bool, px, sym, pre_day) -> Dict:
    exps = sorted(e for e in pre.expiration.unique() if e > post_day)
    if not exps:
        return {"skip": "no_expiry"}
    exp = exps[0]
    c0 = pre[pre.expiration == exp]
    legs = E.legs_for(c0, kind)
    if legs is None:
        return {"skip": "no_legs"}
    q0 = c0.set_index(["strike", "type"])
    # spot guard: parity spot at the ATM strike vs the stored raw close
    atm = E._near(c0[c0.type == "call"], 0.50)
    s_close = last_close(px, sym, pre_day)
    if atm is None or s_close is None or (atm.strike, "put") not in q0.index:
        return {"skip": "no_spot"}
    s_par = atm.strike + atm.mid - q0.loc[(atm.strike, "put")].mid
    if abs(s_par / s_close - 1) > 0.02:
        return {"skip": "spot_guard"}
    st = last_close(px, sym, exp)
    if st is None or exp > px[sym].index[-1]:
        return {"skip": "no_expiry_close"}
    cash0 = settle = 0.0
    for k, t, sgn in legs.values():
        r0 = q0.loc[(k, t)]
        px0 = r0.mid if mid else (r0.ask if sgn > 0 else r0.bid)
        cash0 -= sgn * px0
        settle += sgn * (max(st - k, 0.0) if t == "call" else max(k - st, 0.0))
    pnl = 100.0 * (cash0 + settle)
    if kind == "straddle":
        car = -100.0 * cash0
    else:
        w = max(legs["lc"][0] - legs["sc"][0], legs["sp"][0] - legs["lp"][0])
        car = 100.0 * (w - cash0)
        if cash0 <= 0:
            return {"skip": "bad_price"}
    if car <= 0:
        return {"skip": "bad_price"}
    return {"exp": exp, "pnl": pnl, "car": car, "r": pnl / car, "move": st / s_close - 1}


def run(con, ev, px, kind, mid=False) -> pd.DataFrame:
    out: List[Dict] = []
    skips: Dict[str, int] = {}
    for e in ev.itertuples():
        t = trade(E.chain(con, e.symbol, e.pre), e.post, kind, mid, px, e.symbol, e.pre)
        if "skip" in t:
            skips[t["skip"]] = skips.get(t["skip"], 0) + 1
            continue
        wk = E._d(e.D).isocalendar()
        out.append({**e._asdict(), **t, "week": f"{wk[0]}-W{wk[1]:02d}",
                    "year": int(e.D[:4]),
                    "days_after": (E._d(t["exp"]) - E._d(e.D)).days})
    print(f"  [{kind}{' mid' if mid else ''}] traded {len(out)}/{len(ev)}; skipped {skips}")
    return pd.DataFrame(out)


def main() -> None:
    con = sqlite3.connect(f"file:{E.DB}?mode=ro", uri=True)
    ev = E.events(con)
    px = closes(con)
    print(f"events {len(ev)}, {ev.symbol.nunique()} symbols; closes for {len(px)} symbols")
    res, frames = {}, {}
    for kind, label in (("condor", "E3-S short 25/10 condor to expiry"),
                        ("fly", "E3-F short ATM iron fly to expiry"),
                        ("straddle", "E3-L long ATM straddle to expiry")):
        print(f"\n== {label}")
        df = run(con, ev, px, kind)
        frames[kind] = df
        res[kind] = E.judge(label, df)
    print("\n== Sensitivities (no authority)")
    for kind, df in frames.items():
        E.judge(f"{kind} mid entry", run(con, ev, px, kind, mid=True), claim=False)
        E.judge(f"{kind} expiry <= 10d after event", df[df.days_after <= 10], claim=False)
        E.judge(f"{kind} expiry > 10d after event", df[df.days_after > 10], claim=False)
        E.judge(f"{kind} excl. 2020", df[df.year != 2020], claim=False)
        y = df.groupby("year").r.agg(["size", "mean", "median"])
        print(f"  {kind} by year:\n" + y.to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"\nFAMILY: {sum(r['pass'] for r in res.values())}/3 claimants pass (98.33%)")


if __name__ == "__main__":
    main()
