"""EXPLORATORY (no authority): when did the late-day sign flip, and can an adaptive rule trade it?"""
import numpy as np, pandas as pd
from src.intraday.alpaca_bars import load_symbol, fetch_calendar
from src.intraday.sessions import build_tables, stack
from src.intraday.stats import summarise
from scripts import intraday_tests as it

long = stack(build_tables({s: load_symbol(s) for s in it.SYMBOLS}, it.TARGETS, calendar=fetch_calendar()))
t = it.rule_i1(long)
gross = it.daily(t, 3, cents_override=0.0)          # fade, gross
cost = it.daily(t.assign(gross=0.0), 3)             # -cost per day (negative)
cost = -cost
g = gross.copy(); g.index = pd.to_datetime(g.index)

print("half-year gross FADE mean (bps/day):")
hy = g.groupby([g.index.year, (g.index.month > 6) + 1]).mean() * 1e4
print("  " + "  ".join(f"{y}H{h}:{v:+.1f}" for (y, h), v in hy.items()))

r250 = g.rolling(250).mean() * 1e4
print("rolling-250d gross fade first crosses > 0 (after 2018):",
      r250[(r250.index > "2018-06-01") & (r250 > 0)].index.min().date())

# Adaptive: take fade if trailing-N gross fade mean > 0 else follow; decision uses only prior days.
c = cost.reindex(gross.index).values
for N in (60, 120, 250):
    sig = np.sign(pd.Series(gross.values).rolling(N).mean().shift(1)).values
    ok = ~np.isnan(sig) & (sig != 0)
    net = sig[ok] * gross.values[ok] - c[ok]
    dates = np.array(gross.index)[ok]
    s_all = summarise(net)
    pre = net[dates < pd.Timestamp("2024-01-01").date()]; post = net[dates >= pd.Timestamp("2024-01-01").date()]
    print(f"adaptive N={N}: all {s_all['mean']*1e4:+.2f} bps/day CI[{s_all['lo']*1e4:+.2f},{s_all['hi']*1e4:+.2f}]"
          f"  pre-2024 {pre.mean()*1e4:+.2f}  2024+ {post.mean()*1e4:+.2f}  | mean cost {np.mean(c[ok])*1e4:.2f} bps")

# How big is the last-30-min move vs an ATM 0DTE premium? (scale only)
last = (long["p1600"] / long["p1530"] - 1).abs()
print(f"mean |15:30->16:00| move: {last.mean()*1e4:.1f} bps; median {last.median()*1e4:.1f} bps")
