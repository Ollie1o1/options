"""Sleeve-blend engine — does combining strategies beat the best single one? (direction A)

Every Dolt backtest emits a stream of dated trades. A *sleeve* wraps one such
stream with a name and an exposure tag (short_market | market_neutral). This
module measures whether blending sleeves — especially genuinely uncorrelated
ones — produces a portfolio with better risk-adjusted return than any sleeve
alone. That is the only escape from the capacity wall the portfolio layer
exposed (single short-premium sleeves are real per-trade but tiny + mutually
correlated, i.e. the same short-market bet many times).

Method (deliberately simple, YAGNI): trades from all sleeves are merged and
compounded equal-risk (risk_frac of equity each) in exit-date order via
`dolt_portfolio.equity_curve`. Co-movement is measured by bucketing each sleeve's
P&L into MONTHLY returns (sleeves trade on different days/frequencies, so a common
calendar is the honest way to correlate them) and taking pairwise Pearson. Sharpe
is annualized from monthly portfolio returns. The headline is the diversification
benefit: blended Sharpe vs the best standalone sleeve's Sharpe.

CLI:
    python -m src.dolt_blend --start 2022-01-01 --end 2024-12-31
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.dolt_portfolio import equity_curve


class Sleeve:
    """A named trade stream. ``trades`` are dicts with at least entry_date,
    exit_date, ret. ``exposure`` is a free-text tag for grouping."""

    def __init__(self, name: str, trades: List[Dict[str, Any]], exposure: str = ""):
        self.name = name
        self.trades = [t for t in trades if t.get("ret") is not None and t.get("exit_date")]
        self.exposure = exposure

    def __repr__(self):
        return f"Sleeve({self.name!r}, n={len(self.trades)}, {self.exposure})"


def monthly_returns(trades: List[Dict[str, Any]], risk_frac: float = 1.0) -> Dict[str, float]:
    """Sized portfolio P&L per calendar month (YYYY-MM): sum over trades EXITING
    that month of ret*risk_frac. Trade-weighted (a busy month contributes more),
    so it stays consistent with the fixed-fractional dollar equity curve — unlike
    a mean-per-month, which equal-weights months and disagrees with sized reality
    when trade counts per month vary wildly (e.g. ~18 tech trades/mo vs <1 index).
    Additive (not compounded) so many same-month trades can't explode. risk_frac
    is a linear scale; Sharpe and correlation are invariant to it."""
    by_month: Dict[str, float] = {}
    for t in trades:
        m = t["exit_date"][:7]
        by_month[m] = by_month.get(m, 0.0) + t["ret"] * risk_frac
    return by_month


def annualized_sharpe(monthly: Dict[str, float], rf_annual: float = 0.0) -> Optional[float]:
    """Annualized Sharpe from a monthly-return series. None if < 2 months or
    zero dispersion."""
    vals = list(monthly.values())
    if len(vals) < 2:
        return None
    rf_m = rf_annual / 12.0
    excess = [v - rf_m for v in vals]
    mean = sum(excess) / len(excess)
    var = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    return (mean / sd) * math.sqrt(12)


def correlation(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
    """Pearson correlation of two monthly-return series on their COMMON months.
    None if fewer than 3 overlapping months or zero variance in either."""
    months = sorted(set(a) & set(b))
    if len(months) < 3:
        return None
    xs = [a[m] for m in months]
    ys = [b[m] for m in months]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def equal_weight_monthly(sleeves: List[Sleeve]) -> Dict[str, float]:
    """Monthly return of an EQUAL-WEIGHT-per-sleeve, monthly-rebalanced portfolio:
    each month, the mean of the sleeve returns that traded that month. This is the
    fair diversification test — it does NOT let a high-trade-count sleeve dominate
    (which pooling all trades equal-risk would, e.g. 662 tech trades vs 31 index)."""
    per = {s.name: monthly_returns(s.trades) for s in sleeves}
    months = sorted(set().union(*[set(m) for m in per.values()])) if per else []
    out: Dict[str, float] = {}
    for m in months:
        vals = [per[s.name][m] for s in sleeves if m in per[s.name]]
        if vals:
            out[m] = sum(vals) / len(vals)
    return out


def correlation_matrix(sleeves: List[Sleeve]) -> Dict[str, Dict[str, Optional[float]]]:
    """Pairwise monthly-return correlation between sleeves."""
    mr = {s.name: monthly_returns(s.trades) for s in sleeves}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for s in sleeves:
        out[s.name] = {}
        for t in sleeves:
            out[s.name][t.name] = 1.0 if s.name == t.name else correlation(mr[s.name], mr[t.name])
    return out


def blend(sleeves: List[Sleeve], start_equity: float = 100_000.0,
          risk_frac: float = 0.01, rf_annual: float = 0.0) -> Dict[str, Any]:
    """Blend sleeves equal-risk and report the diversification benefit.

    Returns per-sleeve standalone stats, the blended portfolio stats, the
    correlation matrix, and the headline: blended Sharpe vs best standalone."""
    sleeves = [s for s in sleeves if s.trades]
    per_sleeve: Dict[str, Any] = {}
    for s in sleeves:
        eq = equity_curve(s.trades, start_equity=start_equity, risk_frac=risk_frac)
        mr = monthly_returns(s.trades)
        per_sleeve[s.name] = {
            "n": len(s.trades), "exposure": s.exposure,
            "total_return": eq["total_return"], "max_drawdown": eq["max_drawdown"],
            "cagr": eq["cagr"], "sharpe": annualized_sharpe(mr, rf_annual),
        }
    # Fair blend: equal weight per sleeve, monthly-rebalanced (the diversification test).
    blended_mr = equal_weight_monthly(sleeves)
    blended_sharpe = annualized_sharpe(blended_mr, rf_annual)
    # Secondary "pooled" view: dump all trades in equal-risk per TRADE (dollar
    # return/drawdown) — dominated by trade count, so reported but not the headline.
    merged: List[Dict[str, Any]] = [t for s in sleeves for t in s.trades]
    pooled_eq = equity_curve(merged, start_equity=start_equity, risk_frac=risk_frac)

    standalone_sharpes = [v["sharpe"] for v in per_sleeve.values() if v["sharpe"] is not None]
    best_standalone = max(standalone_sharpes) if standalone_sharpes else None
    benefit = (blended_sharpe - best_standalone) if (blended_sharpe is not None
                                                     and best_standalone is not None) else None
    return {
        "sleeves": per_sleeve,
        "correlation": correlation_matrix(sleeves),
        "blended": {                       # equal-weight-per-sleeve (the fair test)
            "sharpe": blended_sharpe,
            "n_months": len(blended_mr),
            "mean_monthly": (sum(blended_mr.values()) / len(blended_mr)) if blended_mr else None,
        },
        "pooled": {                        # all trades pooled equal-risk (dollar view)
            "n": pooled_eq["n"], "total_return": pooled_eq["total_return"],
            "max_drawdown": pooled_eq["max_drawdown"], "cagr": pooled_eq["cagr"],
            "max_concurrent": pooled_eq["max_concurrent"],
        },
        "best_standalone_sharpe": best_standalone,
        "diversification_benefit": benefit,
    }


# ── Real-sleeve builders ────────────────────────────────────────────────────
def build_real_sleeves(db_path=None, start="2022-01-01", end="2024-12-31",
                       tech=("AAPL", "MSFT", "GOOG", "META", "AMZN"),
                       earnings=("AAPL", "NVDA", "AMZN", "TSLA", "META")) -> List[Sleeve]:
    """The three sleeves this research produced: index put spread + tech short
    puts (both short_market) and the earnings strangle (market_neutral)."""
    from src import dolt_options as _do
    from src.dolt_spread import run_spread_backtest
    from src.dolt_short import run_short_backtest
    from src.dolt_earnings_sell import run_earnings_sell_backtest
    dates = _do._date_range(start, end, weekly=True)
    spread = run_spread_backtest(["SPY"], dates, db_path=db_path)
    short = run_short_backtest(list(tech), dates, db_path=db_path, opt_type="put")
    earn = run_earnings_sell_backtest(list(earnings), db_path=db_path, start=start, end=end)
    return [
        Sleeve("index_spread", spread.get("trades", []), "short_market"),
        Sleeve("tech_short_put", short.get("trades", []), "short_market"),
        Sleeve("earnings_strangle", earn.get("trades", []), "market_neutral"),
    ]


def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Sleeve-blend engine over real-marks sleeves")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--risk-frac", type=float, default=0.01)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    cfg = {}
    try:
        cfg = json.load(open("config.json")).get("dolt_options", {})
    except Exception:
        pass
    db = args.db or cfg.get("cache_path")
    sleeves = build_real_sleeves(db_path=db, start=args.start, end=args.end)
    res = blend(sleeves, risk_frac=args.risk_frac)

    def f(x, p=True):
        if x is None:
            return "  -  "
        return f"{x:+.1%}" if p else f"{x:+.2f}"
    print(f"Sleeve blend {args.start}..{args.end}  risk={args.risk_frac:.1%}/trade\n")
    print(f"  {'sleeve':18} {'exposure':14} {'n':>4} {'totRet':>8} {'maxDD':>7} {'Sharpe':>7}")
    for name, v in res["sleeves"].items():
        print(f"  {name:18} {v['exposure']:14} {v['n']:>4} {f(v['total_return']):>8} "
              f"{f(v['max_drawdown']):>7} {f(v['sharpe'], p=False):>7}")
    b, p = res["blended"], res["pooled"]
    print(f"\n  equal-weight-per-sleeve blend (the fair test): Sharpe {f(b['sharpe'], p=False)} "
          f"over {b['n_months']} months, mean monthly {f(b['mean_monthly'])}")
    print(f"  pooled-all-trades (dollar view, trade-count-dominated): total {f(p['total_return'])}, "
          f"maxDD {f(p['max_drawdown'])}, maxConcurrent {p['max_concurrent']}")
    print(f"\n  best standalone Sharpe: {f(res['best_standalone_sharpe'], p=False)}   "
          f"diversification benefit (blend - best): {f(res['diversification_benefit'], p=False)}")
    print("\n  correlation (monthly returns):")
    names = list(res["correlation"])
    print("    " + " " * 18 + "".join(f"{n[:10]:>12}" for n in names))
    for a in names:
        row = "".join(f"{f(res['correlation'][a][b2], p=False):>12}" for b2 in names)
        print(f"    {a:18}{row}")


if __name__ == "__main__":
    _cli()
