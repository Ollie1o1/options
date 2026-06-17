"""Plain-text report. Sharpe is annualized from per-trade mean/std using the
entry frequency (trades/year). Pure.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def annualized_sharpe(mean: float, std: float, trades_per_year: float) -> float:
    if std <= 0:
        return 0.0
    return (mean / std) * math.sqrt(trades_per_year)


def format_report(currency: str, stats: Dict[str, float], tstat: float,
                  ci: Tuple[float, float], cost_breakeven_mult: Optional[float],
                  mark_rmse: Optional[float], regime: Optional[Dict[str, float]],
                  trades_per_year: float = 52.0, oos: Optional[Dict[str, Dict]] = None,
                  header_extra: str = "") -> str:
    sharpe = annualized_sharpe(stats.get("mean", 0), stats.get("std", 0), trades_per_year)
    L = []
    L.append(f"=== Vol-carry backtest — {currency} {header_extra}===".rstrip())
    L.append(f"trades={stats['n']}  mean=${stats['mean']:.1f}  hit={stats['hit_rate']*100:.1f}%  "
             f"PF={stats['profit_factor']:.2f}  total=${stats['total']:.0f}")
    L.append(f"Sharpe(ann)={sharpe:.2f}  maxDD=${stats['max_drawdown']:.0f}  "
             f"CVaR5=${stats['cvar5']:.0f}  worst=${stats['worst']:.0f}")
    L.append(f"significance: t-stat(NW)={tstat:.2f}  bootstrap95%CI=(${ci[0]:.1f}, ${ci[1]:.1f})")
    if cost_breakeven_mult is not None:
        if cost_breakeven_mult == float("inf"):
            L.append("cost breakeven multiple: survives >5x quoted spread")
        else:
            L.append(f"cost breakeven multiple: {cost_breakeven_mult:.2f}x quoted spread")
    if mark_rmse is not None:
        L.append(f"mark check: DVOL-vs-real-prints RMSE={mark_rmse:.2f} vol pts")
    if regime:
        L.append("regime mean $: " + "  ".join(f"{k}={v:+.1f}" for k, v in regime.items()))
    if oos:
        for tag, s in oos.items():
            if s.get("n", 0):
                L.append(f"  [{tag}] n={s['n']} mean=${s['mean']:.1f} "
                         f"hit={s['hit_rate']*100:.1f}% total=${s['total']:.0f}")
    return "\n".join(L)
