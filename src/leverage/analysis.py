"""Risk + performance analytics over a BacktestResult. Pure: takes the result's
per-trade returns/sides/exit-reasons and produces decision-grade risk metrics —
profit factor, per-trade Sharpe/Sortino, max drawdown, win/loss asymmetry,
per-side skill, and the exit-reason mix (is it taking profit or getting stopped?).
"""
from __future__ import annotations
from typing import Dict
import numpy as np


def analyze(result) -> Dict:
    """Return a metrics dict for a BacktestResult. Empty result -> n=0."""
    pnls = np.asarray(result.trades, dtype=float)
    out: Dict = {"n": int(len(pnls))}
    if len(pnls) == 0:
        return out
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    std = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    downside = pnls[pnls < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    out.update({
        "win_rate": float((pnls > 0).mean()),
        "expectancy": float(pnls.mean()),
        "total_return": float(np.prod(1.0 + pnls) - 1.0),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "payoff": (float(wins.mean()) / abs(float(losses.mean())))
                  if len(wins) and len(losses) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "sharpe_per_trade": (float(pnls.mean()) / std) if std > 0 else 0.0,
        "sortino_per_trade": (float(pnls.mean()) / dstd) if dstd > 0 else 0.0,
        "max_dd": float(getattr(result, "max_dd", 0.0)),
    })
    sides = list(getattr(result, "sides", []))
    if len(sides) == len(pnls) and sides:
        out["by_side"] = {}
        for side in ("long", "short"):
            mask = np.array([s == side for s in sides])
            if mask.any():
                sp = pnls[mask]
                out["by_side"][side] = {
                    "n": int(mask.sum()),
                    "win_rate": float((sp > 0).mean()),
                    "expectancy": float(sp.mean()),
                }
    reasons = list(getattr(result, "exit_reasons", []))
    if reasons:
        out["exit_mix"] = {r: reasons.count(r) / len(reasons)
                           for r in sorted(set(reasons))}
    return out


def render_analysis(result, label: str = "") -> str:
    a = analyze(result)
    if a["n"] == 0:
        return f"{label}: no trades."
    lines = [f"{label}  (n={a['n']})" if label else f"n={a['n']}"]
    pf = a["profit_factor"]
    lines.append(
        f"  return {a['total_return']*100:+.1f}%   expectancy {a['expectancy']*100:+.4f}%/trade"
        f"   win {a['win_rate']*100:.1f}%   payoff {a['payoff']:.2f}")
    lines.append(
        f"  profit factor {pf:.2f}   Sharpe/trade {a['sharpe_per_trade']:.3f}"
        f"   Sortino/trade {a['sortino_per_trade']:.3f}   maxDD {a['max_dd']*100:.0f}%")
    lines.append(
        f"  avg win {a['avg_win']*100:+.3f}%   avg loss {a['avg_loss']*100:+.3f}%")
    if a.get("by_side"):
        for side, s in a["by_side"].items():
            lines.append(f"  {side:5}: n={s['n']:<5} win {s['win_rate']*100:.1f}%"
                         f"   expectancy {s['expectancy']*100:+.4f}%")
    if a.get("exit_mix"):
        mix = "  ".join(f"{k} {v*100:.0f}%" for k, v in a["exit_mix"].items())
        lines.append(f"  exits: {mix}")
    verdict = "EDGE" if a["expectancy"] > 0 and pf > 1.0 else "NO EDGE"
    lines.append(f"  -> {verdict}")
    return "\n".join(lines)
