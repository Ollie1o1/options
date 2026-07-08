"""Net-of-cost walk-forward harness for the leverage candidates.

Runs each candidate twice — once at nominal per-asset cost, once at `stress`x
cost — and emits a verdict. A candidate is PROMOTE only if, out-of-sample, it is
net-profitable (mean ret_net > 0), profit factor >= 1.2, still net-positive
under the cost-stress, and has >= min_n trades (else UNDERPOWERED). Everything
else is DEAD. Strict on purpose: the project's history is that loose bars
promote noise.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateReport:
    name: str
    n_oos: int
    win_rate: float
    mean_ret_net: float
    expectancy_R: float
    profit_factor: float
    max_drawdown: float
    stress_mean_ret_net: float
    verdict: str


def _stats(rets: List[float], rs: List[float]) -> dict:
    if not rets:
        return {"n": 0, "win_rate": 0.0, "mean_ret": 0.0, "expectancy_R": 0.0,
                "pf": 0.0, "mdd": 0.0}
    a = np.array(rets)
    gains = a[a > 0].sum()
    losses = -a[a < 0].sum()
    eq = np.cumprod(1 + a)
    peak = np.maximum.accumulate(eq)
    mdd = float((1 - eq / peak).max()) if len(eq) else 0.0
    return {
        "n": len(a),
        "win_rate": float((a > 0).mean()),
        "mean_ret": float(a.mean()),
        "expectancy_R": float(np.mean(rs)) if rs else 0.0,
        "pf": float(gains / losses) if losses > 0 else float("inf"),
        "mdd": mdd,
    }


def evaluate(candidate, frames: Dict[str, pd.DataFrame],
             funding: Dict[str, pd.Series], costs: Dict[str, float],
             train_frac: float = 0.6, stress: float = 1.5,
             min_n: int = 20) -> CandidateReport:
    _, oos = candidate.walk_forward(frames, funding, costs, train_frac)
    stressed_costs = {k: v * stress for k, v in costs.items()}
    _, oos_s = candidate.walk_forward(frames, funding, stressed_costs, train_frac)

    s = _stats([t.ret_net for t in oos], [t.r_multiple for t in oos])
    ss = _stats([t.ret_net for t in oos_s], [t.r_multiple for t in oos_s])

    if s["n"] < min_n:
        verdict = "UNDERPOWERED"
    elif (s["mean_ret"] > 0 and s["pf"] >= 1.2 and ss["mean_ret"] > 0):
        verdict = "PROMOTE"
    else:
        verdict = "DEAD"

    return CandidateReport(
        name=candidate.name, n_oos=s["n"], win_rate=s["win_rate"],
        mean_ret_net=s["mean_ret"], expectancy_R=s["expectancy_R"],
        profit_factor=s["pf"], max_drawdown=s["mdd"],
        stress_mean_ret_net=ss["mean_ret"], verdict=verdict)


def render_report(reports: List[CandidateReport]) -> str:
    lines = ["  Leverage candidate validation (out-of-sample, net of cost)",
             "  " + "-" * 78,
             f"  {'candidate':<20}{'n':>4}{'win%':>7}{'meanRet':>10}"
             f"{'PF':>7}{'maxDD':>8}{'stress':>9}  verdict",
             "  " + "-" * 78]
    for r in reports:
        pf = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
        lines.append(
            f"  {r.name:<20}{r.n_oos:>4}{r.win_rate*100:>6.0f}%"
            f"{r.mean_ret_net*100:>9.2f}%{pf:>7}{r.max_drawdown*100:>7.1f}%"
            f"{r.stress_mean_ret_net*100:>8.2f}%  {r.verdict}")
    lines.append("  " + "-" * 78)
    lines.append("  PROMOTE = OOS net-positive, PF>=1.2, survives 1.5x cost, n>=20")
    return "\n".join(lines)
