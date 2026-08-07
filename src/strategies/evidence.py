"""Land a backtest result on the setup that produced it.

Everything arrives through amend(), never a direct write, so a setup that was
validated and later killed keeps both facts and both reasons.

n_trials travels with the evidence deliberately: a deflated Sharpe is only
interpretable next to the size of the search that produced it, and separating
them is how an inflated number later gets quoted as though it were clean.

A `cost_profile` measured during the replay lands too, and takes precedence over
the ledger-wide friction table on the board: friction measured on the setup's own
fills beats friction measured on the structure in general.
"""
from __future__ import annotations

from typing import Any, Dict

from .record import StrategyRecord

_VERDICT_STATUS = {"promote": "validated",
                   "liquid_only": "backtesting",
                   "reject": "dead"}


def apply_result(record: StrategyRecord, result: Dict[str, Any],
                 date: str) -> StrategyRecord:
    evidence = {
        "dsr": result.get("dsr"), "pbo": result.get("pbo"),
        "tstat": result.get("tstat"), "sharpe": result.get("sharpe"),
        "by_stratum": result.get("by_stratum", {}),
        "capacity": result.get("capacity", {}),
        "n_trials": result.get("n_trials"), "window": result.get("window"),
        "evaluated": date,
    }
    verdict = result.get("verdict", "reject")
    out = record.amend("evidence", evidence, reason=f"backtest {date}", date=date)
    out = out.amend("verdict", verdict, reason=f"backtest {date}", date=date)
    cost_profile = result.get("cost_profile")
    if cost_profile:
        out = out.amend("cost_profile", cost_profile,
                        reason=f"friction measured in backtest {date}", date=date)
    return out.amend("status", _VERDICT_STATUS.get(verdict, "backtesting"),
                     reason=f"verdict={verdict}", date=date)


def beats_benchmark(result: Dict[str, Any],
                    benchmark: Dict[str, Any]) -> bool:
    """Strictly better than selling on no condition at all.

    Ties fail. Selectivity degraded this book monotonically, so a signal that
    merely matches the unselected benchmark is added complexity for nothing.
    """
    return float(result.get("sharpe", 0.0)) > float(benchmark.get("sharpe", 0.0))
