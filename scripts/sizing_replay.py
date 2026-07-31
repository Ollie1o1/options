#!/usr/bin/env python3
"""Replay the closed Long Call cohort through the sizing engine real money would use.

Every row in the ledger has `quantity = 1.0`. That is the column default from a
schema migration; nothing ever writes it. So every statistic the Phase-1 gate
reads describes a **one-contract-per-signal** book — while the thing a READY
verdict would authorise is a book sized by `src/execution/sizing.py` (half-Kelly,
clamped by a risk cap and a cost cap).

Those are not the same book, and not only in scale. `size_position` returns 0
contracts whenever its caps round the position down — so the sizing rule is also
a **filter**, silently dropping trades the unsized cohort counts. This script
asks the question that follows: does the gate's answer survive it?

What it reports, at a given account value:

- how many cohort trades size to zero, i.e. would never have been opened;
- net P&L of the sized book against the unsized (one-contract) book;
- return on capital actually deployed;
- the scorer IC — Pearson and Spearman — recomputed over only the trades that
  survive sizing, beside the IC over the full cohort.

The IC comparison is the point. Per-trade returns do not change when a position
is scaled, so sizing can only move the IC by changing *which* trades are in it.
If the surviving subset's IC differs materially from the full cohort's, then the
gate is measuring a population the execution layer would not have traded.

Read-only: opens the ledger through a `mode=ro` URI. Usage:

    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/sizing_replay.py
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/sizing_replay.py --account 10000
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from src.execution.sizing import size_position

# The stop the long-option exit rule enforces, as a fraction of premium paid.
# Read from config so the replay tracks the rule rather than a copy of it.
_DEFAULT_LONG_STOP = -0.50


def load_long_stop(config_path: str = "config.json") -> float:
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        rules = (cfg.get("exit_rules") or {}).get("long_option") or {}
        return float(rules.get("stop_loss", _DEFAULT_LONG_STOP))
    except Exception:
        return _DEFAULT_LONG_STOP


def fetch_cohort(db_path: str, phase1_start: str) -> List[Dict[str, Any]]:
    """The gate's own cohort definition, plus the columns sizing needs."""
    sql = (
        "SELECT date, ticker, quality_score, pnl_pct, pnl_usd, entry_price, "
        "       capital_at_risk, quantity "
        "FROM trades "
        "WHERE strategy_name = 'Long Call' AND status = 'CLOSED' "
        "  AND COALESCE(paper_only, 0) = 0 "
        "  AND date >= ? "
        "  AND quality_score IS NOT NULL AND pnl_pct IS NOT NULL "
        "  AND entry_price IS NOT NULL AND entry_price > 0"
    )
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, (phase1_start,))]


def _ics(scores: Sequence[float], returns: Sequence[float]) -> Dict[str, Optional[float]]:
    """Pearson and Spearman, or Nones when the sample cannot support them."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    s, r = np.asarray(scores, dtype=float), np.asarray(returns, dtype=float)
    if len(s) < 3 or s.std() < 1e-8 or r.std() < 1e-8:
        return {"n": len(s), "pearson": None, "p_pearson": None,
                "spearman": None, "p_spearman": None}
    pr, pp = pearsonr(s, r)
    sr, sp = spearmanr(s, r)
    return {"n": len(s), "pearson": float(pr), "p_pearson": float(pp),
            "spearman": float(sr), "p_spearman": float(sp)}


def replay(rows: Sequence[Dict[str, Any]], account_value: float,
           stop_fraction: float = _DEFAULT_LONG_STOP,
           max_risk_pct: float = 0.02,
           max_position_pct: float = 0.10) -> Dict[str, Any]:
    """Size every cohort trade and compare the sized book to the unsized one."""
    sized: List[Dict[str, Any]] = []
    zeroed: List[Dict[str, Any]] = []

    for row in rows:
        entry = float(row["entry_price"])
        # stop_fraction is negative (-0.50 = "exit at half the premium paid").
        stop_price = entry * (1.0 + stop_fraction)
        s = size_position(account_value=account_value, entry_price=entry,
                          stop_price=stop_price, max_risk_pct=max_risk_pct,
                          max_position_pct=max_position_pct)
        rec = dict(row)
        rec["contracts"] = s.contracts
        rec["cost_basis"] = s.cost_basis
        rec["sizing_note"] = s.notes
        # pnl_usd in the ledger is the result of the single logged contract.
        rec["sized_pnl"] = (float(row["pnl_usd"]) * s.contracts
                            if row.get("pnl_usd") is not None else None)
        (zeroed if s.contracts == 0 else sized).append(rec)

    def _sum(items, key):
        return sum(float(i[key]) for i in items if i.get(key) is not None)

    unsized_pnl = _sum(rows, "pnl_usd")
    sized_pnl = _sum(sized, "sized_pnl")
    deployed = _sum(sized, "cost_basis")

    return {
        "account_value": account_value,
        "stop_fraction": stop_fraction,
        "n_cohort": len(rows),
        "n_sized": len(sized),
        "n_zeroed": len(zeroed),
        "unsized_net_pnl": unsized_pnl,
        "sized_net_pnl": sized_pnl,
        "capital_deployed": deployed,
        "return_on_deployed": (sized_pnl / deployed) if deployed > 0 else None,
        "contracts_total": sum(i["contracts"] for i in sized),
        "ic_full_cohort": _ics([r["quality_score"] for r in rows],
                               [r["pnl_pct"] for r in rows]),
        "ic_survivors": _ics([r["quality_score"] for r in sized],
                             [r["pnl_pct"] for r in sized]),
        "cost_capped": sum(1 for i in sized if i["sizing_note"] == "cost-capped"),
        "risk_capped": sum(1 for i in sized if i["sizing_note"] == "risk-capped"),
    }


def _fmt_ic(d: Dict[str, Optional[float]]) -> str:
    if d["pearson"] is None:
        return f"n={d['n']}  (too few / degenerate to correlate)"
    return (f"n={d['n']}  Pearson {d['pearson']:+.3f} (p={d['p_pearson']:.3f})  "
            f"Spearman {d['spearman']:+.3f} (p={d['p_spearman']:.3f})")


def format_report(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Sizing replay — account ${r['account_value']:,.0f}, "
                 f"stop {r['stop_fraction'] * 100:+.0f}% of premium")
    lines.append("")
    lines.append(f"Cohort trades              : {r['n_cohort']}")
    lines.append(f"  sized to >= 1 contract   : {r['n_sized']} "
                 f"({r['risk_capped']} risk-capped, {r['cost_capped']} cost-capped)")
    lines.append(f"  sized to ZERO contracts  : {r['n_zeroed']}  "
                 f"— the sizing rule would never have opened these")
    lines.append("")
    lines.append(f"Unsized book (1 contract each) : {r['unsized_net_pnl']:+,.2f}")
    lines.append(f"Sized book                     : {r['sized_net_pnl']:+,.2f} "
                 f"across {r['contracts_total']:,} contracts")
    if r["return_on_deployed"] is not None:
        lines.append(f"Capital deployed               : ${r['capital_deployed']:,.2f} "
                     f"→ {r['return_on_deployed'] * 100:+.1f}% on deployed")
        if r["capital_deployed"] > r["account_value"]:
            # Sizing each trade independently against the full account is what
            # the engine does; summing those costs across a window is NOT a
            # claim that the account ever held them at once.
            lines.append("    (cumulative across the whole window, not peak "
                         "concurrent exposure — each trade was sized against "
                         "the full account, so overlapping positions are not "
                         "netted here)")
    lines.append("")
    lines.append("Scorer IC — the question that matters:")
    lines.append(f"  full cohort (what the gate reads) : {_fmt_ic(r['ic_full_cohort'])}")
    lines.append(f"  survivors of sizing               : {_fmt_ic(r['ic_survivors'])}")
    lines.append("")
    if r["n_zeroed"] == 0:
        lines.append("Sizing excluded nothing at this account value, so the gate's "
                     "cohort and the tradeable cohort are the same population.")
    else:
        lines.append(f"Sizing drops {r['n_zeroed']} of {r['n_cohort']} trades. The gate "
                     "reads the full cohort, so it is scoring signals the execution "
                     "layer would have declined to take.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay the closed LC cohort through the sizing engine")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--account", type=float, default=None,
                    help="account value; omit to sweep a range")
    ap.add_argument("--max-risk-pct", type=float, default=0.02)
    ap.add_argument("--max-position-pct", type=float, default=0.10)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    phase1_start = (cfg.get("auto_log") or {}).get("phase1_start_date")
    if not phase1_start:
        raise SystemExit("config.json missing auto_log.phase1_start_date")

    rows = fetch_cohort(args.db, phase1_start)
    if not rows:
        raise SystemExit("no closed Long Call cohort trades found")

    stop = load_long_stop(args.config)
    accounts = [args.account] if args.account else [5_000, 10_000, 25_000, 50_000]
    for i, acct in enumerate(accounts):
        if i:
            print("\n" + "-" * 72 + "\n")
        print(format_report(replay(rows, acct, stop,
                                   args.max_risk_pct, args.max_position_pct)))


if __name__ == "__main__":
    main()
