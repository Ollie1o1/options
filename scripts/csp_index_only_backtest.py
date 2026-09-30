"""Close out `csp_index_only` (src/strategies/seed.py) — the one hypothesis in
the strategy library that was never actually measured: SPY-only, $5-wide bull
put spreads, IV rank >= 50, DTE 25-60. Its cost_profile carries an explicit
`unmeasured=True` because no index fills existed to price it.

Runs the registered spec EXACTLY ONCE — no sweep — on real crossing-cost
marks via src.dolt_spread (sell the short at its bid, buy the wing at its
ask; no modeled slippage). Also runs the required unselected control (same
width/delta/DTE window, no IV-rank filter): per
src.strategies.evidence.beats_benchmark, csp_index_only must strictly beat
it or the result is decoration, not a finding.

IV rank is trailing min-max over the last IV_RANK_WINDOW weekly observations
of the SAME 0.25-delta put this setup would actually sell (not an ATM proxy),
using only observations strictly BEFORE the candidate date — a rank must
never see its own future.

CLI:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/csp_index_only_backtest.py
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from scipy import stats as _stats

import numpy as np

from src import dolt_options as _do
from src.alloc.report import MIN_DSR, MIN_N, MIN_TSTAT
from src.alloc.validate import deflated_sharpe, effective_n, sharpe
from src.dolt_spread import run_spread_backtest

SYMBOL = "SPY"
DB_PATH = "data/dolt_options.db"
START = "2022-01-01"
END = "2026-06-12"
LOOKBACK_START = "2020-06-01"   # extra history so IV rank has a trailing window
SHORT_DELTA = 0.25
WIDTH = 5.0
DTE_MIN, DTE_MAX = 25, 60
IV_RANK_MIN = 50.0
IV_RANK_WINDOW = 52        # trailing weekly observations (~1 year)
IV_RANK_MIN_HISTORY = 26  # refuse to rank on less than half a year of history


def _build_iv_history(symbol: str, dates: List[str], db_path: str) -> Dict[str, float]:
    """The 0.25-delta put's own IV on each date actually resolved by
    `get_chain_near` — keyed by the ACTUAL date it snapped to, since that is
    what `simulate_spread`'s entry_filter will hand back as `ctx["date"]`."""
    out: Dict[str, float] = {}
    for d in dates:
        try:
            actual, chain = _do.get_chain_near(symbol, d, db_path=db_path)
        except _do.DoltQueryError:
            continue
        legs = [c for c in chain if c.get("type") == "put"
                and c.get("delta") is not None and c.get("iv") is not None]
        if not legs:
            continue
        short = min(legs, key=lambda c: abs(abs(c["delta"]) - SHORT_DELTA))
        out[actual] = float(short["iv"])
    return out


def _iv_rank_series(iv_hist: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Trailing min-max IV rank (0-100) at each date, window strictly PRIOR."""
    ordered = sorted(iv_hist.keys())
    out: Dict[str, Optional[float]] = {}
    for i, d in enumerate(ordered):
        window = [iv_hist[x] for x in ordered[max(0, i - IV_RANK_WINDOW):i]]
        if len(window) < IV_RANK_MIN_HISTORY:
            out[d] = None
            continue
        lo, hi = min(window), max(window)
        if hi - lo < 1e-9:
            out[d] = 50.0
            continue
        out[d] = max(0.0, min(1.0, (iv_hist[d] - lo) / (hi - lo))) * 100.0
    return out


def _make_filter(iv_rank: Dict[str, Optional[float]], require_iv_rank: bool):
    def entry_filter(ctx: Dict[str, Any]) -> bool:
        dte = ctx.get("dte")
        if dte is None or not (DTE_MIN <= dte <= DTE_MAX):
            return False
        if not require_iv_rank:
            return True
        rank = iv_rank.get(ctx["date"])
        # A date with no verifiable rank (too little trailing history, or a
        # chain that never resolved) is refused, never admitted as if clear —
        # the same NULL-is-not-zero convention this repo uses everywhere.
        return rank is not None and rank >= IV_RANK_MIN
    return entry_filter


def _clustered_tstat(trades: List[Dict[str, Any]]) -> float:
    """t-statistic over entry-day means — same arithmetic as
    src.alloc.report.clustered_tstat, fed from this script's plain
    ret/entry_date dicts instead of trade objects with pnl/capital_at_risk.
    Trades sharing an entry day share that day's move, so scoring them as
    independent overstates significance.
    """
    by_day: Dict[str, List[float]] = {}
    for t in trades:
        by_day.setdefault(str(t["entry_date"]), []).append(float(t["ret"]))
    days = np.array([np.mean(v) for v in by_day.values()], dtype=float)
    if days.size < 3 or days.std(ddof=1) == 0:
        return 0.0
    return float(days.mean() / (days.std(ddof=1) / np.sqrt(days.size)))


def _verdict(stats: Dict[str, Any]) -> str:
    """promote/reject/insufficient, against this project's own established
    promotion bar (src.alloc.report.promotion_verdict) rather than an
    invented threshold. The BROAD-stratum check in that function is skipped
    here on purpose: it asks whether an edge lives only in famous names,
    which has no meaning for a single-symbol (SPY-only) study.
    """
    n = stats.get("n", 0)
    if n < MIN_N:
        return "insufficient"
    if (stats.get("dsr", 0.0) < MIN_DSR
            or abs(stats.get("tstat_clustered", 0.0)) < MIN_TSTAT
            or stats.get("tstat_clustered", 0.0) < 0):
        return "reject"
    return "promote"


def _measure(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n": 0}
    rets = [t["ret"] for t in trades]
    n_eff = effective_n([t["entry_date"] for t in trades],
                        [t["exit_date"] for t in trades])
    wins = sum(1 for r in rets if r > 0)
    return {
        "n": len(trades),
        "n_eff": n_eff,
        "win_rate": round(100.0 * wins / len(trades), 2),
        "mean_return_on_capital": round(sum(rets) / len(rets), 6),
        "sharpe": round(sharpe(rets), 4),
        "dsr": round(deflated_sharpe(rets, 1, n_eff), 4),
        "tstat_clustered": round(_clustered_tstat(trades), 3),
        "n_trials": 1,
        "skew": round(float(_stats.skew(rets)), 3) if len(rets) >= 3 else None,
        "window": [START, END],
    }


def main() -> int:
    all_dates = _do._date_range(LOOKBACK_START, END, weekly=True)
    entry_dates = _do._date_range(START, END, weekly=True)

    print(f"Building IV history for {SYMBOL} across {len(all_dates)} weeks "
         f"({LOOKBACK_START}..{END})...", file=sys.stderr)
    iv_hist = _build_iv_history(SYMBOL, all_dates, DB_PATH)
    iv_rank = _iv_rank_series(iv_hist)
    print(f"IV history resolved: {len(iv_hist)}/{len(all_dates)} weeks.",
         file=sys.stderr)

    print("Running csp_index_only (selected: DTE 25-60, IV rank >= 50)...",
         file=sys.stderr)
    selected = run_spread_backtest(
        [SYMBOL], entry_dates, short_delta=SHORT_DELTA, width=WIDTH,
        entry_filter=_make_filter(iv_rank, require_iv_rank=True),
        side="put", db_path=DB_PATH)

    print("Running the unselected control (DTE 25-60 only, no IV-rank filter)...",
         file=sys.stderr)
    control = run_spread_backtest(
        [SYMBOL], entry_dates, short_delta=SHORT_DELTA, width=WIDTH,
        entry_filter=_make_filter(iv_rank, require_iv_rank=False),
        side="put", db_path=DB_PATH)

    selected_stats = _measure(selected.get("trades", []))
    control_stats = _measure(control.get("trades", []))
    beats = bool(selected_stats.get("n") and control_stats.get("n")
                and selected_stats["sharpe"] > control_stats["sharpe"])

    print(json.dumps({
        "selected": selected_stats,
        "control": control_stats,
        "beats_unselected_control": beats,
        "verdict": _verdict(selected_stats),
        "selected_partial": selected.get("partial"),
        "control_partial": control.get("partial"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
