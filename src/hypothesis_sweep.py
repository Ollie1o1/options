"""Preregistered hypothesis sweep over strategy/weights/entry-exit/gates.

The six hypotheses are locked in
docs/superpowers/specs/2026-09-19-hypothesis-sweep-design.md and copied
verbatim to docs/HYPOTHESIS_SWEEP_PREREG.md. This module only computes — it
must never add, remove, or retune a hypothesis after seeing a result.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.alloc.validate import deflated_sharpe, effective_n
from src.backtest_optimizer import DEFAULT_UNIVERSE, ENTRY_DTE as _CURRENT_ENTRY_DTE
from src.backtest_spreads import (
    DEFAULT_SURFACE_PATH, STOP_LOSS_MULT as _CURRENT_STOP_MULT, SPREAD_IDX,
    SpreadTrade, load_default_surface, run_vertical_backtest,
)
from src.prereg_ranker import cluster_bootstrap_ci, rank_ic

# The size of the preregistered family. Every deflated_sharpe call in this
# module passes this literal value — never a computed count, never adjusted
# because a hypothesis refuses. See NTrialsIsLockedTest.
N_TRIALS = 6

# Repo convention (backtest_optimizer._run_strategy's own floor).
MIN_RAW_TRADES = 30

# deflated_sharpe itself treats n_eff < 3 as unmeasurable and returns 0.0 —
# checked explicitly here so that outcome is distinguishable from a genuine
# "search alone explains this" DSR of 0.0.
MIN_EFFECTIVE_N = 3

DSR_SURVIVAL_BAR = 0.95

# H5 (stop-loss multiple) comparison: variant tested at 1.5x, current at 2.0x
H5_VARIANT_STOP_MULT = 1.5

# H4 (entry DTE) comparison: variant tested at 30 DTE, current at 45 DTE
H4_VARIANT_ENTRY_DTE = 30

# H6 (credit-to-width floor) comparison: current floor at 0.20, variant at 0.25
H6_CURRENT_FLOOR = 0.20
H6_VARIANT_FLOOR = 0.25

# H3 (spread-score IC) Bonferroni-adjusted alpha across the family of 6 hypotheses
H3_BONFERRONI_ALPHA = 0.05 / N_TRIALS


@dataclass
class HypothesisResult:
    id: str
    category: str
    statistic_type: str
    value: Optional[float]
    n_eff: Optional[int]
    n_raw_trades: int
    survives: bool
    refused: bool
    reason: Optional[str] = None
    notes: str = ""


def _dsr_from_trades(
    trades: List[SpreadTrade],
) -> Tuple[Optional[float], Optional[int], bool, Optional[str]]:
    """DSR for a pooled trade population, or a refusal.

    Directly on the full return series with no train/test folds — matching
    src/backtester.py's own deflated_sharpe usage. See the design spec's
    "No train/test folds or purging" section for why: nothing here is fit
    in-sample, so there is no leakage channel for a fold/purge step to
    guard against.
    """
    n_raw = len(trades)
    if n_raw < MIN_RAW_TRADES:
        return None, None, True, "insufficient_raw_trades"

    starts = [t.entry_date for t in trades]
    ends = [t.exit_date for t in trades]
    n_eff = effective_n(starts, ends)
    if n_eff < MIN_EFFECTIVE_N:
        return None, n_eff, True, "insufficient_effective_n"

    pnl = np.array([t.pnl_pct for t in trades], dtype=float)
    dsr = deflated_sharpe(pnl, N_TRIALS, n_eff)
    return dsr, n_eff, False, None


def run_h1_bull_put(tickers: Optional[List[str]] = None) -> HypothesisResult:
    """H1: Bull Put spread DSR across the full ticker universe."""
    universe = tickers if tickers is not None else DEFAULT_UNIVERSE
    surface, _ = load_default_surface()
    trades = run_vertical_backtest(universe, option_type="put", surface=surface)
    dsr, n_eff, refused, reason = _dsr_from_trades(trades)
    survives = (not refused) and dsr is not None and dsr >= DSR_SURVIVAL_BAR
    return HypothesisResult("H1", "strategy", "dsr", dsr, n_eff, len(trades),
                            survives, refused, reason)


def run_h2_bear_call(tickers: Optional[List[str]] = None) -> HypothesisResult:
    """H2: Bear Call spread DSR across the full ticker universe."""
    universe = tickers if tickers is not None else DEFAULT_UNIVERSE
    surface, _ = load_default_surface()
    trades = run_vertical_backtest(universe, option_type="call", surface=surface)
    dsr, n_eff, refused, reason = _dsr_from_trades(trades)
    survives = (not refused) and dsr is not None and dsr >= DSR_SURVIVAL_BAR
    return HypothesisResult("H2", "strategy", "dsr", dsr, n_eff, len(trades),
                            survives, refused, reason)


def run_h5_stop_loss(tickers: Optional[List[str]] = None) -> HypothesisResult:
    """H5: Stop-loss multiple (2.0x current vs 1.5x variant) matched-pair DSR.

    Runs the same roll-forward calendar twice at different stop-loss multiples,
    matches trades by (symbol, entry_date) key, computes per-trade P&L
    difference (variant - current), and runs DSR on the difference series.
    """
    universe = tickers if tickers is not None else DEFAULT_UNIVERSE
    surface, _ = load_default_surface()
    current = run_vertical_backtest(universe, option_type="put",
                                    stop_mult=_CURRENT_STOP_MULT, surface=surface)
    variant = run_vertical_backtest(universe, option_type="put",
                                    stop_mult=H5_VARIANT_STOP_MULT, surface=surface)
    current_by_key = {(t.symbol, t.entry_date): (t.pnl_pct, t.exit_date)
                      for t in current}

    diffs: List[float] = []
    entries = []
    exits = []
    for t in variant:
        key = (t.symbol, t.entry_date)
        if key in current_by_key:
            cur_pnl, cur_exit = current_by_key[key]
            diffs.append(t.pnl_pct - cur_pnl)
            entries.append(t.entry_date)
            # A paired difference's true interval spans until whichever
            # side exits later — using only the variant's exit date would
            # systematically understate the interval (the tighter 1.5x
            # stop tends to exit earlier than 2.0x), inflating n_eff and,
            # through it, deflated_sharpe's z-score.
            exits.append(max(t.exit_date, cur_exit))

    n_raw = len(diffs)
    if n_raw < MIN_RAW_TRADES:
        return HypothesisResult("H5", "entry_exit", "dsr", None, None, n_raw,
                                False, True, "insufficient_raw_trades")

    n_eff = effective_n(entries, exits)
    if n_eff < MIN_EFFECTIVE_N:
        return HypothesisResult("H5", "entry_exit", "dsr", None, n_eff, n_raw,
                                False, True, "insufficient_effective_n")

    dsr = deflated_sharpe(np.array(diffs, dtype=float), N_TRIALS, n_eff)
    survives = dsr >= DSR_SURVIVAL_BAR
    notes = f"mean_diff={float(np.mean(diffs)):.4f} n_pairs={n_raw}"
    return HypothesisResult("H5", "entry_exit", "dsr", dsr, n_eff, n_raw,
                            survives, False, None, notes)


def run_h4_entry_dte(tickers: Optional[List[str]] = None) -> HypothesisResult:
    """H4: Entry DTE (45 current vs 30 variant) independent simulation DSR compare.

    Runs two independent simulations at different entry_dte values — these visit
    different roll-forward points since the DTE itself changes the calendar, so
    trades do NOT correspond 1:1. Computes DSR independently for each, then
    compares: survives only if variant's DSR clears the bar AND exceeds current's.
    """
    universe = tickers if tickers is not None else DEFAULT_UNIVERSE
    surface, _ = load_default_surface()
    current = run_vertical_backtest(universe, option_type="put",
                                    entry_dte=_CURRENT_ENTRY_DTE, surface=surface)
    variant = run_vertical_backtest(universe, option_type="put",
                                    entry_dte=H4_VARIANT_ENTRY_DTE, surface=surface)
    dsr_cur, n_eff_cur, refused_cur, reason_cur = _dsr_from_trades(current)
    dsr_var, n_eff_var, refused_var, reason_var = _dsr_from_trades(variant)

    if refused_cur or refused_var:
        reason = reason_var if refused_var else reason_cur
        n_eff = n_eff_var if refused_var else n_eff_cur
        return HypothesisResult("H4", "entry_exit", "dsr_compare", None,
                                n_eff, len(variant), False, True, reason)

    # Neither side refused, so _dsr_from_trades guarantees both are floats —
    # narrows Optional[float] for mypy past this point.
    assert dsr_cur is not None and dsr_var is not None
    survives = dsr_var >= DSR_SURVIVAL_BAR and dsr_var > dsr_cur
    notes = f"dsr_current(dte={_CURRENT_ENTRY_DTE})={dsr_cur:.4f}"
    return HypothesisResult("H4", "entry_exit", "dsr_compare", dsr_var - dsr_cur,
                            n_eff_var, len(variant), survives, False, None, notes)


def run_h6_credit_to_width(tickers: Optional[List[str]] = None) -> HypothesisResult:
    """H6: Credit-to-width floor (0.20 current vs 0.25 variant) subset DSR compare.

    Runs one simulation and splits it into two NESTED subsets by filtering
    credit_to_width thresholds. Computes DSR independently for each subset,
    then compares: survives only if variant's DSR clears the bar AND exceeds
    current's.
    """
    universe = tickers if tickers is not None else DEFAULT_UNIVERSE
    surface, _ = load_default_surface()
    trades = run_vertical_backtest(universe, option_type="put", surface=surface)
    floor_020 = [t for t in trades if t.credit_to_width >= H6_CURRENT_FLOOR]
    floor_025 = [t for t in trades if t.credit_to_width >= H6_VARIANT_FLOOR]
    dsr_020, n_eff_020, refused_020, reason_020 = _dsr_from_trades(floor_020)
    dsr_025, n_eff_025, refused_025, reason_025 = _dsr_from_trades(floor_025)

    if refused_020 or refused_025:
        reason = reason_025 if refused_025 else reason_020
        n_eff = n_eff_025 if refused_025 else n_eff_020
        return HypothesisResult("H6", "gates", "dsr_compare", None, n_eff,
                                len(floor_025), False, True, reason)

    # Neither side refused, so _dsr_from_trades guarantees both are floats —
    # narrows Optional[float] for mypy past this point.
    assert dsr_020 is not None and dsr_025 is not None
    survives = dsr_025 >= DSR_SURVIVAL_BAR and dsr_025 > dsr_020
    notes = (f"n_survivors_floor_{H6_CURRENT_FLOOR}={len(floor_020)} "
            f"n_survivors_floor_{H6_VARIANT_FLOOR}={len(floor_025)} "
            f"dsr_floor_{H6_CURRENT_FLOOR}={dsr_020:.4f}")
    return HypothesisResult("H6", "gates", "dsr_compare", dsr_025 - dsr_020,
                            n_eff_025, len(floor_025), survives, False, None, notes)


def run_h3_spread_weight(tickers: Optional[List[str]] = None) -> HypothesisResult:
    """H3: Spread-score IC (within-quarter rank correlation) test.

    Does the spread-score component order outcomes within each quarter?
    Uses demeaned ranks to eliminate between-quarter differences (strategy
    context). Tests via cluster bootstrap CI on whole symbols, comparing to
    Bonferroni-adjusted alpha across the family of 6.
    """
    universe = tickers if tickers is not None else DEFAULT_UNIVERSE
    surface, _ = load_default_surface()
    trades = run_vertical_backtest(universe, option_type="put", surface=surface)

    n_raw = len(trades)
    if n_raw < MIN_RAW_TRADES:
        return HypothesisResult("H3", "weights", "ic", None, None, n_raw,
                                False, True, "insufficient_raw_trades")

    rows = []
    for t in trades:
        entry_ts = pd.Timestamp(t.entry_date)
        quarter = f"{entry_ts.year}Q{(entry_ts.month - 1) // 3 + 1}"
        rows.append({
            "symbol": t.symbol,
            "entry_quarter": quarter,
            "spread_score": float(t.components[SPREAD_IDX]),
            "pnl_pct": t.pnl_pct,
        })
    df = pd.DataFrame(rows)
    n_clusters = int(df["symbol"].nunique())

    ic = rank_ic(df, "spread_score", "pnl_pct", ["entry_quarter"])
    lo, hi = cluster_bootstrap_ci(df, "spread_score", "pnl_pct",
                                  ["entry_quarter"], "symbol",
                                  alpha=H3_BONFERRONI_ALPHA)
    if ic is None or (lo is None and hi is None):
        # rank_ic's own signal for "the feature or outcome has zero
        # variance" (src/prereg_ranker.py:67) — currently hit every run
        # because backtest_ticker_vertical's friction lookup is keyed on
        # target_delta/wing_delta/entry_dte (fixed params), never the
        # per-trade realized delta/DTE, so spread_score is constant across
        # every trade. A refused hypothesis must stay distinguishable from
        # a measured null — see docs/HYPOTHESIS_SWEEP_PREREG.md.
        return HypothesisResult("H3", "weights", "ic", None, n_clusters, n_raw,
                                False, True, "ic_undefined_constant_feature")
    survives = lo is not None and hi is not None and (lo > 0 or hi < 0)
    notes = f"ci=({lo}, {hi}) alpha={H3_BONFERRONI_ALPHA:.5f}"
    return HypothesisResult("H3", "weights", "ic", ic, n_clusters, n_raw,
                            survives, False, None, notes)


def run_all(tickers: Optional[List[str]] = None) -> List[HypothesisResult]:
    """Run every preregistered hypothesis, in H1..H6 order. A refusal on
    one hypothesis never stops or shrinks the batch, and N_TRIALS stays 6
    for every hypothesis that does run."""
    return [
        run_h1_bull_put(tickers=tickers),
        run_h2_bear_call(tickers=tickers),
        run_h3_spread_weight(tickers=tickers),
        run_h4_entry_dte(tickers=tickers),
        run_h5_stop_loss(tickers=tickers),
        run_h6_credit_to_width(tickers=tickers),
    ]


def format_report(results: List[HypothesisResult],
                  provenance: Optional[str] = None) -> str:
    """Format hypothesis results into a readable table.

    `provenance` is the friction source string from `load_default_surface`
    ("surface" or "fallback_flat"). When given, it is printed in the header
    so a reader can't mistake a flat-fallback run for the real fitted
    surface — see the design spec's "Missing data/spread_surface.json" note.
    """
    lines = [
        f"Hypothesis sweep — family size N_TRIALS={N_TRIALS}",
    ]
    if provenance is not None:
        if provenance == "surface":
            lines.append(f"friction=surface ({DEFAULT_SURFACE_PATH})")
        else:
            lines.append(f"friction={provenance} ({DEFAULT_SURFACE_PATH} absent)")
    lines.append("-" * 72)
    for r in results:
        if r.refused:
            verdict = "REFUSED"
            detail = f"reason={r.reason}"
        else:
            verdict = "SURVIVES" if r.survives else "null"
            detail = f"{r.statistic_type}={r.value}"
        lines.append(
            f"{r.id:<3} {r.category:<10} {verdict:<9} n_raw={r.n_raw_trades:<5} "
            f"n_eff={r.n_eff} {detail} {r.notes}"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for the hypothesis sweep."""
    ap = argparse.ArgumentParser(description="Preregistered hypothesis sweep")
    ap.add_argument("--out", default=None, help="write results as JSON to this path")
    ap.add_argument("--only", nargs="*", default=None,
                    help="debug subset, e.g. --only H3 H4 (N_TRIALS still reports 6)")
    args = ap.parse_args()

    results = run_all()
    if args.only:
        results = [r for r in results if r.id in args.only]

    # Cheap local-file re-read purely to surface the friction provenance in
    # the report/JSON header — the six runners already loaded the surface
    # themselves for the actual backtest; this does not change any result.
    _, provenance = load_default_surface()

    print(format_report(results, provenance))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "friction_provenance": provenance,
                "results": [r.__dict__ for r in results],
            }, fh, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
