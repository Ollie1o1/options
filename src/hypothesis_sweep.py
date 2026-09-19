"""Preregistered hypothesis sweep over strategy/weights/entry-exit/gates.

The six hypotheses are locked in
docs/superpowers/specs/2026-09-19-hypothesis-sweep-design.md and copied
verbatim to docs/HYPOTHESIS_SWEEP_PREREG.md. This module only computes — it
must never add, remove, or retune a hypothesis after seeing a result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.alloc.validate import deflated_sharpe, effective_n
from src.backtest_optimizer import DEFAULT_UNIVERSE
from src.backtest_spreads import SpreadTrade, load_default_surface, run_vertical_backtest

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
