"""What a board is allowed to show, and what it is not allowed to claim.

Every board in this repo used to answer "which is best?" with some ordering —
`quality_score` on the squeeze board and the executive summary, `return_on_risk`
on condors, `lottery_ticket_score` on lottery, `ev_per_contract` after the
verdict work. Six keys, fifteen call sites, and a "#1" that meant something
different on every screen.

Two questions were put to the ledger on 2026-08-09 (`scripts/validate_gates.py`,
walk-forward, every threshold fitted strictly before the fold it is applied to):

  REMOVAL — does refusing what the ledger measures as a loser improve what is
  left? Yes. Five folds out of five, kept beat refused (+0.050 against -0.170
  mean return on capital), $17,639 of losses avoided. Note that $15,644 of that
  is one week in mid-July: the direction is consistent, the magnitude is not.

  RANKING — does any ordering of the survivors beat `quality_score` at the #1
  slot? No. A theta-cost ranker won 23 of 48 paired (day, board) cells,
  Wilcoxon p=0.89. `ev_score` is -0.325 on condors, `return_on_risk` -0.216,
  `pop` -0.302, `quality_score` -0.131 on long calls. Nothing ranks.

So this module refuses, and it does not rank. Survivors come back ordered by
cheapest carry, which is disclosed on the board as an ordering and not a
judgement. There is no #1, because the evidence does not support one.

Failure-safe like the rest of the scan path: if anything raises, every row is
kept and the board renders. A crash must never be able to hide candidates.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# Underlyings on which a condor is allowed. n=139 over the whole book: +9.5%
# mean return on capital and a 78% win rate here, against -11.8% and 37%
# everywhere else, Mann-Whitney p < 1e-5, same sign in both halves of the
# sample. Two earlier independent results agree — the equity-VRP backtest and
# the DoltHub index put spread — which is why this is the best-evidenced rule
# in the repo and the only one stated as a universe rather than a weight.
BROAD_INDEX = frozenset({"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI"})

LONG_SINGLE = frozenset({"Long Call", "Long Put"})

# G6 removes the top quintile of `quality_score` on long single legs: that
# bucket ran -15.7% mean and -$10,173 across the book while the three buckets
# below it made +$17,739. The cutoff is refit from the ledger on every scan
# rather than frozen here, so it tracks the book instead of rotting.
TOP_QUINTILE = 0.80

# Enough closed long-single-leg trades to set a quintile at all. Below this the
# gate disables itself rather than let a handful of trades judge the board.
MIN_LEDGER_ROWS = 30

# Evidence class per gate, printed on the board so a rule backed by the ledger
# is never confused with one backed by arithmetic alone.
MEASURED = "MEASURED"        # out-of-sample support in the ledger
ARITHMETIC = "ARITHMETIC"    # true by construction, no prediction involved
CONSISTENCY = "CONSISTENCY"  # the system refusing to contradict itself


@dataclass(frozen=True)
class Gate:
    """One reason a candidate is not shown."""
    key: str
    label: str
    evidence: str


GATES: Dict[str, Gate] = {
    "unquotable": Gate("unquotable", "no two-sided quote on every leg", ARITHMETIC),
    "friction": Gate("friction", "friction exceeds 25% of reward", ARITHMETIC),
    "credit_gone": Gate("credit_gone", "credit disappears once crossed", ARITHMETIC),
    "condor_universe": Gate("condor_universe", "condor off broad index", MEASURED),
    "top_quintile": Gate("top_quintile", "composite in its losing top quintile", MEASURED),
    "negative_ev": Gate("negative_ev", "negative expected value after cost", CONSISTENCY),
}

# Priority order for attributing a refusal. A candidate usually fails several
# gates; the board reports the first, running from "cannot be traded at all"
# through "measured to lose" to "the system contradicting itself".
_GATE_ORDER = ["unquotable", "friction", "credit_gone",
               "condor_universe", "top_quintile", "negative_ev"]


@dataclass
class BoardResult:
    """What survived, what did not, and why."""
    kept: pd.DataFrame
    refused: pd.DataFrame
    reasons: Counter = field(default_factory=Counter)
    scanned: int = 0
    cutoff: Optional[float] = None
    order_basis: str = "cheapest carry first — an ordering, not a ranking"
    # True when `gate_board` fell through to its failure-safe branch. Without
    # this, that branch returns a result byte-identical to a board on which
    # every candidate legitimately cleared every gate — so anything recording
    # the outcome would state "all passed" about a board that was never gated.
    # Recorded by `candidate_record`; no decision reads it.
    gating_failed: bool = False

    @property
    def empty(self) -> bool:
        return len(self.kept) == 0

    def summary_lines(self) -> List[str]:
        """Refusal counts, worst first, for the empty-board panel."""
        out = []
        for key, n in self.reasons.most_common():
            gate = GATES.get(key)
            label = gate.label if gate else key
            out.append(f"{n:>5}  {label}")
        return out


def _num(value: Any) -> Optional[float]:
    """A finite float, or None. Never raises.

    `dict.get` is typed `Any | None`, so passing its result straight to `float`
    is an error mypy blocks even when a `try` would catch it at runtime. A
    None-returning helper says the same thing in the type system, and NaN --
    which `float()` accepts happily -- is filtered here rather than by every
    caller remembering to.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _ticker_of(row: Any) -> str:
    for key in ("symbol", "ticker", "Symbol", "Ticker"):
        val = row.get(key)
        if val:
            return str(val).strip().upper()
    return ""


@lru_cache(maxsize=1)
def _top_quintile_cutoff(db_path: str, mtime: float) -> float:
    """G6's threshold, refit from closed long single legs in the ledger.

    `mtime` is not used in the body — it is in the signature so the cache
    invalidates when the ledger changes, which is the only thing that should
    move this number. Returns +inf when the ledger cannot be read or is too
    thin, which disables the gate rather than guessing a threshold.
    """
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "select quality_score from trades "
                "where status='CLOSED' and duplicate_of is null "
                "and strategy_name in ('Long Call','Long Put') "
                "and quality_score is not null"
            ).fetchall()
    except Exception:
        log.debug("ledger unreadable; top-quintile gate disabled", exc_info=True)
        return float("inf")

    scores = sorted(r[0] for r in rows if r[0] is not None)
    if len(scores) < MIN_LEDGER_ROWS:
        return float("inf")
    return float(pd.Series(scores).quantile(TOP_QUINTILE))


def top_quintile_cutoff(db_path: str = "paper_trades.db") -> float:
    """Public wrapper that keys the cache on the ledger's modification time."""
    try:
        import os
        mtime = os.path.getmtime(db_path)
    except OSError:
        return float("inf")
    return _top_quintile_cutoff(db_path, mtime)


def _refusal_for(row: Dict[str, Any], cutoff: float,
                 win_rates: Optional[Dict[str, float]] = None) -> Optional[str]:
    """The first gate this candidate fails, or None if it clears them all."""
    from . import candidate_verdict as cv

    strategy = str(row.get("strategy_name") or row.get("type") or "")
    failed = set()

    # G1-G3 come from the existing cost work, which already prices a candidate
    # at what it would actually fill for.
    try:
        name = row.get("strategy_name")
        wr = (win_rates or {}).get(name) if isinstance(name, str) else None
        verdict = cv.verdict_for(row, historical_win_rate=wr)
        if not verdict.priced:
            failed.add("unquotable")
        elif not verdict.passed:
            reason = (verdict.reason or "").lower()
            if "credit disappears" in reason:
                failed.add("credit_gone")
            else:
                failed.add("friction")
    except Exception:
        log.debug("verdict failed for a row; not refused on that basis",
                  exc_info=True)

    # G5 — measured, and scoped to condors only. It says nothing about single
    # legs and is not applied to them.
    if strategy == "Iron Condor" and _ticker_of(row) not in BROAD_INDEX:
        failed.add("condor_universe")

    # G6 — measured, and scoped to long single legs only. It says nothing about
    # spreads or condors and is not applied to them.
    if strategy in LONG_SINGLE:
        score = _num(row.get("quality_score"))
        if score is not None and score >= cutoff:
            failed.add("top_quintile")

    # G4 — not a prediction and not claimed as one. If the system computes a
    # negative expected value it must not also print BUY; this gate exists so
    # the board cannot contradict itself, which is the failure that made the
    # old cards unreadable (7 of 9 detail cards read NEGATIVE EV and all 9
    # printed a DO BUY line).
    #
    # It defers to `decide_verdict`, the same rule the top-N table and the
    # tearsheet render, rather than testing `ev < 0` directly. Net EV is
    # Black-Scholes over an implied vol this repo corrects on most scans, so a
    # raw sign test refuses -$1 while admitting +$9 — two numbers inside the
    # same error bar. A consistency gate that disagrees with the number printed
    # beside it would be the very defect it exists to remove.
    try:
        from .tearsheet.render import decide_verdict
        from .cli_display import _ev_noise_for_row

        decision, _ = decide_verdict(
            row.get("ev_per_contract"),
            row.get("ev_gross_per_contract"),
            row.get("ev_cost_per_contract"),
            None,
            noise=_ev_noise_for_row(row),
        )
        if decision == "SKIP":
            failed.add("negative_ev")
    except Exception:
        log.debug("EV verdict unavailable; not refused on that basis",
                  exc_info=True)

    for key in _GATE_ORDER:
        if key in failed:
            return key
    return None


def _carry_key(df: pd.DataFrame) -> pd.Series:
    """Cheapest carry first. Disclosed as an ordering, never as a ranking.

    Deliberately not a quality signal: no ordering of the survivors beat
    `quality_score` at the #1 slot out of sample (23 of 48 cells, p=0.89), so
    the board sorts on what a position costs to hold and says so.
    """
    def _col(name: str) -> pd.Series:
        """A numeric column, or an all-NaN one when the frame lacks it.

        `df.get(name)` returns None for a missing column and `pd.to_numeric`
        turns that into a bare float, not a Series — which raised on `.abs()`
        and sent the whole board through the failure-safe path, bypassing
        every gate. Ordering must degrade to "unordered", never to "ungated".
        """
        if name not in df.columns:
            return pd.Series(float("nan"), index=df.index, dtype="float64")
        return pd.to_numeric(df[name], errors="coerce")

    theta = _col("theta").abs()
    premium = _col("premium")
    key = theta / premium.replace(0, pd.NA)
    return key.fillna(key.max() if key.notna().any() else 0.0)


def gate_board(df: Optional[pd.DataFrame],
               *,
               db_path: str = "paper_trades.db",
               win_rates: Optional[Dict[str, float]] = None,
               label_structures: bool = False) -> BoardResult:
    """Split a board into what may be shown and what may not.

    `label_structures` names spread and condor rows first, because the gates
    read structure off `strategy_name` and those rows do not carry one.
    """
    if df is None or len(df) == 0:
        return BoardResult(kept=df if df is not None else pd.DataFrame(),
                           refused=pd.DataFrame(), scanned=0)

    try:
        work = df.copy()
        if label_structures and "strategy_name" not in work.columns:
            from .options_screener import structure_strategy_name
            work["strategy_name"] = [structure_strategy_name(r)
                                     for _, r in work.iterrows()]

        cutoff = top_quintile_cutoff(db_path)
        refusals = [_refusal_for(r, cutoff, win_rates)
                    for r in work.to_dict("records")]
        work["refused_by"] = refusals

        mask = work["refused_by"].isna()
        kept = work[mask].copy()
        refused = work[~mask].copy()

        if len(kept):
            kept = kept.assign(_carry=_carry_key(kept)) \
                       .sort_values("_carry") \
                       .drop(columns=["_carry"]) \
                       .reset_index(drop=True)

        return BoardResult(
            kept=kept.drop(columns=["refused_by"]),
            refused=refused,
            reasons=Counter(r for r in refusals if r),
            scanned=len(work),
            cutoff=None if cutoff == float("inf") else cutoff,
        )
    except Exception:
        # A board that renders everything is a bug; a board that renders
        # nothing because of a bug is worse. Keep the scan alive.
        log.warning("gating failed; showing every candidate", exc_info=True)
        return BoardResult(kept=df, refused=pd.DataFrame(), scanned=len(df),
                           gating_failed=True)
