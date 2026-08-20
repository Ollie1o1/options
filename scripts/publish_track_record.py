#!/usr/bin/env python3
"""
Publish a public paper-trading track record to reports/TRACK_RECORD.md.

Reads closed trades from paper_trades.db and renders an honest, plainly-caveated
summary: dollar-weighted headline, per-strategy breakdown, the forward-cohort
gate status, methodology notes, and a full timestamped table of closed trades.

Why dollar-weighted
-------------------
The lead figure is net dollar P&L and return on total capital risked
(sum pnl_usd / sum capital_at_risk). An unweighted mean of per-trade percentage
returns counts a $28 credit spread and a $27,000 cash-secured put equally, and
this book's risk per trade spans three orders of magnitude — so the unweighted
mean describes a portfolio nobody could have held. It is kept as a clearly
labelled secondary line, never as the headline. Every percentage in this
document names its basis (of capital risked / of credit collected / unweighted).

Why an equal-weighted section beside it
---------------------------------------
The dollar headline describes a book in which position size was never chosen:
every ledger row carried `quantity = 1.0` until 2026-08-20, so bet size was the
option's premium — share price and implied volatility, not the pick. A reader
cannot tell from the headline alone how much of it is the picks and how much is
which trades happened to be large. `equal_weighted` answers that by giving every
closed trade the SAME capital at risk.

The basis is capital at risk, deliberately, not entry premium: premium is a
debit on long structures and a credit on short ones, so equalising it compares
two different quantities — and on this book the two bases disagree about the
SIGN of the result. The interval is published beside the point estimate and the
document says in words whether it contains 1, because a profit factor of 1.06
reads like an edge and an interval of [0.88, 1.26] is not one. The bootstrap is
seeded so a regenerated file does not churn.

Publish flow
------------
Weekly startup maintenance regenerates this file (`_run_track_record` in
src/maintenance.py, throttled to once a week) and the regenerated file is meant
to be **committed as part of that publish** — reports/TRACK_RECORD.md is one of
the few files under reports/ that is git-tracked (see the `!reports/TRACK_RECORD.md`
negation in .gitignore). Left uncommitted it silently drifts from the ledger and
the published record stops matching the DB. So, after any maintenance run:

    ./scripts/test.sh track_record                  # sanity
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/publish_track_record.py
    git add reports/TRACK_RECORD.md && git commit -m "chore: refresh track record"

The script never writes trade rows, and every query it issues opens the db
through a read-only URI (`file:...?mode=ro`), so it cannot migrate the schema
or take a write lock on a ledger another process is using.

Pure rendering (`render_track_record`) is separated from I/O so it is testable
against a seeded in-memory SQLite db.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import statistics
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Allow running as a plain script (python scripts/publish_track_record.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence import load_model_evidence, format_evidence_banner  # noqa: E402

_CLOSED_COLUMNS = [
    "date", "ticker", "strategy_name", "type", "strike", "expiration",
    "entry_price", "exit_price", "pnl_pct", "pnl_usd", "exit_reason",
    "paper_only", "status", "capital_at_risk", "net_credit", "quantity",
]

# Structures opened for a credit. Their capital_at_risk denominator is posted
# collateral (cash-secured shorts) or spread width, not premium paid, so return
# on credit collected is reported beside return on risk for these lines.
_CREDIT_STRATEGIES = ("Short Put", "Bull Put", "Bear Call", "Iron Condor")

# Cash-secured shorts specifically: collateral is the whole strike, which is
# 50-100x the credit, so their return-on-risk is compressed towards zero by the
# denominator rather than by flat performance.
_CASH_SECURED = ("Short Put",)


# --------------------------------------------------------------------------- #
# Data access (read-only)
# --------------------------------------------------------------------------- #

def fetch_closed_trades(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all CLOSED trades as a list of plain dicts, oldest first.

    Selects the intersection of the wanted columns and the columns the table
    actually has, so an older or seeded schema (no capital_at_risk) still
    renders — the dollar-weighted figures degrade to 'n/a' rather than raising.
    """
    cur = conn.cursor()
    try:
        have = {r[1] for r in cur.execute("PRAGMA table_info(trades)")}
    except sqlite3.OperationalError:
        return []
    cols = [c for c in _CLOSED_COLUMNS if c in have]
    if not cols:
        return []
    # Rows ruled double-logs are one decision recorded twice; publishing both
    # double-counts their P&L in the headline. They stay in the ledger, and out
    # of the record. Older ledgers have no such column and are unaffected.
    dupes = " AND duplicate_of IS NULL" if "duplicate_of" in have else ""
    try:
        cur.execute(
            f"SELECT {', '.join(cols)} FROM trades "
            f"WHERE UPPER(status) = 'CLOSED'{dupes} ORDER BY date ASC"
        )
    except sqlite3.OperationalError:
        return []
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _load_budget_cap(config_path: str = "config.json") -> Optional[float]:
    """`auto_log.max_capital_at_risk` — the per-position ceiling the ledger
    enforces. Never hardcoded here: the published subset must match whatever
    the ledger is actually refusing trades above."""
    try:
        with open(config_path) as f:
            cap = (json.load(f).get("auto_log") or {}).get("max_capital_at_risk")
        return float(cap) if cap not in (None, "", 0, False) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #

def _f(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _risked(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows carrying a usable (capital_at_risk > 0, pnl_usd recorded) pair."""
    out = []
    for r in rows:
        car, pnl = _f(r.get("capital_at_risk")), _f(r.get("pnl_usd"))
        if car and car > 0 and pnl is not None:
            out.append(r)
    return out


def per_trade_returns_on_risk(rows: Iterable[Dict[str, Any]]) -> List[float]:
    """pnl_usd / capital_at_risk for each row that has both."""
    return [_f(r["pnl_usd"]) / _f(r["capital_at_risk"]) for r in _risked(rows)]


def credit_collected(row: Dict[str, Any]) -> Optional[float]:
    """Dollars of premium taken in on one credit position.

    `net_credit` is stored per share on spreads and condors and is NULL on
    single-leg cash-secured shorts, where the entry price *is* the credit.
    Both are per-share, so x100 x contracts.
    """
    per_share = _f(row.get("net_credit"))
    if per_share is None:
        per_share = _f(row.get("entry_price"))
    if per_share is None:
        return None
    qty = _f(row.get("quantity"))
    if qty is None or qty <= 0:
        qty = 1.0
    return abs(per_share) * 100.0 * qty


#: Capital at risk every trade is given in the equal-weighted view. A round
#: number, close to the live per-trade budget (2% of ~$41k equity = ~$820), and
#: fixed rather than derived so the published figure does not move when the
#: book's equity does.
EQUAL_WEIGHT_RISK = 1_000.0

#: Bootstrap resamples, and the seed that makes the interval reproducible. This
#: file is committed; an unseeded interval would differ on every regeneration
#: and the diff would stop being readable.
_BOOTSTRAP_N = 4_000
_BOOTSTRAP_SEED = 20260820


def profit_factor(returns: Sequence[float]) -> Optional[float]:
    """Gross wins over gross losses, or None when the ratio has no meaning.

    None for an empty sample and None for a sample with NO losses: an infinite
    profit factor is a statement about sample size, not about performance, and
    printing "inf" invites reading it as one.
    """
    wins = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    if not returns or losses <= 0:
        return None
    return wins / losses


def _pf_interval(returns: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Seeded 95% bootstrap interval for `profit_factor`."""
    if len(returns) < 2:
        return None, None
    rnd = random.Random(_BOOTSTRAP_SEED)
    draws: List[float] = []
    for _ in range(_BOOTSTRAP_N):
        pf = profit_factor([rnd.choice(returns) for _ in returns])
        if pf is not None:
            draws.append(pf)
    if len(draws) < _BOOTSTRAP_N // 2:
        # Too many resamples had no losing trade for an interval to mean much.
        return None, None
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def equal_weighted(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """The book with every trade given the same capital at risk.

    Answers "how much of the headline is the picks and how much is which trades
    happened to be large" — the question the dollar figures cannot separate
    while every historical row carries `quantity = 1.0`.

    Rows with no recorded capital at risk are EXCLUDED, never counted as zero
    risk: NULL means the column was not written, and a zero denominator is not
    a free trade.
    """
    returns = per_trade_returns_on_risk(rows)
    pf = profit_factor(returns)
    low, high = _pf_interval(returns)
    return {
        "n": len(returns),
        "profit_factor": pf,
        "ci_low": low,
        "ci_high": high,
        "risk_per_trade": EQUAL_WEIGHT_RISK,
        "net_pnl": sum(returns) * EQUAL_WEIGHT_RISK if returns else None,
        "mean_return": (sum(returns) / len(returns)) if returns else None,
        # Same exercise on entry premium instead of capital at risk. Published
        # as the contrast, because on this book it flips the sign.
        "profit_factor_on_premium": profit_factor(
            [r for r in (_f(x.get("pnl_pct")) for x in rows) if r is not None]),
    }


def summarize_equal_weighted_strategies(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """`equal_weighted` per strategy, sorted by closed count.

    A strategy too short to bootstrap is still listed, with a blank interval:
    dropping it would silently remove a line from the comparison, which is the
    opposite of what a track record is for.
    """
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(str(row.get("strategy_name") or "Unknown"), []).append(row)
    out = []
    for name, group in by.items():
        stats = equal_weighted(group)
        stats["strategy"] = name
        stats["n_closed"] = len(group)
        out.append(stats)
    return sorted(out, key=lambda s: (-s["n_closed"], s["strategy"]))


def summarize_book(rows: Sequence[Dict[str, Any]],
                   budget_cap: Optional[float] = None) -> Dict[str, Any]:
    """Headline figures for the whole closed book.

    `return_on_risk` is the dollar-weighted number that leads the document;
    `mean_return_unweighted` is the same statistic the old headline used and is
    retained only as a labelled secondary line.
    """
    scored = [r for r in rows if _f(r.get("pnl_pct")) is not None]
    wins = [r for r in scored if _f(r["pnl_pct"]) > 0]
    risked = _risked(rows)
    total_risk = sum(_f(r["capital_at_risk"]) for r in risked)
    pnl_on_risk = sum(_f(r["pnl_usd"]) for r in risked)
    net_pnl = sum(_f(r["pnl_usd"]) for r in rows if _f(r.get("pnl_usd")) is not None)

    summary: Dict[str, Any] = {
        "n_closed": len(rows),
        "n_scored": len(scored),
        "n_wins": len(wins),
        "n_unscored": len(rows) - len(scored),
        "win_rate": (len(wins) / len(scored)) if scored else None,
        "net_pnl": net_pnl if any(_f(r.get("pnl_usd")) is not None for r in rows) else None,
        "n_with_pnl": sum(1 for r in rows if _f(r.get("pnl_usd")) is not None),
        "capital_at_risk": total_risk if risked else None,
        "return_on_risk": (pnl_on_risk / total_risk) if total_risk > 0 else None,
        # The numerator `return_on_risk` actually divides. It differs from
        # `net_pnl` whenever a closed trade has a dollar result but no recorded
        # capital at risk, so the published ratio must quote this one to
        # reconcile against its own denominator.
        "net_pnl_risked": pnl_on_risk if risked else None,
        "n_risked": len(risked),
        "median_return_on_risk": _median(per_trade_returns_on_risk(rows)),
        "mean_return_unweighted": (
            sum(_f(r["pnl_pct"]) for r in scored) / len(scored)) if scored else None,
        "budget_cap": budget_cap,
        "affordable": None,
    }

    if budget_cap:
        aff = [r for r in risked if _f(r["capital_at_risk"]) <= budget_cap]
        aff_risk = sum(_f(r["capital_at_risk"]) for r in aff)
        if aff_risk > 0:
            summary["affordable"] = {
                "n": len(aff),
                "net_pnl": sum(_f(r["pnl_usd"]) for r in aff),
                "capital_at_risk": aff_risk,
                "return_on_risk": sum(_f(r["pnl_usd"]) for r in aff) / aff_risk,
            }
    return summary


def summarize_strategies(
    rows: Sequence[Dict[str, Any]],
    breakdown: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Per-strategy stats, one dict per strategy, sorted by name.

    `breakdown` is the output of `PaperManager.get_strategy_breakdown()`,
    which already defines `return_on_risk` as total P&L over total capital at
    risk. When supplied it is the authority for that field so the published
    number and the portfolio view's number cannot drift apart; the medians and
    net dollars (which the breakdown does not carry — its `total_pnl` is a sum
    of percentages, not dollars) are computed here from the rows.
    """
    by_strat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_strat.setdefault(r.get("strategy_name") or "Unknown", []).append(r)
    ror_from_breakdown = {
        (b.get("strategy") or "Unknown"): b.get("return_on_risk")
        for b in (breakdown or [])
    }

    out: List[Dict[str, Any]] = []
    for strat in sorted(by_strat):
        srows = by_strat[strat]
        scored = [r for r in srows if _f(r.get("pnl_pct")) is not None]
        wins = [r for r in scored if _f(r["pnl_pct"]) > 0]
        risked = _risked(srows)
        total_risk = sum(_f(r["capital_at_risk"]) for r in risked)
        net_pnl = sum(_f(r["pnl_usd"]) for r in srows
                      if _f(r.get("pnl_usd")) is not None)
        own_ror = (sum(_f(r["pnl_usd"]) for r in risked) / total_risk
                   if total_risk > 0 else None)
        out.append({
            "strategy": strat,
            "n_closed": len(srows),
            "n_scored": len(scored),
            "n_wins": len(wins),
            "win_rate": (len(wins) / len(scored)) if scored else None,
            "net_pnl": net_pnl,
            "capital_at_risk": total_risk if risked else None,
            # Presence-and-not-None, never `or`: a legitimate 0.0 from the
            # breakdown must survive rather than fall through to the recompute.
            "return_on_risk": (ror_from_breakdown[strat]
                               if ror_from_breakdown.get(strat) is not None
                               else own_ror),
            "median_return_on_risk": _median(per_trade_returns_on_risk(srows)),
            "mean_return_unweighted": (
                sum(_f(r["pnl_pct"]) for r in scored) / len(scored)) if scored else None,
        })
    return out


def summarize_credit_strategies(
    rows: Sequence[Dict[str, Any]],
    strategies: Sequence[str] = _CREDIT_STRATEGIES,
) -> List[Dict[str, Any]]:
    """Return-on-credit companion for credit structures.

    On a cash-secured short the risk denominator is the posted collateral —
    strike x 100 less the credit — which is two orders of magnitude larger than
    the premium at stake, so return on risk reads as a near-flat line whatever
    the trade did. Return on credit collected is the figure that actually moves.
    """
    wanted = set(strategies)
    by_strat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        name = r.get("strategy_name") or "Unknown"
        if name in wanted:
            by_strat.setdefault(name, []).append(r)

    out: List[Dict[str, Any]] = []
    for strat in sorted(by_strat):
        srows = by_strat[strat]
        pairs = [(credit_collected(r), _f(r.get("pnl_usd"))) for r in srows]
        pairs = [(c, p) for c, p in pairs if c and c > 0 and p is not None]
        if not pairs:
            continue
        credit = sum(c for c, _ in pairs)
        pnl = sum(p for _, p in pairs)
        risked = _risked(srows)
        total_risk = sum(_f(r["capital_at_risk"]) for r in risked)
        out.append({
            "strategy": strat,
            "n": len(pairs),
            "credit_collected": credit,
            "net_pnl": pnl,
            "return_on_credit": pnl / credit,
            "median_return_on_credit": _median([p / c for c, p in pairs]),
            "return_on_risk": (sum(_f(r["pnl_usd"]) for r in risked) / total_risk
                               if total_risk > 0 else None),
            "cash_secured": strat in _CASH_SECURED,
        })
    return out


def sign_divergences(strategies: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strategy lines whose aggregate and median return on risk disagree in
    sign — i.e. the line is carried by a small number of large trades and the
    typical trade went the other way."""
    out = []
    for s in strategies:
        agg, med = s.get("return_on_risk"), s.get("median_return_on_risk")
        if agg is None or med is None:
            continue
        if (agg > 0) != (med > 0):
            out.append(s)
    return out


def _largest_contributor(rows: Sequence[Dict[str, Any]],
                         strategy: str) -> Optional[Dict[str, Any]]:
    """The single closed trade with the largest P&L in a strategy, plus its
    share of that strategy's net — the arithmetic behind a sign divergence."""
    srows = [r for r in rows
             if (r.get("strategy_name") or "Unknown") == strategy
             and _f(r.get("pnl_usd")) is not None]
    if not srows:
        return None
    # Largest by MAGNITUDE, not by signed value: when a line's aggregate is
    # negative and its median positive, the trade that explains the divergence
    # is the big loser, and picking the best winner would print a confidently
    # wrong explanation.
    top = max(srows, key=lambda r: abs(_f(r["pnl_usd"])))
    net = sum(_f(r["pnl_usd"]) for r in srows)
    return {
        "ticker": top.get("ticker") or "—",
        "date": top.get("date") or "—",
        "pnl_usd": _f(top["pnl_usd"]),
        "share_of_net": (_f(top["pnl_usd"]) / net) if net else None,
    }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def _fmt_pct(v: Optional[float], places: int = 1) -> str:
    """Fractions (0.42) rendered as signed percent (+42.0%)."""
    if v is None:
        return "n/a"
    try:
        return f"{float(v) * 100.0:+.{places}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_money(v: Optional[float]) -> str:
    try:
        return f"${float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_signed_money(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{'-' if float(v) < 0 else '+'}${abs(float(v)):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _fmt_dollars(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------- #
# Methodology notes
# --------------------------------------------------------------------------- #

def methodology_notes(rows: Sequence[Dict[str, Any]],
                      book: Dict[str, Any],
                      strategies: Sequence[Dict[str, Any]],
                      credit: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(heading, body) pairs rendered under '## Methodology notes'.

    Each note is independent and appended in order. To add one, write a
    `_note_*` helper returning `Optional[Tuple[str, str]]` and append its
    result below — nothing else in the renderer needs to change.
    """
    notes: List[Optional[Tuple[str, str]]] = [
        _note_weighting_basis(book),
        _note_median_vs_aggregate(rows, strategies),
        _note_collateral_denominator(credit),
        _note_stop_overshoot(rows),
    ]
    # Further notes append here (one `_note_*` helper each).
    return [n for n in notes if n]


def _note_stop_overshoot(rows: Sequence[Dict[str, Any]]
                         ) -> Optional[Tuple[str, str]]:
    """How far stopped trades ran past their stop, and why.

    Measured by scripts/overshoot_report.py, rendered here so a reader of the
    published record cannot see the -157.5% rows without also seeing that they
    are a check-frequency artifact rather than a market gap.
    """
    try:
        from scripts.overshoot_report import (format_summary, is_stop_exit,
                                              summarize)
    except Exception:  # pragma: no cover - defensive
        return None
    stops = [r for r in rows if is_stop_exit(r.get("exit_reason"))]
    if not stops:
        return None
    summary = summarize(stops)
    if not summary["n_levelled"]:
        return None
    return ("Stops overshot because exits were checked by hand",
            format_summary(summary))


def _note_weighting_basis(book: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    cap = book.get("budget_cap")
    aff = book.get("affordable")
    body = (
        "Every percentage in this document names its basis. **Of capital risked** "
        "means dollars of P&L over dollars of capital_at_risk (the ledger's own "
        "per-position risk figure: premium paid on debits, collateral or width "
        "less credit on credits). **Of credit collected** is P&L over premium "
        "taken in. **Unweighted mean** is the arithmetic mean of per-trade "
        "percentage returns, which counts every trade equally no matter its size "
        "and is reported only as a secondary line."
    )
    if cap and aff:
        body += (
            f"\n\nSize dominates the raw aggregate: of "
            f"{_fmt_dollars(book.get('capital_at_risk'))} risked across the book, "
            f"only {_fmt_dollars(aff.get('capital_at_risk'))} sat inside the "
            f"{_fmt_dollars(cap)} per-position ceiling the ledger now enforces "
            "(`auto_log.max_capital_at_risk`). The oversized positions are a "
            "sizing artifact of an unbounded feeder, not a strategy result, which "
            "is why the affordable subset is published beside the whole book."
        )
    return ("Weighting and bases", body)


def _note_median_vs_aggregate(rows: Sequence[Dict[str, Any]],
                              strategies: Sequence[Dict[str, Any]]
                              ) -> Optional[Tuple[str, str]]:
    body = (
        "Aggregate return on risk is a dollar-weighted number: one large "
        "contract can carry a whole line. The median per-trade return on risk is "
        "published beside it so the typical trade is visible. Where the two "
        "disagree in sign, the aggregate is a story about one or two positions."
    )
    diverging = sign_divergences(strategies)
    if not diverging:
        body += "\n\nNo strategy line currently disagrees in sign."
        return ("Median vs aggregate", body)

    lines = []
    for s in diverging:
        top = _largest_contributor(rows, s["strategy"])
        frag = (
            f"- **{s['strategy']}**: aggregate "
            f"{_fmt_pct(s['return_on_risk'])} of capital risked but median "
            f"{_fmt_pct(s['median_return_on_risk'])} per trade"
        )
        if top and top.get("share_of_net") is not None:
            frag += (
                f" — one {top['ticker']} trade ({_fmt_signed_money(top['pnl_usd'])}) "
                f"is {top['share_of_net'] * 100:+.0f}% of the line's net"
            )
        lines.append(frag + ".")
    return ("Median vs aggregate", body + "\n\n" + "\n".join(lines))


def _note_collateral_denominator(credit: Sequence[Dict[str, Any]]
                                 ) -> Optional[Tuple[str, str]]:
    cash = [c for c in credit if c.get("cash_secured")]
    if not cash:
        return None
    lines = []
    for c in cash:
        lines.append(
            f"- **{c['strategy']}**: {_fmt_pct(c['return_on_risk'])} of capital "
            f"risked vs {_fmt_pct(c['return_on_credit'])} of "
            f"{_fmt_dollars(c['credit_collected'])} credit collected "
            f"({c['n']} closed)."
        )
    body = (
        "A cash-secured short posts the whole strike as collateral, so its "
        "capital_at_risk denominator is roughly fifty to a hundred times the "
        "premium at stake. Return on risk therefore reads as a near-flat line "
        "however the trade actually went — that flatness is the denominator, not "
        "the result. Return on credit collected is published as the companion "
        "figure and is the one that moves.\n\n" + "\n".join(lines)
    )
    return ("Cash-secured collateral denominator", body)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_pf(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _fmt_ci(low: Optional[float], high: Optional[float]) -> str:
    if low is None or high is None:
        return "n/a"
    return f"[{low:.2f}, {high:.2f}]"


def _render_equal_weighted(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The same book with every trade given the same capital at risk."""
    eq = equal_weighted(rows)
    out: List[str] = ["## Equal-weighted — the same book with size taken out", ""]
    if not eq["n"]:
        out.append("_No closed trade carries a recorded capital at risk._")
        out.append("")
        return out

    size = _fmt_dollars(eq["risk_per_trade"])
    out.append(
        f"Every closed trade given the **same {size} of capital at risk**. Until "
        "2026-08-20 no ledger row recorded a chosen position size — `quantity` "
        "was 1.0 on all of them — so bet size was the option's premium, a "
        "function of share price and implied volatility rather than of the pick. "
        "This is the headline with that removed."
    )
    out.append("")
    out.append(f"- Equal-weighted P&L: **{_fmt_signed_money(eq['net_pnl'])}** "
               f"at {size} a trade across {eq['n']} closed trades")
    out.append(f"- Mean return per trade: **{_fmt_pct(eq['mean_return'])}** of "
               "capital risked")
    pf_txt = _fmt_pf(eq["profit_factor"])
    out.append(f"- Profit factor, equal-weighted: **{pf_txt}** "
               f"(95% bootstrap CI {_fmt_ci(eq['ci_low'], eq['ci_high'])})")
    if eq["ci_low"] is not None and eq["ci_high"] is not None:
        if eq["ci_low"] <= 1.0 <= eq["ci_high"]:
            out.append("  - **The interval contains 1**, so no book-level edge is "
                       "established: this book has not been shown to make money "
                       "per trade, whatever the dollar headline says.")
        elif eq["ci_low"] > 1.0:
            out.append("  - The interval sits entirely above 1.")
        else:
            out.append("  - The interval sits entirely below 1.")
    prem = eq["profit_factor_on_premium"]
    if prem is not None and eq["profit_factor"] is not None:
        out.append(
            f"- The same exercise on **entry premium** instead of capital at "
            f"risk gives {_fmt_pf(prem)}"
            + (", a different sign of result from the same trades — which is why "
               "the basis has to be stated. Capital at risk is used above "
               "because premium is a debit on long structures and a credit on "
               "short ones."
               if (prem - 1.0) * (eq["profit_factor"] - 1.0) < 0 else
               ". Both bases agree on the direction here.")
        )
    out.append("")

    per_strategy = summarize_equal_weighted_strategies(rows)
    if len(per_strategy) > 1:
        out.append("| Strategy | Closed | Profit factor (equal-weighted) | "
                   "95% CI | Mean return on risk |")
        out.append("|----------|-------:|------:|:------:|------:|")
        for s in per_strategy:
            out.append(
                f"| {s['strategy']} | {s['n_closed']} | "
                f"{_fmt_pf(s['profit_factor'])} | "
                f"{_fmt_ci(s['ci_low'], s['ci_high'])} | "
                f"{_fmt_pct(s['mean_return'])} |"
            )
        out.append("")
        out.append("_A line whose interval contains 1 has not been shown to have "
                   "an edge at this sample size; one whose interval sits below 1 "
                   "has been shown to lose. Intervals are seeded bootstraps over "
                   "per-trade returns on capital at risk, so they are stable "
                   "across regenerations of this file._")
        out.append("")
    return out


def render_track_record(rows: List[Dict[str, Any]],
                        evidence: Dict[str, Any],
                        breakdown: Optional[Sequence[Dict[str, Any]]] = None,
                        budget_cap: Optional[float] = None) -> str:
    """Render the closed-trade rows + model evidence into a Markdown document."""
    book = summarize_book(rows, budget_cap=budget_cap)
    strategies = summarize_strategies(rows, breakdown=breakdown)
    credit = summarize_credit_strategies(rows)
    n = book["n_closed"]

    out: List[str] = []
    out.append("# Paper Trading Track Record")
    out.append("")
    out.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M} • {n} closed trades_")
    out.append("")
    out.append("> **Methodology & caveats.** These are **paper trades**, not live "
               "fills. Entries and exits use **delayed retail data** (Yahoo Finance) "
               "and a **modeled friction** assumption (spread/slippage), so realized "
               "results would differ. The descriptive stats below are real; the "
               "**predictive edge of the ranking model is still under out-of-sample "
               "evaluation** and is *not* established — see "
               "[docs/VALIDATION_POWER.md](../docs/VALIDATION_POWER.md).")
    out.append("")
    out.append(f"_{format_evidence_banner(evidence)}_")
    out.append("")

    # --- Headline (dollar-weighted) -----------------------------------------
    out.append("## Headline")
    out.append("")
    out.append(f"- Net P&L: **{_fmt_signed_money(book['net_pnl'])}** across "
               f"{book['n_with_pnl']} closed trades with a recorded dollar result")
    # The ratio's numerator is the risked subset's net, not the book's — those
    # differ whenever a closed trade has a dollar result but no capital_at_risk
    # (legacy rows). Render the numerator that is actually divided, so the
    # published line always reconciles against its own denominator.
    out.append(f"- Return on capital risked: **{_fmt_pct(book['return_on_risk'])}** "
               f"({_fmt_signed_money(book['net_pnl_risked'])} of "
               f"{_fmt_dollars(book['capital_at_risk'])} risked across "
               f"{book['n_risked']} trades with capital_at_risk recorded)")
    if book.get("n_with_pnl") != book.get("n_risked"):
        out.append(
            f"  - {_plural(book['n_with_pnl'] - book['n_risked'], 'closed trade')} "
            f"carry a dollar result but no recorded capital at risk, so the "
            f"ratio above is computed over less than the full net P&L"
        )
    aff = book.get("affordable")
    if aff and book.get("budget_cap"):
        out.append(
            f"- Within the {_fmt_dollars(book['budget_cap'])} per-position ceiling: "
            f"**{_fmt_pct(aff['return_on_risk'])}** of capital risked "
            f"({_fmt_signed_money(aff['net_pnl'])} of "
            f"{_fmt_dollars(aff['capital_at_risk'])} risked, {aff['n']} trades) — "
            "the subset the account could actually have held"
        )
    out.append("")
    out.append("Secondary, size-blind figures:")
    out.append("")
    wr = book["win_rate"]
    wr_txt = f"{wr * 100:.1f}%" if wr is not None else "n/a"
    out.append(
        f"- Win rate: **{wr_txt}** = {_plural(book['n_wins'], 'win')} / "
        f"{_plural(book['n_scored'], 'closed trade')} with a recorded return; "
        f"{_plural(book['n_unscored'], 'closed trade')} excluded for missing "
        "returns"
    )
    out.append(f"- Mean return per trade: **{_fmt_pct(book['mean_return_unweighted'])}** "
               "(unweighted mean of per-trade returns **on entry premium** — a "
               "$28 spread counts the same as a $27,000 cash-secured put, and "
               "the premium denominator is a debit on long structures but a "
               "credit on short ones; not the headline for either reason)")
    out.append(f"- Median return per trade: **{_fmt_pct(book['median_return_on_risk'])}** "
               "of capital risked (typical trade, size-blind)")
    out.append("")

    # --- Equal-weighted: the headline with size taken out --------------------
    out.extend(_render_equal_weighted(rows))

    # --- Per-strategy breakdown ----------------------------------------------
    out.append("## By strategy")
    out.append("")
    out.append("| Strategy | Closed | Win rate | Net $ | Return on risk "
               "(aggregate, of capital risked) | Median return on risk "
               "(per trade, of capital risked) | Mean return per trade "
               "(unweighted) |")
    out.append("|----------|-------:|---------:|------:|------:|------:|------:|")
    for s in strategies:
        swr = f"{s['win_rate'] * 100:.1f}%" if s["win_rate"] is not None else "n/a"
        out.append(
            f"| {s['strategy']} | {s['n_closed']} | {swr} | "
            f"{_fmt_signed_money(s['net_pnl'])} | "
            f"{_fmt_pct(s['return_on_risk'])} | "
            f"{_fmt_pct(s['median_return_on_risk'])} | "
            f"{_fmt_pct(s['mean_return_unweighted'])} |"
        )
    out.append("")
    out.append("_Win rate counts trades with a recorded return; aggregate return on "
               "risk and median return on risk count trades with capital_at_risk "
               "recorded. Where aggregate and median disagree in sign, see the "
               "methodology notes._")
    out.append("")

    # --- Credit structures: return on credit ---------------------------------
    if credit:
        out.append("## Credit structures: return on credit collected")
        out.append("")
        out.append("| Strategy | Closed | Credit collected | Net $ | Return on "
                   "credit (of credit collected) | Median return on credit "
                   "(per trade) | Return on risk (of capital risked) |")
        out.append("|----------|-------:|-----------------:|------:|------:|------:|------:|")
        for c in credit:
            out.append(
                f"| {c['strategy']}{' (cash-secured)' if c['cash_secured'] else ''} | "
                f"{c['n']} | {_fmt_dollars(c['credit_collected'])} | "
                f"{_fmt_signed_money(c['net_pnl'])} | "
                f"{_fmt_pct(c['return_on_credit'])} | "
                f"{_fmt_pct(c['median_return_on_credit'])} | "
                f"{_fmt_pct(c['return_on_risk'])} |"
            )
        out.append("")

    # --- Forward-cohort gate -------------------------------------------------
    out.append("## Forward-cohort gate")
    out.append("")
    out.append(f"- Gate decision: **{evidence.get('gate_decision', 'UNKNOWN')}**")
    out.append(f"- Cohort size: **{evidence.get('cohort_n', 0)}** "
               "(closed cohort trades accumulated since the gate window opened)")
    out.append("")

    # --- Methodology notes ---------------------------------------------------
    notes = methodology_notes(rows, book, strategies, credit)
    if notes:
        out.append("## Methodology notes")
        out.append("")
        for heading, body in notes:
            out.append(f"### {heading}")
            out.append("")
            out.append(body)
            out.append("")

    # --- Full table ----------------------------------------------------------
    out.append("## Closed trades")
    out.append("")
    out.append("")
    out.append("`P&L % of premium` is the per-trade return on the **entry "
               "premium** — the debit paid on Long Call/Long Put, and the "
               "credit received on Short Put and every spread. Those are "
               "different denominators, so this column is not comparable "
               "across structures; `P&L $` and `Capital at risk` are.")
    out.append("")
    out.append("| Date | Ticker | Structure | Entry | Exit | P&L % of premium | "
               "P&L $ | Capital at risk | Exit reason |")
    out.append("|------|--------|-----------|------:|-----:|------:|------:|"
               "----------------:|-------------|")
    for r in rows:
        out.append(
            f"| {r.get('date', '—')} | {r.get('ticker', '—')} | "
            f"{r.get('strategy_name', '—')} | {_fmt_money(r.get('entry_price'))} | "
            f"{_fmt_money(r.get('exit_price'))} | {_fmt_pct(r.get('pnl_pct'))} | "
            f"{_fmt_signed_money(_f(r.get('pnl_usd')))} | "
            f"{_fmt_dollars(_f(r.get('capital_at_risk')))} | "
            f"{r.get('exit_reason') or '—'} |"
        )
    out.append("")
    return "\n".join(out)


def _load_breakdown(db_path: str) -> Optional[List[Dict[str, Any]]]:
    """`PaperManager.get_strategy_breakdown()`, or None if unavailable.

    Reused rather than reimplemented so the published return-on-risk and the
    portfolio view's return-on-risk are the same definition by construction.

    The query itself is a single SELECT, but constructing a `PaperManager`
    is NOT read-only — it runs `_init_db`/`_migrate_db` (CREATE TABLE IF NOT
    EXISTS, ALTER TABLEs, `PRAGMA user_version`). Publishing must never be the
    thing that migrates a ledger, least of all an archived copy someone kept
    precisely to preserve its old shape. So the schema version is read through
    a read-only connection first, and the manager is constructed only when the
    db is ALREADY at the current version — which makes the migration a
    provable no-op rather than a hoped-for one.

    Otherwise this returns None and `summarize_strategies` recomputes the same
    ratio from the rows. A None is not fatal, but it is worth a warning —
    silently falling back is how the two definitions drift apart unnoticed.
    """
    try:
        from src.paper_manager import _SCHEMA_VERSION, PaperManager
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as probe:
            on_disk = probe.execute("PRAGMA user_version").fetchone()[0]
        if on_disk != _SCHEMA_VERSION:
            # An empty ledger has no strategies to break down, so nothing is
            # being skipped and the two definitions cannot drift. Warning here
            # greets every fresh clone with `schema v0, not v17` before
            # anything has happened, and a warning that fires when nothing is
            # wrong is what makes the real one easy to miss.
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as probe:
                try:
                    has_rows = probe.execute(
                        "SELECT 1 FROM trades LIMIT 1").fetchone() is not None
                except sqlite3.Error:
                    has_rows = False
            if has_rows:
                print(f"warning: {db_path} is at schema v{on_disk}, not "
                      f"v{_SCHEMA_VERSION}; strategy breakdown skipped rather "
                      "than migrating it (recomputing return on risk from rows)",
                      file=sys.stderr)
            return None
        return PaperManager(db_path=db_path).get_strategy_breakdown()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"warning: strategy breakdown unavailable ({exc}); "
              "recomputing return on risk from rows", file=sys.stderr)
        return None


def publish(db_path: str = "paper_trades.db", reports_dir: str = "reports",
            config_path: str = "config.json") -> Optional[str]:
    """Read the db, render, and write reports/TRACK_RECORD.md. Returns the path.

    Commit the result — see 'Publish flow' in the module docstring.
    """
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        rows = fetch_closed_trades(conn)
    finally:
        conn.close()
    evidence = load_model_evidence(reports_dir)
    md = render_track_record(
        rows, evidence,
        breakdown=_load_breakdown(db_path),
        budget_cap=_load_budget_cap(config_path),
    )
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "TRACK_RECORD.md")
    with open(out_path, "w") as f:
        f.write(md)
    return out_path


def main() -> int:
    path = publish()
    if path:
        print(f"Wrote {path}")
        print("Remember to commit it: git add reports/TRACK_RECORD.md")
        return 0
    print("paper_trades.db not found; nothing to publish.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
