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
    try:
        cur.execute(
            f"SELECT {', '.join(cols)} FROM trades "
            "WHERE UPPER(status) = 'CLOSED' ORDER BY date ASC"
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
    ]
    # Further notes append here (one `_note_*` helper each).
    return [n for n in notes if n]


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
            print(f"warning: {db_path} is at schema v{on_disk}, not "
                  f"v{_SCHEMA_VERSION}; strategy breakdown skipped rather than "
                  "migrating it (recomputing return on risk from rows)",
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
