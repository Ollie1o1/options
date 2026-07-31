#!/usr/bin/env python3
"""Flag candidate duplicate rows in the paper-trade ledger. READ-ONLY.

The auto-log feeder re-scans the same market and can log the same contract
twice on consecutive days — a catch-up replay after a missed window is the
easy way to get there. Two rows for one decision double-count that decision in
every cohort statistic and every dollar total, so they have to be visible
before anyone reasons off the ledger.

A *candidate* is a group of rows sharing ``(ticker, strategy_name, strike,
expiration)`` with ``entry_price`` equal to the cent, entered within
``--window-days`` of each other (transitively: A-B and B-C chain into one
group even when A and C are further apart). That is a strong signal and not a
proof — the same contract legitimately re-entered days later at exactly the
same price is indistinguishable from a double-log at this level. The operator
rules on each group; this script never touches the data.

READ-ONLY IS ENFORCED, NOT PROMISED: the database is opened through a
``file:...?mode=ro`` URI, so any write would raise rather than land. Nothing in
here issues INSERT/UPDATE/DELETE, and nothing should ever be added that does.

    python scripts/audit_duplicate_trades.py
    python scripts/audit_duplicate_trades.py --window-days 5 --db paper_trades.db
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_WINDOW_DAYS = 3

_COLUMNS = [
    "entry_id", "date", "ticker", "strategy_name", "type", "strike", "expiration",
    "entry_price", "exit_price", "exit_date", "pnl_pct", "pnl_usd", "status",
    "quantity", "paper_only", "weight_profile", "capital_at_risk", "exit_reason",
]


def _parse_day(value: Any) -> Optional[datetime]:
    """Calendar day of a ledger timestamp.

    The ``date`` column holds both bare dates ("2026-07-07") and full
    timestamps ("2026-07-07 14:06:55") depending on which log site wrote the
    row, so only the leading date is parsed and the clock time is discarded:
    two entries hours apart on the same day are the same day.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def fetch_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Every ledger row as a plain dict, oldest first. Read-only."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(trades)")
    present = {r[1] for r in cur.fetchall()}
    cols = [c for c in _COLUMNS if c in present]
    if not cols:
        return []
    cur.execute(f"SELECT {', '.join(cols)} FROM trades ORDER BY date ASC, entry_id ASC")
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _group_key(row: Dict[str, Any]):
    """(ticker, strategy, strike, entry price to the cent, expiration).

    ``strike`` is rounded to four places because it is stored as a REAL and a
    half-strike can come back as 262.49999999999994; ``entry_price`` to two,
    which is the brief's "equal to the cent" and also the only resolution an
    option quote actually has.
    """
    strike = _num(row.get("strike"))
    price = _num(row.get("entry_price"))
    return (
        str(row.get("ticker") or "").upper(),
        str(row.get("strategy_name") or ""),
        round(strike, 4) if strike is not None else None,
        str(row.get("expiration") or ""),
        round(price, 2) if price is not None else None,
    )


def find_candidate_duplicates(rows: Sequence[Dict[str, Any]],
                              window_days: int = DEFAULT_WINDOW_DAYS) -> List[Dict[str, Any]]:
    """Candidate duplicate groups, most recent first.

    Rows with an unparseable date, a missing strike, or a missing entry price
    cannot be placed on the timeline and are excluded rather than guessed at.
    Chaining is on *consecutive* gaps, so a daily re-log across a week comes
    back as one group of five and not ten overlapping pairs.
    """
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _group_key(row)
        if key[2] is None or key[4] is None:
            continue
        day = _parse_day(row.get("date"))
        if day is None:
            continue
        enriched = dict(row)
        enriched["_day"] = day
        buckets[key].append(enriched)

    groups: List[Dict[str, Any]] = []
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: (r["_day"], r.get("entry_id") or 0))
        cluster = [members[0]]
        for row in members[1:]:
            if (row["_day"] - cluster[-1]["_day"]).days <= window_days:
                cluster.append(row)
            else:
                if len(cluster) > 1:
                    groups.append(_summarise(key, cluster))
                cluster = [row]
        if len(cluster) > 1:
            groups.append(_summarise(key, cluster))

    groups.sort(key=lambda g: (g["last_date"], g["ticker"]), reverse=True)
    return groups


def _summarise(key, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    ticker, strategy, strike, expiration, entry_price = key
    closed = [r for r in cluster if str(r.get("status") or "").upper() == "CLOSED"]
    pnl_values = [_num(r.get("pnl_usd")) for r in cluster]
    pnl_total = sum(v for v in pnl_values if v is not None)
    # What a duplicate actually costs the statistics: every row past the first
    # is an extra copy of one decision, so its P&L is double-counted.
    excess_pnl = sum(v for v in pnl_values[1:] if v is not None)
    identical_exits = len({
        (r.get("exit_price"), str(r.get("exit_date") or "")[:10]) for r in closed
    }) == 1 and len(closed) > 1
    return {
        "ticker": ticker,
        "strategy_name": strategy,
        "strike": strike,
        "expiration": expiration,
        "entry_price": entry_price,
        "rows": cluster,
        "size": len(cluster),
        "first_date": cluster[0]["_day"].strftime("%Y-%m-%d"),
        "last_date": cluster[-1]["_day"].strftime("%Y-%m-%d"),
        "span_days": (cluster[-1]["_day"] - cluster[0]["_day"]).days,
        "n_closed": len(closed),
        "n_open": len(cluster) - len(closed),
        "pnl_total": pnl_total,
        "excess_pnl": excess_pnl,
        "identical_exits": identical_exits,
    }


def _fmt_money(v: Optional[float]) -> str:
    return "—" if v is None else f"${v:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100.0:+.1f}%"


def render_report(groups: Sequence[Dict[str, Any]], *, total_rows: int,
                  window_days: int, db_path: str,
                  generated: Optional[str] = None) -> str:
    """The markdown report. Pure — no I/O, so it is testable on seeded rows."""
    generated = generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    excess_rows = sum(g["size"] - 1 for g in groups)
    excess_pnl = sum(g["excess_pnl"] for g in groups)
    flagged_rows = sum(g["size"] for g in groups)

    out: List[str] = [
        "# Duplicate-trade audit — candidate rows in the paper ledger",
        "",
        f"- Generated: {generated}",
        f"- Database: `{db_path}` (opened read-only; this audit never writes)",
        f"- Rows scanned: **{total_rows}**",
        f"- Match key: same `(ticker, strategy_name, strike, expiration)` with "
        f"`entry_price` equal to the cent, entered within **{window_days} days** of each other",
        "",
        "## Summary",
        "",
        f"- Candidate duplicate groups: **{len(groups)}**",
        f"- Rows involved: **{flagged_rows}**",
        f"- Excess rows (every row past the first in its group): **{excess_rows}**",
        f"- P&L carried by those excess rows: **{_fmt_money(excess_pnl)}** "
        "(the amount double-counted if the groups are true duplicates)",
        "",
        "**Nothing has been deleted or edited.** A match is a candidate, not a verdict: "
        "a contract legitimately re-entered at the same price a day later looks identical "
        "at this resolution. The operator rules on each group; only then does anything change.",
        "",
        "A group whose rows also share an exit price and exit date is marked "
        "`identical exits` — that is the strongest tell, because two genuinely separate "
        "positions would have to be closed by the same sweep at the same mark to look that way.",
        "",
    ]

    if not groups:
        out += ["## Candidates", "", "None found. The ledger has no rows matching the key "
                "inside the window.", ""]
        return "\n".join(out) + "\n"

    out += ["## Candidates", ""]
    for i, g in enumerate(groups, 1):
        flag = " — **identical exits**" if g["identical_exits"] else ""
        out += [
            f"### {i}. {g['ticker']} {g['strategy_name']} ${g['strike']:g} exp {g['expiration']} "
            f"@ ${g['entry_price']:.2f}{flag}",
            "",
            f"{g['size']} rows between {g['first_date']} and {g['last_date']} "
            f"(span {g['span_days']}d) — {g['n_closed']} closed, {g['n_open']} open. "
            f"Excess P&L {_fmt_money(g['excess_pnl'])}.",
            "",
            "| entry_id | logged | status | entry | exit | exit date | P&L % | P&L $ | qty | profile |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in g["rows"]:
            out.append(
                f"| {r.get('entry_id')} "
                f"| {str(r.get('date') or '')[:10]} "
                f"| {str(r.get('status') or '—')} "
                f"| {_fmt_money(_num(r.get('entry_price')))} "
                f"| {_fmt_money(_num(r.get('exit_price')))} "
                f"| {str(r.get('exit_date') or '—')[:10]} "
                f"| {_fmt_pct(_num(r.get('pnl_pct')))} "
                f"| {_fmt_money(_num(r.get('pnl_usd')))} "
                f"| {_num(r.get('quantity')) or 1:g} "
                f"| {r.get('weight_profile') or '—'} |"
            )
        out.append("")

    out += [
        "## What to do with this",
        "",
        "1. Rule on each group: true double-log, or a real re-entry that happens to match.",
        "2. True duplicates stay in the ledger until the operator decides otherwise — "
        "the record is what was traded, and rewriting it silently is worse than the "
        "double-count it fixes.",
        "3. The auto-log dedup guard (`auto_log.dedup_window_days`) refuses new entries "
        "matching the same key inside the window, so this list should stop growing from "
        "the automated feeders regardless of how the existing rows are ruled on.",
        "",
    ]
    return "\n".join(out) + "\n"


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open the ledger so writes are impossible, not merely unattempted."""
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit the paper ledger for candidate duplicate rows")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--output", default=os.path.join("reports", "duplicate_trades_audit.md"))
    ap.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                    help=f"days between entries to still count as a candidate (default {DEFAULT_WINDOW_DAYS})")
    ap.add_argument("--stdout", action="store_true", help="print the report instead of writing it")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"No such database: {args.db}", file=sys.stderr)
        return 2

    conn = _connect_readonly(args.db)
    try:
        rows = fetch_rows(conn)
    finally:
        conn.close()

    groups = find_candidate_duplicates(rows, window_days=args.window_days)
    report = render_report(groups, total_rows=len(rows),
                           window_days=args.window_days, db_path=args.db)

    if args.stdout:
        print(report)
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Scanned {len(rows)} rows; {len(groups)} candidate duplicate group(s), "
              f"{sum(g['size'] - 1 for g in groups)} excess row(s).")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
