"""Cohort filters shared by every query that draws evidence from the ledger.

One question — "is this row evidence?" — must have exactly one answer, and the
answer has to be importable from anywhere without dragging pandas in. That is
the whole reason this module exists rather than the filter living in
`paper_manager` (which hard-imports pandas/numpy) beside the migration that
adds the column: `backtester` treats pandas as optional, so importing the
ledger module there would turn an optional dependency into a required one.

Nothing here touches the filesystem or mutates a row. These are SQL fragments.
"""
from __future__ import annotations

import sqlite3


def exclude_ruled_duplicates(conn: sqlite3.Connection) -> str:
    """SQL fragment dropping rows ruled double-logs, or '' on older ledgers.

    A ruled duplicate is one real decision recorded twice. Counting it twice
    inflates both n and the evidence drawn from it, so it stays in the ledger —
    which records what happened — and out of every cohort.

    This filter previously lived in `phase1_checkpoint`, which meant the two
    gate cohorts honoured the ruling while `backtester.run_paper_trade_ic` and
    `walk_forward.load_trades` — the pooled IC, and the OOS IC printed in the
    evidence banner on every scan — did not. One ledger, two answers, and the
    difference was invisible because the single ruled row happened to be a
    strategy the walk-forward path filters out anyway.

    `duplicate_of` arrived in schema v17. Probing for it rather than assuming it
    keeps every cohort query working against a ledger written before the
    migration — including the minimal fixtures the tests build by hand.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    except sqlite3.Error:
        return ""
    return " AND duplicate_of IS NULL" if "duplicate_of" in cols else ""


def budget_ceiling_sql(conn: sqlite3.Connection) -> str:
    """SQL for the ceiling a trade's capital at risk must respect.

    Returns a fragment ending in one `?` placeholder, to be bound with the
    CALLER's ceiling:

        "COALESCE(budget_at_entry, ?)"   on a v22+ ledger
        "?"                              on anything older

    `budget_at_entry` is the budget that governed a trade AT ENTRY, and until
    this existed nothing read it — both cohorts compared against the caller's
    single ceiling, so a trade logged under a chosen $10,000 budget dropped out
    of the "inside budget" subset although it was inside its own.

    COALESCE rather than a bare `budget_at_entry` comparison, and the
    difference was measured on the live book before choosing. NULL means "no
    limit was in force" — the pre-2026-07-29 unbounded-feeder era — so reading
    NULL as "inside its budget" would admit the $27k and $83k positions the cap
    exists to exclude wherever they were also small. Requiring the column to be
    non-null instead cut the reported Long Call subset from 130 to 31, because
    101 of the 132 cohort rows predate the cap. Falling back to the caller's
    ceiling reproduces 130 exactly: nothing changes on today's data, and a
    per-trade budget is honoured the moment one differs.

    Probed rather than assumed, so hand-built fixtures and any ledger written
    before schema v22 still read — the same idiom, and the same reason, as
    `exclude_ruled_duplicates` and `duplicate_of`.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    except sqlite3.Error:
        return "?"
    return "COALESCE(budget_at_entry, ?)" if "budget_at_entry" in cols else "?"
