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
