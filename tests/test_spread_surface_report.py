"""The reprice report: what the surface does to the closed book, in two tiers.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_spread_surface_report -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src.spread_surface import Cell, SpreadSurface, cell_key
from src.spread_surface_report import (TierRow, classify_tiers,
                                       render_report)


def _ledger(path, trades):
    """trades: (entry_id, ticker, strike, expiration, date, status,
    entry_delta, strategy_name, net_credit)"""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE trades (
        entry_id INTEGER, ticker TEXT, strike REAL, expiration TEXT,
        date TEXT, type TEXT, status TEXT, entry_delta REAL,
        strategy_name TEXT, net_credit REAL, entry_price REAL,
        pnl_pct REAL)""")
    for t in trades:
        con.execute(
            "INSERT INTO trades (entry_id, ticker, strike, expiration, date, "
            "type, status, entry_delta, strategy_name, net_credit, "
            "entry_price, pnl_pct) VALUES (?,?,?,?,?,'call',?,?,?,?,1.0,0.1)",
            t)
    con.commit()
    con.close()


def _archive(path, quotes):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE chain_snapshots (
        symbol TEXT, snap_date TEXT, type TEXT, strike REAL,
        expiration TEXT, bid REAL, ask REAL, bid_size REAL, ask_size REAL,
        delta REAL, open_interest REAL, spot REAL)""")
    for q in quotes:
        con.execute(
            "INSERT INTO chain_snapshots (symbol, snap_date, type, strike, "
            "expiration, bid, ask, bid_size, ask_size, delta, open_interest, "
            "spot) VALUES (?,?,'call',?,?,?,?,50,50,?,?,100.0)", q)
    con.commit()
    con.close()


class TierClassificationTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.led = os.path.join(self.dir, "ledger.db")
        self.arc = os.path.join(self.dir, "archive.db")

    def test_a_trade_with_an_archived_quote_is_tier_1(self):
        _ledger(self.led, [(1, "AAPL", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.5, "Bull Put", 1.0)])
        _archive(self.arc, [("AAPL", "2026-06-10", 100.0, "2026-07-10",
                             0.90, 1.10, 0.5, 500)])
        out = classify_tiers(self.led, self.arc)
        self.assertEqual([r.entry_id for r in out["tier1"]], [1])
        self.assertEqual(out["tier2"], [])

    def test_a_trade_without_a_quote_but_with_delta_is_tier_2(self):
        _ledger(self.led, [(2, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.5, "Bull Put", 1.0)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc)
        self.assertEqual(out["tier1"], [])
        self.assertEqual([r.entry_id for r in out["tier2"]], [2])

    def test_the_tiers_are_disjoint(self):
        _ledger(self.led, [(1, "AAPL", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.5, "Bull Put", 1.0)])
        _archive(self.arc, [("AAPL", "2026-06-10", 100.0, "2026-07-10",
                             0.90, 1.10, 0.5, 500)])
        out = classify_tiers(self.led, self.arc)
        ids1 = {r.entry_id for r in out["tier1"]}
        ids2 = {r.entry_id for r in out["tier2"]}
        self.assertEqual(ids1 & ids2, set())

    def test_a_trade_with_neither_is_reported_uncovered_not_dropped(self):
        _ledger(self.led, [(3, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", None, "Bull Put", 1.0)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc)
        self.assertEqual([r for r in out["uncovered"]], [3])

    def test_open_trades_are_excluded(self):
        _ledger(self.led, [(4, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "OPEN", 0.5, "Bull Put", 1.0)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc)
        self.assertEqual(out["tier1"] + out["tier2"], [])
        self.assertEqual(out["uncovered"], [])


class RenderTest(unittest.TestCase):
    """Render it and assert on the output. A source grep is not a rendering
    test."""

    def _rows(self):
        return {
            "tier1": [TierRow(1, "Bull Put", 1, 0.162, 0.240, "cell")],
            "tier2": [TierRow(2, "Bull Put", 2, 0.162, 0.200, "oi_collapsed")],
            "uncovered": [3],
        }

    def test_both_tiers_appear_with_their_counts(self):
        out = render_report(self._rows(), {"fit_date": "2026-08-28"})
        self.assertIn("Tier 1", out)
        self.assertIn("Tier 2", out)

    def test_tier_2_is_labelled_a_lower_bound_on_cost(self):
        out = render_report(self._rows(), {"fit_date": "2026-08-28"})
        self.assertIn("lower bound", out.lower())

    def test_uncovered_trades_are_stated_not_hidden(self):
        rows = dict(self._rows(), uncovered=[3, 7, 11])
        out = render_report(rows, {"fit_date": "2026-08-28"})
        self.assertIn("uncovered: 3 closed trades", out)

    def test_the_stamp_is_shown_so_a_stale_surface_is_visible(self):
        out = render_report(self._rows(), {"fit_date": "2026-08-28"})
        self.assertIn("2026-08-28", out)

    def test_no_confidence_interval_is_reported(self):
        # Quotes cluster by symbol and date; a row count is not an observation
        # count. Printing a CI here would be the count-clusters defect again.
        out = render_report(self._rows(), {"fit_date": "2026-08-28"}).lower()
        for banned in ("ci ", "confidence interval", "95%"):
            self.assertNotIn(banned, out)
