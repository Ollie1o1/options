"""The reprice report: what the surface does to the closed book, in two tiers.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_spread_surface_report -v
"""
from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest

from src.execution_costs import CostModel
from src.spread_surface import (Cell, SpreadSurface, cell_key, save_surface,
                                REFIT_COMMAND, _cli_main)
from src.spread_surface_report import (TierRow, UnpricedRow, classify_tiers,
                                       render_report)


def _flat_cost_model(default: float = 0.05) -> CostModel:
    """A CostModel with no per-strategy table, so half_spread(strategy)
    always returns the flat default. Deterministic, and touches no real
    database — classify_tiers's own default (load_measured_model) reads
    data/chain_archive.db and paper_trades.db, which tests must never name."""
    return CostModel(table={}, default_half_spread=default)


def _ledger(path, trades):
    """trades: (entry_id, ticker, strike, expiration, date, status,
    entry_delta, strategy_name, net_credit, entry_price)

    entry_price is an explicit fixture field, not a hardcoded constant: for a
    multi-leg row entry_price is a NET CREDIT (differs from any leg mid), and
    the tier-1 pricing test below needs entry_price and the archived quote's
    mid to differ on purpose."""
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
            "entry_price, pnl_pct) VALUES (?,?,?,?,?,'call',?,?,?,?,?,0.1)",
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
                            "CLOSED", 0.5, "Bull Put", 1.0, 1.0)])
        _archive(self.arc, [("AAPL", "2026-06-10", 100.0, "2026-07-10",
                             0.90, 1.10, 0.5, 500)])
        out = classify_tiers(self.led, self.arc, cost_model=_flat_cost_model())
        self.assertEqual([r.entry_id for r in out["tier1"]], [1])
        self.assertEqual(out["tier2"], [])

    def test_a_single_leg_trade_without_a_quote_but_with_delta_is_tier_2(self):
        # net_credit is None: a single leg, so entry_price really is a mid.
        _ledger(self.led, [(2, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.5, "Bull Put", None, 1.0)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc, cost_model=_flat_cost_model())
        self.assertEqual(out["tier1"], [])
        self.assertEqual([r.entry_id for r in out["tier2"]], [2])
        self.assertEqual(out["no_leg_mid"], [])

    def test_a_multileg_trade_without_a_quote_lands_in_no_leg_mid(self):
        # net_credit is set: a multi-leg structure. entry_price is a net
        # credit, not a leg mid, and there is no archived quote either — no
        # leg mid exists anywhere in the ledger, so this must not be priced.
        _ledger(self.led, [(9, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.5, "Bull Put", 2.5, 2.5)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc, cost_model=_flat_cost_model())
        self.assertEqual(out["tier1"], [])
        self.assertEqual(out["tier2"], [])
        self.assertEqual(out["uncovered"], [])
        self.assertEqual(
            [(r.entry_id, r.strategy) for r in out["no_leg_mid"]],
            [(9, "Bull Put")])

    def test_the_buckets_are_disjoint_and_exhaustive(self):
        # One trade per bucket: tier1 (archived quote), tier2 (single leg,
        # no quote), no_leg_mid (multi leg, no quote), uncovered (no delta).
        _ledger(self.led, [
            (1, "AAPL", 100.0, "2026-07-10", "2026-06-10", "CLOSED", 0.5,
             "Bull Put", 1.0, 1.0),
            (2, "ZZZZ", 100.0, "2026-07-10", "2026-06-10", "CLOSED", 0.5,
             "Long Call", None, 1.0),
            (9, "ZZZZ", 100.0, "2026-07-10", "2026-06-10", "CLOSED", 0.5,
             "Bull Put", 2.5, 2.5),
            (3, "ZZZZ", 100.0, "2026-07-10", "2026-06-10", "CLOSED", None,
             "Bull Put", 1.0, 1.0),
        ])
        _archive(self.arc, [("AAPL", "2026-06-10", 100.0, "2026-07-10",
                             0.90, 1.10, 0.5, 500)])
        out = classify_tiers(self.led, self.arc, cost_model=_flat_cost_model())
        ids = {
            "tier1": {r.entry_id for r in out["tier1"]},
            "tier2": {r.entry_id for r in out["tier2"]},
            "no_leg_mid": {r.entry_id for r in out["no_leg_mid"]},
            "uncovered": set(out["uncovered"]),
        }
        self.assertEqual(ids["tier1"], {1})
        self.assertEqual(ids["tier2"], {2})
        self.assertEqual(ids["no_leg_mid"], {9})
        self.assertEqual(ids["uncovered"], {3})
        all_ids = ids["tier1"] | ids["tier2"] | ids["no_leg_mid"] | ids["uncovered"]
        self.assertEqual(len(all_ids), sum(len(v) for v in ids.values()))
        self.assertEqual(len(all_ids), 4)

    def test_tier_1_prices_off_the_archived_quotes_mid_not_entry_price(self):
        # entry_price (5.0, a net credit for this multi-leg row) deliberately
        # differs from the archived quote's own mid ((0.90+1.10)/2 = 1.00).
        # Multiplying a relative half-spread by entry_price instead of the
        # quote's mid would understate/overstate friction by 5x here.
        _ledger(self.led, [(1, "AAPL", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.5, "Bull Put", 5.0, 5.0)])
        _archive(self.arc, [("AAPL", "2026-06-10", 100.0, "2026-07-10",
                             0.90, 1.10, 0.5, 500)])
        # dte = julianday(2026-07-10) - julianday(2026-06-10) = 30 ->
        # DTE_EDGES=(7,21,45,90) bucket index 2. abs_delta=0.5 ->
        # DELTA_EDGES=(0.10,0.25,0.40,0.60) bucket index 2. oi=500 ->
        # OI_EDGES=(10,100,1000,10000) bucket index 2.
        key = cell_key(abs_delta=0.5, dte=30.0, open_interest=500.0)
        surface = SpreadSurface(
            {key: Cell(n=100, rel_half_spread=0.20, median_depth=50)}, {})
        out = classify_tiers(self.led, self.arc, surface=surface,
                             cost_model=_flat_cost_model())
        self.assertEqual(len(out["tier1"]), 1)
        row = out["tier1"][0]
        # rel(0.20) * archived quote mid(1.00) = 0.20, NOT rel * entry_price
        # (5.0) = 1.00.
        self.assertAlmostEqual(row.new_friction, 0.20, places=6)

    def test_tier_2_carries_a_central_estimate_below_its_conservative_bound(
            self):
        # Wiring-level version of the SpreadSurface-level test in
        # test_spread_surface.py: classify_tiers must actually call
        # oi_collapsed_relative for the central figure and
        # relative(open_interest=None) for the conservative one, not both
        # off the same number.
        _ledger(self.led, [(2, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.50, "Bull Put", None, 1.0)])
        _archive(self.arc, [])
        # dte = 30 -> bucket 2; delta 0.50 -> bucket 2. Three OI buckets
        # populated with different values so the OI-collapsed median (0.02)
        # differs from the bucket-0 (illiquid) pin (0.05).
        surface = SpreadSurface({
            cell_key(abs_delta=0.50, dte=30.0, open_interest=5.0):
                Cell(n=40, rel_half_spread=0.05, median_depth=10),
            cell_key(abs_delta=0.50, dte=30.0, open_interest=50.0):
                Cell(n=40, rel_half_spread=0.02, median_depth=10),
            cell_key(abs_delta=0.50, dte=30.0, open_interest=500.0):
                Cell(n=40, rel_half_spread=0.01, median_depth=10),
        }, {})
        out = classify_tiers(self.led, self.arc, surface=surface,
                             cost_model=_flat_cost_model())
        self.assertEqual(len(out["tier2"]), 1)
        row = out["tier2"][0]
        self.assertEqual(row.provenance, "oi_collapsed")
        self.assertEqual(row.conservative_provenance, "cell")
        self.assertIsNotNone(row.conservative_friction)
        assert row.conservative_friction is not None  # narrow for mypy
        self.assertLess(row.new_friction, row.conservative_friction)
        # mid = entry_price = 1.0, so friction == relative directly.
        self.assertAlmostEqual(row.new_friction, 0.02, places=6)
        self.assertAlmostEqual(row.conservative_friction, 0.05, places=6)

    def test_a_trade_with_neither_is_reported_uncovered_not_dropped(self):
        _ledger(self.led, [(3, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", None, "Bull Put", 1.0, 1.0)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc, cost_model=_flat_cost_model())
        self.assertEqual([r for r in out["uncovered"]], [3])

    def test_open_trades_are_excluded(self):
        _ledger(self.led, [(4, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "OPEN", 0.5, "Bull Put", 1.0, 1.0)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc, cost_model=_flat_cost_model())
        self.assertEqual(out["tier1"] + out["tier2"], [])
        self.assertEqual(out["no_leg_mid"], [])
        self.assertEqual(out["uncovered"], [])


class RenderTest(unittest.TestCase):
    """Render it and assert on the output. A source grep is not a rendering
    test."""

    def _rows(self):
        return {
            "tier1": [TierRow(1, "Bull Put", 1, 0.162, 0.240, "cell")],
            # central (0.130) strictly below conservative (0.200), matching
            # what oi_collapsed_relative vs relative(open_interest=None)
            # actually produce when OI buckets differ.
            "tier2": [TierRow(2, "Bull Put", 2, 0.162, 0.130, "oi_collapsed",
                              0.200, "cell")],
            "no_leg_mid": [UnpricedRow(9, "Bull Put")],
            "uncovered": [3],
        }

    def _surface(self):
        # A single populated cell is enough to take render_report out of the
        # empty-surface refusal path; the fixture rows above already carry
        # their own friction figures and provenance independent of this.
        return SpreadSurface(
            {cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)},
            {"fit_date": "2026-08-28"})

    def test_both_tiers_appear_with_their_counts(self):
        out = render_report(self._rows(), self._surface())
        self.assertIn("Tier 1", out)
        self.assertIn("Tier 2", out)

    def test_no_leg_mid_bucket_is_stated_and_not_priced(self):
        out = render_report(self._rows(), self._surface())
        # Anchored to the no_leg_mid block's OWN count, not just "n=1"
        # appearing anywhere in the output — Tier 1 also renders "(n=1)" for
        # an unrelated reason, so a bare "n=1" substring check passes
        # regardless of what no_leg_mid actually contains.
        self.assertIn(
            "No leg mid — multi-leg net credit, no archived quote  (n=1)",
            out)
        self.assertIn("not computed", out.lower())

    def test_tier_2_states_open_interest_is_unknown_not_a_lower_bound(self):
        # Tier 2's original defect: relative(..., open_interest=None) pins
        # to the most illiquid (highest-cost) bucket, yet was labelled a
        # LOWER bound. It is neither a lower bound nor the only number —
        # open interest is unknown, so the report must say that and show
        # both a central estimate and a conservative bound instead.
        out = render_report(self._rows(), self._surface())
        self.assertNotIn("lower bound", out.lower())
        self.assertIn("open interest is unknown", out.lower())
        self.assertIn("central", out.lower())
        self.assertIn("conserv", out.lower())

    def test_tier_2_renders_both_the_central_and_conservative_figures(self):
        out = render_report(self._rows(), self._surface())
        self.assertIn("0.130", out)
        self.assertIn("0.200", out)

    def test_uncovered_trades_are_stated_not_hidden(self):
        rows = dict(self._rows(), uncovered=[3, 7, 11])
        out = render_report(rows, self._surface())
        self.assertIn("uncovered: 3 closed trades", out)

    def test_the_stamp_is_shown_so_a_stale_surface_is_visible(self):
        out = render_report(self._rows(), self._surface())
        self.assertIn("2026-08-28", out)

    def test_no_confidence_interval_is_reported(self):
        # Quotes cluster by symbol and date; a row count is not an observation
        # count. Printing a CI here would be the count-clusters defect again.
        out = render_report(self._rows(), self._surface()).lower()
        for banned in ("ci ", "confidence interval", "95%"):
            self.assertNotIn(banned, out)

    def test_states_no_book_wide_total_is_offered_because_of_unpriced_trades(
            self):
        # no_leg_mid is a real fraction of the closed book (29% in the real
        # ledger) and is entirely unpriced. Any total a reader assembles from
        # tier1 + tier2 alone would undercount, so the report must say so
        # rather than let a reader compute a silent, wrong total.
        out = render_report(self._rows(), self._surface()).lower()
        self.assertIn("no book-wide total", out)
        self.assertIn("unpriced", out)
        # 1 of 4 fixture trades is no_leg_mid -> 25%.
        self.assertIn("25%", out)

    def test_units_are_stated_in_the_output(self):
        out = render_report(self._rows(), self._surface())
        self.assertIn("$/share", out)

    def test_provenance_counts_are_rendered_per_block(self):
        # The report must surface which rows are real measured cells versus
        # a fallback rung, not just a bare "surface fit unknown" hint on an
        # otherwise ordinary-looking line.
        out = render_report(self._rows(), self._surface())
        self.assertIn("provenance: cell=1", out)
        self.assertIn("central provenance: oi_collapsed=1", out)
        self.assertIn("conservative provenance: cell=1", out)

    def test_an_unfitted_surface_refuses_to_render_measurements(self):
        # data/ is gitignored, so a fresh clone's spread_surface.json is
        # absent and load_surface() returns an EMPTY surface. Every figure
        # the tables would show is then a caller_default fallback — nothing
        # on the page would distinguish it from a real measurement, so
        # render_report must refuse outright rather than print fiction that
        # reads as fact.
        empty = SpreadSurface({}, {})
        out = render_report(self._rows(), empty)
        self.assertIn("REFUSING", out)
        self.assertIn(REFIT_COMMAND, out)
        self.assertNotIn("Tier 1", out)
        self.assertNotIn("Tier 2", out)
        self.assertNotIn("baseline", out.lower())


class CliReportTest(unittest.TestCase):
    def test_report_flag_renders_without_touching_the_real_ledger(self):
        d = tempfile.mkdtemp()
        led, arc = os.path.join(d, "l.db"), os.path.join(d, "a.db")
        surf = os.path.join(d, "s.json")
        _ledger(led, [(1, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                       "CLOSED", 0.5, "Bull Put", None, 1.0)])
        _archive(arc, [])
        # --surface is pinned explicitly. Omitting it reads whatever
        # data/spread_surface.json happens to exist on the machine running
        # the test (or nothing, on a fresh clone) — a different code path
        # per machine, and one that would never hit the empty-surface
        # refusal path this suite covers elsewhere.
        save_surface(
            SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)},
                          {"fit_date": "2026-08-28"}),
            surf)
        argv = sys.argv
        sys.argv = ["prog", "--report", "--ledger", led, "--archive", arc,
                    "--surface", surf]
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _cli_main()
        finally:
            sys.argv = argv
        out = buf.getvalue()
        self.assertIn("REPRICE REPORT", out)
        self.assertIn("Tier 2", out)


class BaselineMeasuredTest(unittest.TestCase):
    """The "baseline" column must not be described as a measurement for a
    strategy execution_costs.py itself could not measure. A table entry
    existing is not the same as it being used: `half_spread_for` falls back
    to the model's own $0.05 default below MIN_OBSERVATIONS even when the
    strategy is present in the table."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.led = os.path.join(self.dir, "ledger.db")
        self.arc = os.path.join(self.dir, "archive.db")

    def _thin_cost_model(self):
        # "Long Put" has a table entry (0.10) but only 6 matched quotes —
        # below MIN_OBSERVATIONS (10) — so half_spread_for returns the
        # $0.05 default, not the table's 0.10.
        return CostModel(
            table={"Long Put": {"n": 6, "median_half_spread": 0.10}},
            default_half_spread=0.05)

    def test_classify_tiers_marks_the_row_as_not_measured(self):
        _ledger(self.led, [(1, "ZZZZ", 100.0, "2026-07-10", "2026-06-10",
                            "CLOSED", 0.5, "Long Put", None, 1.0)])
        _archive(self.arc, [])
        out = classify_tiers(self.led, self.arc,
                             cost_model=self._thin_cost_model())
        self.assertEqual(len(out["tier2"]), 1)
        row = out["tier2"][0]
        # The $0.05 default was used, NOT the table's 0.10 — n=6 is below
        # MIN_OBSERVATIONS.
        self.assertAlmostEqual(row.baseline_friction, 0.05, places=6)
        self.assertFalse(row.baseline_measured)

    def test_render_report_does_not_describe_the_default_as_measured(self):
        # Fails against the wording in place before this fix: the tier 2
        # note said unconditionally "the measured per-strategy half-spread
        # ... not the historic flat $0.05" while THIS row's baseline is
        # exactly that flat $0.05 — the note denied what the number was.
        surface = SpreadSurface(
            {cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)},
            {"fit_date": "2026-08-28"})
        rows = {
            "tier1": [],
            "tier2": [TierRow(1, "Long Put", 2, 0.05, 0.03, "cell", 0.06,
                              "cell", baseline_measured=False)],
            "no_leg_mid": [],
            "uncovered": [],
        }
        out = render_report(rows, surface)
        self.assertIn("*", out)
        self.assertIn("not a measurement", out.lower())
