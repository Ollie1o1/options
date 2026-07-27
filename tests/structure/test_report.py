import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src import formatting as fmt
from src.structure import report as R
from src.structure.types import Expression, Rejection, StructureMargin, View


class TestReport(unittest.TestCase):
    def setUp(self):
        # Pin color off at the module flag - never via env vars.
        self._prev = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._prev

    def test_report_shows_breakeven_vs_realized(self):
        view = View("NVDA", "BEARISH", 0.31, ["momentum -1.2z"])
        exprs = [Expression("Long Put", 0.122, 0.229, 0.351, 340.0, 3.3, 1)]
        rej = [Rejection("Long Call", "BENCHED (margin -12.8 pts)")]
        table = {"Long Put": StructureMargin(
            "Long Put", 37, 13, 24, 815.0, 242.0, 0.229, 0.351, 0.122,
            "ACTIVE", 0.01, 0.25)}
        out = R.render(view, exprs, rej, table, 511.0)
        self.assertIn("Long Put", out)
        self.assertIn("22.9", out)     # breakeven
        self.assertIn("35.1", out)     # realized
        self.assertIn("BENCHED", out)  # rejection reason is shown

    def test_neutral_view_states_suppression(self):
        view = View("SPY", "NEUTRAL", 0.1, [])
        out = R.render(view, [], [], {}, 511.0)
        self.assertIn("NEUTRAL", out)
        self.assertIn("suppress", out.lower())

    def test_empty_table_says_no_evidence(self):
        view = View("SPY", "BULLISH", 0.9, [])
        out = R.render(view, [], [], {}, 511.0)
        self.assertIn("no structure evidence", out.lower())

    def test_ci_including_zero_is_marked_untrusted(self):
        view = View("SPY", "NEUTRAL", 0.1, [])
        table = {"Iron Condor": StructureMargin(
            "Iron Condor", 73, 34, 39, 311.0, 223.0, 0.457, 0.466, 0.008,
            "ACTIVE", -0.10, 0.12)}
        out = R.render(view, [], [], table, 511.0)
        self.assertIn("~", out)

    def test_no_candidates_is_not_reported_as_no_edge(self):
        # A plumbing gap (no contracts supplied) must not be misreported as a
        # verdict that nothing cleared its breakeven.
        view = View("NVDA", "BEARISH", 0.4, [])
        table = {"Long Put": StructureMargin(
            "Long Put", 41, 13, 28, 815.0, 242.0, 0.211, 0.317, 0.106,
            "ACTIVE", 0.01, 0.25)}
        rej = [Rejection("Long Put", "no candidate contract found")]
        out = R.render(view, [], rej, table, 511.0)
        self.assertIn("no candidate contracts supplied", out)
        self.assertNotIn("nothing clears its own breakeven", out)
