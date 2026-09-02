"""tests/test_gate_rd_test.py

Pure computation and temp-sqlite-fixture tests only — data/candidates.db is
never opened here.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gate_rd_test import (  # noqa: E402
    CUTOFF,
    relative_spread,
    running_variable,
)


def _bull_put_row(short_bid, short_ask, long_bid, long_ask, **extra):
    blob = {
        "short_bid": short_bid, "short_ask": short_ask,
        "long_bid": long_bid, "long_ask": long_ask,
        "entry_delta": extra.get("entry_delta", -0.20),
    }
    return {
        "strategy_name": "Bull Put",
        "round_trip_pct": extra.get("round_trip_pct"),
        "features_json": json.dumps(blob),
    }


class RunningVariableTests(unittest.TestCase):
    def test_uses_stored_value_when_present(self):
        row = _bull_put_row(1.0, 1.1, 0.4, 0.5, round_trip_pct=0.18)
        rtp, status = running_variable(row)
        self.assertEqual(status, "measured")
        self.assertAlmostEqual(rtp, 0.18)

    def test_recomputes_from_features_json_when_stored_is_null(self):
        # short 1.00/1.10 sell, long 0.40/0.50 buy: mid credit = 1.05-0.45=0.60
        # crossed (sell at bid, buy at ask): 1.00-0.50=0.50; slip=0.10; round trip 2*0.10/0.60=0.333..
        row = _bull_put_row(1.00, 1.10, 0.40, 0.50)
        rtp, status = running_variable(row)
        self.assertEqual(status, "recomputed")
        self.assertGreater(rtp, CUTOFF)  # this one refuses on friction

    def test_credit_gone_is_reported_and_excluded(self):
        # candidate_verdict checks credit_gone against the "limit" fill
        # (35% of half-spread conceded, execution_truth.DEFAULT_LIMIT_K),
        # not a full cross. short 0.50/1.50 (mid=1.00, half=0.50), long
        # 0.50/1.30 (mid=0.90, half=0.40): mid credit = 1.00-0.90 = 0.10
        # (is_credit=True), but limit net_reward = 0.10 - 0.35*(0.50+0.40)
        # = -0.215 <= 0 -> the credit evaporates once actually filled.
        row = _bull_put_row(0.50, 1.50, 0.50, 1.30)
        rtp, status = running_variable(row)
        self.assertEqual(status, "credit_gone")
        self.assertIsNone(rtp)

    def test_unpriceable_when_a_leg_is_unquoted(self):
        blob = {"short_bid": 1.0, "short_ask": 1.1, "entry_delta": -0.2}
        row = {"strategy_name": "Bull Put", "round_trip_pct": None,
              "features_json": json.dumps(blob)}
        rtp, status = running_variable(row)
        self.assertEqual(status, "unpriceable")
        self.assertIsNone(rtp)


class RelativeSpreadTests(unittest.TestCase):
    def test_computed_from_short_leg_quote(self):
        row = _bull_put_row(1.00, 1.10, 0.40, 0.50)
        # short mid = 1.05, spread = 0.10 -> relative = 0.10/1.05
        rs = relative_spread(row)
        self.assertAlmostEqual(rs, 0.10 / 1.05, places=6)

    def test_none_when_short_leg_unquoted(self):
        row = {"strategy_name": "Bull Put",
              "features_json": json.dumps({"long_bid": 0.4, "long_ask": 0.5})}
        self.assertIsNone(relative_spread(row))


if __name__ == "__main__":
    unittest.main()
