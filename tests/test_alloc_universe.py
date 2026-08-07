"""Universe loading and the coverage audit that gates all backtesting.

The audit can veto. BROAD is the control stratum — if it is mostly empty, an
"it generalises" conclusion would be unsupported, so this runs before anything
is backtested.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_universe -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from src import dolt_options as do
from src.alloc import universe as U


class LoadUniverseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "u.json")
        with open(self.path, "w") as f:
            json.dump({"seed": 1, "strata": {
                "legacy": ["SPY"], "liquid": ["ABT"], "broad": ["AWK"]}}, f)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_returns_strata(self):
        u = U.load_universe(self.path)
        self.assertEqual(u["legacy"], ["SPY"])
        self.assertEqual(set(u), {"legacy", "liquid", "broad"})

    def test_symbol_stratum_inverts_the_mapping(self):
        m = U.symbol_stratum(U.load_universe(self.path))
        self.assertEqual(m["SPY"], "legacy")
        self.assertEqual(m["AWK"], "broad")

    def test_all_symbols_flattens_every_stratum(self):
        self.assertEqual(set(U.all_symbols(U.load_universe(self.path))),
                         {"SPY", "ABT", "AWK"})


class CoverageAuditTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "dolt.db")
        do._ensure_cache(self.db)
        self.uni = {"legacy": ["SPY"], "liquid": ["ABT"], "broad": ["GONE"]}
        for i in range(1, 6):
            d = f"2024-01-0{i}"
            do._cache_write(self.db, "SPY", d, [self._c()])
            do._cache_write(self.db, "ABT", d, [self._c()])
            do._cache_write(self.db, "GONE", d, [])

    def tearDown(self):
        self._tmp.cleanup()

    def _c(self):
        return {"expiration": "2024-02-16", "strike": 100.0, "type": "put",
                "bid": 1.0, "ask": 1.2, "mid": 1.1, "iv": 0.2,
                "delta": -0.3, "gamma": 0.01, "theta": -0.05,
                "vega": 0.1, "rho": 0.01}

    def test_absent_symbol_is_flagged(self):
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        states = {d["symbol"]: d["state"] for d in a["detail"]}
        self.assertEqual(states["GONE"], "ABSENT")

    def test_sparse_symbol_is_flagged(self):
        a = U.audit_coverage(self.db, self.uni, min_usable_days=99)
        states = {d["symbol"]: d["state"] for d in a["detail"]}
        self.assertEqual(states["SPY"], "SPARSE")

    def test_usable_symbol_is_ok(self):
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        states = {d["symbol"]: d["state"] for d in a["detail"]}
        self.assertEqual(states["SPY"], "ok")

    def test_usable_symbols_excludes_absent(self):
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        self.assertNotIn("GONE", U.usable_symbols(a))
        self.assertIn("SPY", U.usable_symbols(a))

    def test_dead_date_detected(self):
        """A date where every symbol returned nothing is a source gap."""
        for s in ("SPY", "ABT", "GONE"):
            do._cache_write(self.db, s, "2024-02-01", [])
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        self.assertIn("2024-02-01", a["dead_dates"])

    def test_viable_is_false_when_a_stratum_has_nothing(self):
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        self.assertFalse(a["viable"], "broad has no usable symbol here")

    def test_viable_is_true_when_every_stratum_has_one(self):
        for i in range(1, 6):
            do._cache_write(self.db, "GONE", f"2024-01-0{i}", [self._c()])
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        self.assertTrue(a["viable"])

    def test_symbols_outside_the_universe_are_ignored(self):
        do._cache_write(self.db, "NOTOURS", "2024-01-01", [self._c()])
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        self.assertNotIn("NOTOURS", [d["symbol"] for d in a["detail"]])

    def test_terminal_date_is_reported_for_each_symbol(self):
        """Tickers end mid-sample (FB, PBCT, WLTW). The engine needs the date."""
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        spy = [d for d in a["detail"] if d["symbol"] == "SPY"][0]
        self.assertEqual(spy["last"], "2024-01-05")
        self.assertEqual(spy["first"], "2024-01-01")

    def test_usable_dates_drops_dates_with_no_data(self):
        """A date with no data is MISSING, not 'no opportunity'.

        Scoring an empty date as a no-trade day would bias any always-on
        benchmark upward for free.
        """
        for s in ("SPY", "ABT", "GONE"):
            do._cache_write(self.db, s, "2024-02-01", [])
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        got = U.usable_dates(a, ["2024-01-02", "2024-02-01", "2024-01-03"])
        self.assertEqual(got, ["2024-01-02", "2024-01-03"])

    def test_usable_dates_keeps_unknown_dates(self):
        """A date never fetched is not known-dead; the caller decides."""
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        self.assertIn("2030-01-01", U.usable_dates(a, ["2030-01-01"]))

    def test_terminal_dates_maps_symbol_to_last_day(self):
        a = U.audit_coverage(self.db, self.uni, min_usable_days=1)
        self.assertEqual(U.terminal_dates(a)["SPY"], "2024-01-05")

    def test_min_usable_days_default_branch(self):
        """Called with no min_usable_days — the default must work."""
        a = U.audit_coverage(self.db, self.uni)
        self.assertIn("summary", a)

    def test_missing_database_does_not_raise(self):
        a = U.audit_coverage(os.path.join(self._tmp.name, "nope.db"), self.uni)
        self.assertFalse(a["viable"])


if __name__ == "__main__":
    unittest.main()
