"""The spread board and the single-leg board run as separate scheduled
processes (`scripts/auto_log_equity.sh`: `-ds` then `-sps`, never together),
so each used to draw its own full `--log-top` regardless of what the OTHER
board had already logged that day. `strategy_allocation.allocate`'s target
share (Bull Put ~89%, everything else ~4% each) could then only ever govern
the mix WITHIN one board, never the split BETWEEN them — found 2026-09-08 via
`src.maintenance_health`'s "alloc drift" check: last 30 eligible entries ran
47% Bull Put against an 89% target, Long Call and Long Put each 27% against a
4% target.

`_family_top_n` fixes this by reading what each family has already logged
TODAY straight out of the ledger (the only state visible across the two
separate processes) and handing back only the slots this family still needs
to reach its allocation share of (today's total so far + this scan's own
ceiling).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_family_top_n -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

from src.options_screener import (
    _SPREAD_FAMILY, _SINGLE_LEG_FAMILY, _family_top_n, _today_family_counts,
)
from src.strategy_allocation import Allocation


def _alloc(weights):
    return Allocation(weights=weights, posterior={}, n_eff={}, explore_rate=0.15,
                      as_of="2026-09-08")


class TodayFamilyCountsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "t.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, paper_only INTEGER)")
        rows = [
            ("2026-09-08", "Bull Put", 0),
            ("2026-09-08", "Bull Put", 0),
            ("2026-09-08", "Long Call", 0),
            ("2026-09-08", "Long Put", 0),
            ("2026-09-08", "Long Put", 0),
            ("2026-09-08", "Long Put", 0),
            ("2026-09-08", "Bear Call", 1),   # paper_only — excluded
            ("2026-09-07", "Bull Put", 0),    # yesterday — excluded
        ]
        conn.executemany(
            "INSERT INTO trades (date, strategy_name, paper_only) VALUES (?, ?, ?)",
            rows)
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_counts_split_by_family_today_eligible_only(self):
        counts = _today_family_counts(self.db_path, "2026-09-08")
        self.assertEqual(counts[_SPREAD_FAMILY], 2)
        self.assertEqual(counts[_SINGLE_LEG_FAMILY], 4)

    def test_unreadable_ledger_fails_to_zero_not_an_exception(self):
        counts = _today_family_counts(
            os.path.join(self.tmpdir, "does_not_exist.db"), "2026-09-08")
        self.assertEqual(counts[_SPREAD_FAMILY], 0)
        self.assertEqual(counts[_SINGLE_LEG_FAMILY], 0)


class FamilyTopNTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.tmpdir, "config.json")
        self.db_path = os.path.join(self.tmpdir, "paper_trades.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, paper_only INTEGER)")
        conn.commit()
        conn.close()
        cfg = {"auto_log": {"allocation": {"enabled": True,
                                           "ledger_path": self.db_path}}}
        with open(self.cfg_path, "w") as f:
            json.dump(cfg, f)
        self.today = datetime.now().strftime("%Y-%m-%d")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _insert(self, strategy_name, n, date=None):
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT INTO trades (date, strategy_name, paper_only) VALUES (?, ?, 0)",
            [(date or self.today, strategy_name)] * n)
        conn.commit()
        conn.close()

    def test_no_allocation_falls_back_to_the_unchanged_log_top(self):
        """Allocation off (None) or empty weights must not change behaviour."""
        with patch("src.options_screener._current_allocation", return_value=None):
            self.assertEqual(
                _family_top_n(_SPREAD_FAMILY, _SINGLE_LEG_FAMILY, 5,
                              cfg_path=self.cfg_path), 5)
        with patch("src.options_screener._current_allocation",
                   return_value=_alloc({})):
            self.assertEqual(
                _family_top_n(_SPREAD_FAMILY, _SINGLE_LEG_FAMILY, 5,
                              cfg_path=self.cfg_path), 5)

    def test_family_already_over_its_share_today_gets_zero(self):
        """Single-leg family target is 11%; 4 single-leg entries already
        logged today against 0 spread entries is already over its share of
        (4 + log_top) — the fix is that this can legitimately be 0."""
        alloc = _alloc({"Bull Put": 0.89, "Long Call": 0.04,
                        "Long Put": 0.04, "Short Put": 0.03})
        self._insert("Long Put", 4)
        with patch("src.options_screener._current_allocation",
                   return_value=alloc):
            n = _family_top_n(_SINGLE_LEG_FAMILY, _SPREAD_FAMILY, 5,
                              cfg_path=self.cfg_path)
        self.assertEqual(n, 0)

    def test_family_under_its_share_keeps_nearly_the_full_ceiling(self):
        """Nothing logged yet today: Bull Put deserves round(0.89 * 5) = 4 of
        the 5 slots — the ceiling only binds a family whose own share of
        (so-far + log_top) meets or exceeds log_top."""
        alloc = _alloc({"Bull Put": 0.89, "Long Call": 0.04,
                        "Long Put": 0.04, "Short Put": 0.03})
        with patch("src.options_screener._current_allocation",
                   return_value=alloc):
            n = _family_top_n(_SPREAD_FAMILY, _SINGLE_LEG_FAMILY, 5,
                              cfg_path=self.cfg_path)
        self.assertEqual(n, 4)

    def test_zero_weight_family_gets_zero_without_touching_the_ledger(self):
        alloc = _alloc({"Bull Put": 1.0})
        with patch("src.options_screener._current_allocation",
                   return_value=alloc):
            n = _family_top_n(_SINGLE_LEG_FAMILY, _SPREAD_FAMILY, 5,
                              cfg_path=self.cfg_path)
        self.assertEqual(n, 0)

    def test_split_is_proportional_partway_through_the_day(self):
        """3 Bull Put already logged today, 0 single-leg. Target ~89%/11% of
        (3 + 5) = 8 total -> Bull Put deserves ~7.1 (round 7), already has 3,
        so may still take min(5, 4) = 4 more this scan."""
        alloc = _alloc({"Bull Put": 0.89, "Long Call": 0.04,
                        "Long Put": 0.04, "Short Put": 0.03})
        self._insert("Bull Put", 3)
        with patch("src.options_screener._current_allocation",
                   return_value=alloc):
            n = _family_top_n(_SPREAD_FAMILY, _SINGLE_LEG_FAMILY, 5,
                              cfg_path=self.cfg_path)
        self.assertEqual(n, 4)


if __name__ == "__main__":
    unittest.main()
