"""Crypto time-exit age guardrail — blocks the same-cycle open->time-exit race.

A position auto-logged near expiry must not be force-closed by the time-exit
within the first hour of its life. See src/crypto/exit_enforcer.MIN_TIME_EXIT_AGE_S.
"""
from __future__ import annotations

import datetime as dt
import unittest

from src.crypto.exit_enforcer import (
    MIN_TIME_EXIT_AGE_S,
    _position_age_seconds,
    _time_exit_allowed,
)


def _stamp(seconds_ago: float) -> str:
    t = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds_ago)
    return t.strftime("%Y-%m-%d %H:%M:%S")


class AgeGuardTests(unittest.TestCase):
    def test_threshold_is_one_hour(self):
        self.assertEqual(MIN_TIME_EXIT_AGE_S, 3600)

    def test_age_seconds_roughly_correct(self):
        age = _position_age_seconds({"date": _stamp(7200)})
        self.assertAlmostEqual(age, 7200, delta=120)

    def test_young_position_blocks_time_exit(self):
        # opened 5 minutes ago — the exact race observed in the ledger
        self.assertFalse(_time_exit_allowed({"date": _stamp(300)}))

    def test_just_under_threshold_blocks(self):
        self.assertFalse(_time_exit_allowed({"date": _stamp(MIN_TIME_EXIT_AGE_S - 60)}))

    def test_old_position_allows_time_exit(self):
        self.assertTrue(_time_exit_allowed({"date": _stamp(MIN_TIME_EXIT_AGE_S + 60)}))

    def test_unparseable_or_missing_fails_open(self):
        # never strand a legacy/corrupt row OPEN forever
        self.assertTrue(_time_exit_allowed({"date": "not-a-date"}))
        self.assertTrue(_time_exit_allowed({"date": None}))
        self.assertEqual(_position_age_seconds({}), float("inf"))

    def test_utc_suffix_tolerated(self):
        age = _position_age_seconds({"date": _stamp(7200) + " UTC"})
        self.assertAlmostEqual(age, 7200, delta=120)

    def test_date_only_stamp_parses(self):
        # a bare YYYY-MM-DD entry (legacy) is treated as midnight UTC, not inf
        self.assertNotEqual(_position_age_seconds({"date": "2020-01-01"}), float("inf"))


if __name__ == "__main__":
    unittest.main()
