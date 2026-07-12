import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src.maintenance import due_morning_briefing


class TestDueMorningBriefing(unittest.TestCase):
    def test_due_on_fresh_business_day(self):
        self.assertTrue(due_morning_briefing({}, "2026-07-10", 5))          # Friday
        self.assertTrue(due_morning_briefing(
            {"last_morning_briefing": "2026-07-09"}, "2026-07-10", 5))

    def test_not_due_twice_same_day(self):
        self.assertFalse(due_morning_briefing(
            {"last_morning_briefing": "2026-07-10"}, "2026-07-10", 5))

    def test_never_due_weekend(self):
        self.assertFalse(due_morning_briefing({}, "2026-07-11", 6))         # Saturday
        self.assertFalse(due_morning_briefing({}, "2026-07-12", 7))         # Sunday


if __name__ == "__main__":
    unittest.main()
