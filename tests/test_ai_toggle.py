"""Tests for the config-level AI scoring toggle.

The AI ranking pipeline was controllable only via the per-run --no-ai flag, so
interactive runs (where the flag can't be passed) always ran it. A config flag
makes "no AI" a persistent setting that behaves like --no-ai everywhere.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_ai_toggle -v
"""
from __future__ import annotations

import unittest

from src.options_screener import ai_scoring_disabled


class AiToggleTest(unittest.TestCase):
    def test_enabled_by_default_when_absent(self):
        self.assertFalse(ai_scoring_disabled({}))
        self.assertFalse(ai_scoring_disabled(None))

    def test_disabled_when_flag_false(self):
        self.assertTrue(ai_scoring_disabled({"ai_scoring": {"enabled": False}}))

    def test_enabled_when_flag_true(self):
        self.assertFalse(ai_scoring_disabled({"ai_scoring": {"enabled": True}}))

    def test_never_raises_on_garbage(self):
        self.assertIn(ai_scoring_disabled({"ai_scoring": "nope"}), (True, False))


if __name__ == "__main__":
    unittest.main()
