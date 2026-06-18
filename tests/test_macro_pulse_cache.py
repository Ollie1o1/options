"""Tests for src/macro_pulse/cache.py — offline."""
from __future__ import annotations

import os
import tempfile
import unittest

from src.macro_pulse import cache as K


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "c.db")

    def test_key_stable_regardless_of_order(self):
        a = K.bundle_key(["b", "a", "c"], "2026-06-18")
        b = K.bundle_key(["c", "b", "a"], "2026-06-18")
        self.assertEqual(a, b)

    def test_key_changes_with_day(self):
        a = K.bundle_key(["a"], "2026-06-18")
        b = K.bundle_key(["a"], "2026-06-19")
        self.assertNotEqual(a, b)

    def test_miss_then_hit(self):
        key = K.bundle_key(["a"], "2026-06-18")
        self.assertIsNone(K.get(key, self.db))
        K.put(key, {"headline": "x"}, self.db)
        self.assertEqual(K.get(key, self.db)["headline"], "x")

    def test_different_day_is_a_miss(self):
        K.put(K.bundle_key(["a"], "2026-06-18"), {"headline": "x"}, self.db)
        self.assertIsNone(K.get(K.bundle_key(["a"], "2026-06-19"), self.db))


if __name__ == "__main__":
    unittest.main()
