"""Tests for src/cross_check.py — Yahoo vs CBOE per-contract agreement (pure)."""
from __future__ import annotations

import unittest

from src.cross_check import compare


def _y(type_, strike, exp, bid, ask, iv):
    return {"type": type_, "strike": strike, "expiration": exp,
            "bid": bid, "ask": ask, "iv": iv}


def _c(type_, strike, exp, bid, ask, iv):
    return {"type": type_, "strike": strike, "expiration": exp,
            "bid": bid, "ask": ask, "iv": iv, "source": "cboe"}


class CompareTest(unittest.TestCase):
    def test_perfect_agreement(self):
        y = [_y("call", 100, "2026-07-17", 5.0, 5.2, 0.30)]
        c = [_c("call", 100, "2026-07-17", 5.0, 5.2, 0.30)]
        r = compare(y, c)
        self.assertEqual(r["matched"], 1)
        self.assertEqual(r["iv_agree"], 1)
        self.assertEqual(r["mid_agree"], 1)
        self.assertEqual(len(r["disagreements"]), 0)

    def test_iv_disagreement_flagged(self):
        y = [_y("call", 100, "2026-07-17", 5.0, 5.2, 0.60)]   # Yahoo IV way off
        c = [_c("call", 100, "2026-07-17", 5.0, 5.2, 0.30)]
        r = compare(y, c)
        self.assertEqual(r["matched"], 1)
        self.assertEqual(r["iv_agree"], 0)
        self.assertEqual(r["mid_agree"], 1)
        self.assertEqual(len(r["disagreements"]), 1)
        self.assertIn("iv", r["disagreements"][0]["why"])

    def test_mid_disagreement_flagged(self):
        y = [_y("put", 95, "2026-07-17", 2.0, 2.2, 0.30)]     # mid 2.10
        c = [_c("put", 95, "2026-07-17", 3.0, 3.2, 0.30)]     # mid 3.10
        r = compare(y, c)
        self.assertEqual(r["mid_agree"], 0)

    def test_small_absolute_mid_diff_tolerated_on_cheap_options(self):
        y = [_y("put", 95, "2026-07-17", 0.10, 0.14, 0.30)]   # mid 0.12
        c = [_c("put", 95, "2026-07-17", 0.12, 0.18, 0.30)]   # mid 0.15: 25% rel but $0.03 abs
        r = compare(y, c)
        self.assertEqual(r["mid_agree"], 1)

    def test_unmatched_contracts_counted_not_compared(self):
        y = [_y("call", 100, "2026-07-17", 5.0, 5.2, 0.30),
             _y("call", 999, "2026-07-17", 0.1, 0.2, 0.90)]   # no CBOE twin
        c = [_c("call", 100, "2026-07-17", 5.0, 5.2, 0.30)]
        r = compare(y, c)
        self.assertEqual(r["matched"], 1)
        self.assertEqual(r["yahoo_only"], 1)

    def test_missing_iv_skips_iv_check_only(self):
        y = [_y("call", 100, "2026-07-17", 5.0, 5.2, None)]
        c = [_c("call", 100, "2026-07-17", 5.0, 5.2, 0.30)]
        r = compare(y, c)
        self.assertEqual(r["matched"], 1)
        self.assertEqual(r["iv_compared"], 0)
        self.assertEqual(r["mid_agree"], 1)

    def test_empty_inputs(self):
        r = compare([], [])
        self.assertEqual(r["matched"], 0)


if __name__ == "__main__":
    unittest.main()
