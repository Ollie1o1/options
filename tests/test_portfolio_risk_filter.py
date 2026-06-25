"""Tests for the RISK-OFF advisory flag.

In paper/research mode the portfolio GEX gate should WARN but not erase the
scan's picks (the picks are data for validating the screener, not live orders).
A config flag controls whether RISK-OFF filters picks or is advisory-only.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_portfolio_risk_filter -v
"""
from __future__ import annotations

import unittest

from src.portfolio_risk import risk_off_filters_picks


class RiskOffFilterFlagTest(unittest.TestCase):
    def test_defaults_to_filtering_when_absent(self):
        # Backward-compatible default: preserve the existing enforcement.
        self.assertTrue(risk_off_filters_picks({}))
        self.assertTrue(risk_off_filters_picks(None))

    def test_false_makes_advisory(self):
        self.assertFalse(risk_off_filters_picks({"portfolio_gex_filter_picks": False}))

    def test_true_keeps_filtering(self):
        self.assertTrue(risk_off_filters_picks({"portfolio_gex_filter_picks": True}))

    def test_never_raises_on_garbage(self):
        self.assertTrue(risk_off_filters_picks({"portfolio_gex_filter_picks": "yes"}) in (True, False))


if __name__ == "__main__":
    unittest.main()
