"""The credit spread's minimum credit-to-width is a config value, not a literal.

`find_credit_spreads` carried `if net_credit > (0.20 * strike_width)` twice —
once on the bull put branch, once on the bear call — with the comment
"Profitability Filter ... (Relaxed)" and no way to reach it. The sister
function `find_iron_condors` had read the same quantity from
`config['filters']['iron_condor']['min_credit_to_width']` since it was
written, so the number was already a tunable on one board and a magic literal
on the other.

The threshold is load-bearing rather than cosmetic. A bull put spread at
credit-to-width `r` pays `r * width` and risks `(1 - r) * width`, so it needs
a **`1 - r` win rate to break even** before costs: 80% at the shipped 0.20,
90% at 0.10. Measured on ^RUT 2026-08-13, dropping the bar to 0.10 took the
board from 1 candidate to 6 and admitted spreads needing 82-87% — against a
realised Bull Put win rate of 66.4% over 131 closed trades. So this is a knob
that decides which losing structures qualify, and it belongs somewhere it can
be seen and changed rather than buried at two call sites.

The DEFAULT does not move. 0.20 is what every result on record was produced
under.
"""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.options_screener import find_credit_spreads
from src.paths import repo_path


def _chain() -> pd.DataFrame:
    """Two puts one strike apart, priced so credit/width lands at 0.15.

    Sits between 0.10 and 0.20 on purpose: refused at the shipped default,
    admitted at a relaxed one, so a single fixture drives both branches.
    """
    rows = []
    for strike, prem, delta in ((100.0, 3.00, -0.30), (95.0, 2.25, -0.18)):
        rows.append({
            "symbol": "TEST", "type": "put", "strike": strike,
            "expiration": "2026-09-18", "premium": prem, "delta": delta,
            "bid": prem - 0.05, "ask": prem + 0.05, "quality_score": 0.5,
        })
    return pd.DataFrame(rows)


def _cfg(value):
    return {"filters": {"credit_spreads": {"min_credit_to_width": value}}}


class TestMinCreditToWidthIsConfigurable(unittest.TestCase):

    def test_the_fixture_sits_between_the_two_thresholds(self):
        """Stated, not assumed — if the fixture drifts the rest is vacuous."""
        chain = _chain()
        credit = 3.00 - 2.25
        width = 100.0 - 95.0
        self.assertAlmostEqual(credit / width, 0.15, places=9)
        self.assertEqual(len(chain), 2)

    def test_the_shipped_default_refuses_it(self):
        """0.15 < 0.20, so no config and an explicit 0.20 must both refuse."""
        self.assertTrue(find_credit_spreads(_chain()).empty)
        self.assertTrue(find_credit_spreads(_chain(), _cfg(0.20)).empty)

    def test_a_relaxed_threshold_admits_it(self):
        out = find_credit_spreads(_chain(), _cfg(0.10))
        self.assertFalse(out.empty, "0.10 must admit a 0.15 credit/width spread")
        self.assertEqual(out.iloc[0]["type"], "Bull Put")

    def test_a_stricter_threshold_refuses_more(self):
        self.assertTrue(find_credit_spreads(_chain(), _cfg(0.50)).empty)

    def test_an_absent_or_broken_config_falls_back_to_the_default(self):
        """Every config read in this codebase has to survive a missing key."""
        for cfg in ({}, {"filters": {}}, {"filters": {"credit_spreads": {}}},
                    {"filters": {"credit_spreads": None}}, None):
            with self.subTest(cfg=cfg):
                self.assertTrue(find_credit_spreads(_chain(), cfg).empty)

    def test_the_bear_call_branch_reads_the_same_knob(self):
        """It carried its own copy of the literal, so it needs its own proof."""
        calls = _chain().copy()
        calls["type"] = "call"
        # Mirror the put ladder: short low, long high, same 0.15 ratio.
        calls.loc[calls["strike"] == 100.0, "delta"] = 0.30
        calls.loc[calls["strike"] == 95.0, "delta"] = 0.18
        calls.loc[calls["strike"] == 95.0, "premium"] = 3.00
        calls.loc[calls["strike"] == 100.0, "premium"] = 2.25
        strict = find_credit_spreads(calls, _cfg(0.20))
        relaxed = find_credit_spreads(calls, _cfg(0.10))
        self.assertTrue(strict.empty)
        self.assertFalse(relaxed.empty,
                         "the bear call branch still holds its own literal")
        self.assertEqual(relaxed.iloc[0]["type"], "Bear Call")


class TestConfigCarriesTheValue(unittest.TestCase):

    def test_config_json_declares_it_at_the_shipped_default(self):
        """The point of the move: the number is visible in config.json."""
        with open(repo_path("config.json")) as fh:
            cfg = json.load(fh)
        block = cfg["filters"]["credit_spreads"]
        self.assertEqual(block["min_credit_to_width"], 0.20)

    def test_it_sits_beside_the_iron_condor_knob_it_mirrors(self):
        with open(repo_path("config.json")) as fh:
            filters = json.load(fh)["filters"]
        self.assertIn("iron_condor", filters)
        self.assertIn("credit_spreads", filters)
        self.assertEqual(filters["iron_condor"]["min_credit_to_width"],
                         filters["credit_spreads"]["min_credit_to_width"],
                         "the two boards' credit floors have silently diverged")


class TestBreakevenArithmetic(unittest.TestCase):
    """Why the number matters, pinned so the docstring cannot rot."""

    def test_breakeven_win_rate_is_one_minus_credit_to_width(self):
        for r in (0.10, 0.20, 0.35):
            width = 10.0
            credit = r * width
            max_profit, max_loss = credit, width - credit
            p = max_loss / (max_profit + max_loss)   # p*profit == (1-p)*loss
            self.assertAlmostEqual(p, 1 - r, places=9)
        self.assertAlmostEqual(1 - 0.20, 0.80, places=9)
        self.assertAlmostEqual(1 - 0.10, 0.90, places=9)


if __name__ == "__main__":
    unittest.main()
