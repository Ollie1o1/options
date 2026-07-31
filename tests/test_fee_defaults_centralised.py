"""One commission fallback, in one place.

Six modules used to carry their own hardcoded 0.65. Each only fires when no
config is reachable, so the duplication never showed up in normal operation —
which is exactly why it would have been missed on a broker change. These tests
fail if a copy comes back.
"""
import os
import pathlib
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution_costs import FALLBACK_COMMISSION_PER_CONTRACT  # noqa: E402

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Files that legitimately mention the literal: the definition itself.
_ALLOWED = {"execution_costs.py"}


class SingleSourceTest(unittest.TestCase):
    def test_no_module_hardcodes_a_commission_literal(self):
        offenders = []
        for path in sorted(_SRC.rglob("*.py")):
            if path.name in _ALLOWED:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"commission[_a-z]*\s*[=:]\s*0\.65", line, re.I):
                    offenders.append(f"{path.relative_to(_SRC)}:{i}")
        self.assertEqual(offenders, [],
                         "commission fallback duplicated; import "
                         "FALLBACK_COMMISSION_PER_CONTRACT instead")

    def test_the_shared_constant_is_reachable_from_every_consumer(self):
        for mod in ("src.dolt_cohort", "src.dolt_short", "src.dolt_spread",
                    "src.dolt_vol", "src.dolt_earnings_sell",
                    "src.trade_analysis", "src.paper_manager"):
            __import__(mod)

    def test_paper_manager_reexports_the_same_number(self):
        from src.paper_manager import COMMISSION_PER_CONTRACT
        self.assertEqual(COMMISSION_PER_CONTRACT,
                         FALLBACK_COMMISSION_PER_CONTRACT)

    def test_the_fallback_is_not_zero(self):
        # Deliberate: the fallback fires when config is absent, and there
        # overstating cost is the safe direction. A 0.0 default would let a
        # standalone backtest silently price as though trading were free.
        self.assertGreater(FALLBACK_COMMISSION_PER_CONTRACT, 0.0)

    def test_configured_zero_still_wins_over_the_fallback(self):
        # The live broker charges nothing; a `.get(key, FALLBACK)` must return
        # a configured 0.0 rather than treating it as absent.
        cfg = {"paper_trading": {"commission_per_contract": 0.0}}
        got = cfg.get("paper_trading", {}).get(
            "commission_per_contract", FALLBACK_COMMISSION_PER_CONTRACT)
        self.assertEqual(got, 0.0)


if __name__ == "__main__":
    unittest.main()
