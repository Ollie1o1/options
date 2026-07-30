"""Affordability filtering of scanner picks, applied BEFORE the top-N cut.

On 2026-07-30 the short-premium auto-log window scored 1,109 contracts, took
the top 5 by quality, and logged nothing: every one was a cash-secured put on
a $134-$747 underlying, so the $4,000 budget gate refused all five at the door.
The window burned a full scan to log zero rows, and reported it as "skipped 5
duplicates".

The scan ranks what the operator cannot buy. Affordability has to be a filter
on the candidate pool, not a rejection after the slots are already spent —
exactly as the allowlist filter is applied before ``.head(top_n)``.

Scan rows are not trades rows: the ticker is ``symbol``, and the multi-leg
worst case arrives as ``max_risk`` (condors) or ``max_loss`` (verticals)
rather than ``max_loss_usd``. That mapping is what these tests pin.
"""
import unittest

from src.capital_risk import capital_at_risk_for_pick, pick_within_budget


class TestSingleLegPicks(unittest.TestCase):
    def test_long_call_pick_risks_the_debit(self):
        pick = {"symbol": "MSFT", "strike": 465.0, "entry_price": 12.50}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Long Call"), 1250.0
        )

    def test_cash_secured_put_pick_risks_the_collateral(self):
        # The AAPL pick refused on 2026-07-30: (318.75 - 0.50) x 100.
        pick = {"symbol": "AAPL", "strike": 318.75, "entry_price": 0.50}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Short Put"), 31825.0
        )

    def test_expensive_short_put_is_outside_a_4000_budget(self):
        pick = {"symbol": "AAPL", "strike": 318.75, "entry_price": 0.50}
        self.assertFalse(pick_within_budget(pick, "Short Put", 4000.0))

    def test_cheap_short_put_is_inside_a_4000_budget(self):
        # A $30 underlying is the shape that CAN be logged under the cap.
        pick = {"symbol": "F", "strike": 30.0, "entry_price": 0.40}
        self.assertTrue(pick_within_budget(pick, "Short Put", 4000.0))

    def test_pick_uses_symbol_not_ticker_for_the_multiplier(self):
        # Crypto rows are whole-coin (multiplier 1), and scan rows key on `symbol`.
        pick = {"symbol": "BTC", "strike": 90000.0, "entry_price": 1200.0}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Long Call"), 1200.0
        )


class TestPremiumColumnPrecedence(unittest.TestCase):
    """A scan row has no ``entry_price``. The logging loop derives the premium as
    ``ask`` -> ``lastPrice`` -> ``premium``, so sizing must use the same ladder or
    every single-leg pick sizes as unknown and the filter drops the whole pool."""

    def test_ask_is_the_premium_when_present(self):
        pick = {"symbol": "MSFT", "strike": 465.0, "ask": 12.50, "lastPrice": 9.99}
        self.assertAlmostEqual(capital_at_risk_for_pick(pick, "Long Call"), 1250.0)

    def test_last_price_is_used_when_ask_is_absent(self):
        pick = {"symbol": "MSFT", "strike": 465.0, "lastPrice": 12.50}
        self.assertAlmostEqual(capital_at_risk_for_pick(pick, "Long Call"), 1250.0)

    def test_premium_is_the_final_fallback(self):
        pick = {"symbol": "MSFT", "strike": 465.0, "premium": 12.50}
        self.assertAlmostEqual(capital_at_risk_for_pick(pick, "Long Call"), 1250.0)

    def test_zero_ask_falls_through_to_last_price(self):
        # A 0 ask is a missing quote, not a free option — same rule the loop uses.
        pick = {"symbol": "MSFT", "strike": 465.0, "ask": 0.0, "lastPrice": 12.50}
        self.assertAlmostEqual(capital_at_risk_for_pick(pick, "Long Call"), 1250.0)

    def test_explicit_entry_price_still_wins(self):
        # Spread payloads and tests pass entry_price directly; keep honouring it.
        pick = {"symbol": "MSFT", "strike": 465.0, "entry_price": 3.00, "ask": 12.50}
        self.assertAlmostEqual(capital_at_risk_for_pick(pick, "Long Call"), 300.0)

    def test_short_put_sizes_from_the_ask_ladder(self):
        # The AAPL row as the scanner actually produces it — no entry_price column.
        pick = {"symbol": "AAPL", "strike": 318.75, "ask": 0.50}
        self.assertAlmostEqual(capital_at_risk_for_pick(pick, "Short Put"), 31825.0)
        self.assertFalse(pick_within_budget(pick, "Short Put", 4000.0))


class TestSpreadPicks(unittest.TestCase):
    def test_iron_condor_pick_uses_scan_max_risk(self):
        # The QQQ condor logged on 2026-07-30 carried max_risk = $3,204.
        pick = {"symbol": "QQQ", "total_credit": 17.96, "max_risk": 3204.0}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Iron Condor"), 3204.0
        )

    def test_iron_condor_over_budget_is_refused(self):
        pick = {"symbol": "QQQ", "total_credit": 9.75, "max_risk": 5525.50}
        self.assertFalse(pick_within_budget(pick, "Iron Condor", 4000.0))

    def test_vertical_pick_uses_scan_max_loss(self):
        # Bear Call AMAT 525/550, $6.80 credit -> $1,820 max loss.
        pick = {"symbol": "AMAT", "net_credit": 6.80, "max_loss": 1820.0}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Bear Call"), 1820.0
        )

    def test_vertical_within_budget_passes(self):
        pick = {"symbol": "IWM", "net_credit": 0.495, "max_loss": 51.0}
        self.assertTrue(pick_within_budget(pick, "Bear Call", 4000.0))


class TestSingleLegMaxLossIsNotAuthoritative(unittest.TestCase):
    """``trade_analysis`` stamps ``max_loss = entry_price * 100`` on EVERY
    single-leg row — the long-premium debit, computed without reference to the
    strategy. On a short put that number is meaningless: the real exposure is
    collateral, ~600x larger.

    Only a defined-risk multi-leg structure records a true worst case, so only
    those may size from a stored ``max_loss``/``max_risk``. Reading it off a
    single leg sized the AAPL cash-secured put at $50 instead of $31,850 and let
    it pass a $4,000 budget filter that then refused it at the ledger door.
    """

    def test_short_put_ignores_the_single_leg_max_loss_column(self):
        pick = {"symbol": "AAPL", "strike": 318.50, "ask": 0.50, "max_loss": 50.0}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Short Put"), 31800.0
        )

    def test_short_put_with_debit_max_loss_is_still_over_budget(self):
        pick = {"symbol": "AAPL", "strike": 318.50, "ask": 0.50, "max_loss": 50.0}
        self.assertFalse(pick_within_budget(pick, "Short Put", 4000.0))

    def test_long_call_agrees_with_its_max_loss_column_anyway(self):
        # For long premium the debit IS the max loss, so the answer is unchanged.
        pick = {"symbol": "MSFT", "strike": 465.0, "ask": 12.50, "max_loss": 1250.0}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Long Call"), 1250.0
        )

    def test_short_call_stays_unbounded_despite_a_max_loss_column(self):
        pick = {"symbol": "TSLA", "strike": 500.0, "ask": 3.20, "max_loss": 320.0}
        self.assertIsNone(capital_at_risk_for_pick(pick, "Short Call"))

    def test_spread_still_trusts_its_stored_worst_case(self):
        # The multi-leg case must keep working — that number is real.
        pick = {"symbol": "SPY", "net_credit": 10.63, "max_loss": 2436.50}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Bull Put"), 2436.50
        )


class TestUnsizablePicks(unittest.TestCase):
    def test_naked_short_call_is_unbounded_and_fails_a_cap(self):
        # Unbounded risk cannot be shown to fit a budget, so it must not pass.
        pick = {"symbol": "TSLA", "strike": 500.0, "entry_price": 3.20}
        self.assertIsNone(capital_at_risk_for_pick(pick, "Short Call"))
        self.assertFalse(pick_within_budget(pick, "Short Call", 4000.0))

    def test_unbounded_pick_passes_when_no_cap_is_set(self):
        pick = {"symbol": "TSLA", "strike": 500.0, "entry_price": 3.20}
        self.assertTrue(pick_within_budget(pick, "Short Call", None))

    def test_missing_price_is_unsizable(self):
        self.assertIsNone(capital_at_risk_for_pick({"symbol": "X"}, "Long Call"))


class TestMappingRules(unittest.TestCase):
    def test_scan_max_risk_wins_over_a_derived_width(self):
        # A stored worst case is authoritative; never re-derive when present.
        pick = {"symbol": "SPY", "net_credit": 10.63, "max_loss": 2436.50,
                "spread_width": 35.0}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Bull Put"), 2436.50
        )

    def test_zero_max_loss_falls_through_to_the_width_derivation(self):
        # log_spread defaults max_loss to 0 when the scan omits it; 0 is "unknown",
        # not "risk-free", so the width/credit derivation must still run.
        pick = {"symbol": "DIA", "net_credit": 0.25, "max_loss": 0,
                "spread_width": 2.5}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Bull Put"), 225.0
        )

    def test_pandas_row_with_nan_max_loss_is_tolerated(self):
        # Candidates arrive as DataFrame rows; absent numeric cells are NaN.
        import math
        pick = {"symbol": "DIA", "net_credit": 0.25, "max_loss": math.nan,
                "spread_width": 2.5}
        self.assertAlmostEqual(
            capital_at_risk_for_pick(pick, "Bull Put"), 225.0
        )


class TestBudgetCapFromConfig(unittest.TestCase):
    """The screener must read the same cap the ledger enforces, or the
    pre-filter and the door-gate disagree and picks vanish silently."""

    def setUp(self):
        import json
        import os
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.tmpdir, "config.json")
        self._json = json
        self._write({"auto_log": {"max_capital_at_risk": 4000}})

    def _write(self, cfg):
        with open(self.cfg_path, "w") as f:
            self._json.dump(cfg, f)

    def test_reads_the_configured_cap(self):
        from src.options_screener import auto_log_budget_cap
        self.assertEqual(auto_log_budget_cap(self.cfg_path), 4000.0)

    def test_absent_cap_means_no_constraint(self):
        from src.options_screener import auto_log_budget_cap
        self._write({"auto_log": {}})
        self.assertIsNone(auto_log_budget_cap(self.cfg_path))

    def test_null_cap_means_no_constraint(self):
        from src.options_screener import auto_log_budget_cap
        self._write({"auto_log": {"max_capital_at_risk": None}})
        self.assertIsNone(auto_log_budget_cap(self.cfg_path))

    def test_unreadable_config_means_no_constraint(self):
        # Never let a config problem silently filter every pick out of the scan.
        from src.options_screener import auto_log_budget_cap
        self.assertIsNone(auto_log_budget_cap(self.cfg_path + ".missing"))

    def test_matches_the_cap_the_ledger_enforces(self):
        # Same config key PaperTradeManager reads, so the two cannot drift.
        from src.options_screener import auto_log_budget_cap
        from src.paper_manager import PaperManager
        import os
        import tempfile
        db = os.path.join(tempfile.mkdtemp(), "t.db")
        pm = PaperManager(db_path=db, config_path=self.cfg_path)
        self.assertEqual(auto_log_budget_cap(self.cfg_path), pm._max_capital_at_risk)


if __name__ == "__main__":
    unittest.main()
