import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.execution import sizing, exits, ticket, slippage


# ── Sizing ──────────────────────────────────────────────────────────────────

class TestSizing(unittest.TestCase):
    def test_basic_risk_cap_binds(self):
        s = sizing.size_position(account_value=10_000, entry_price=4.0, stop_price=2.0)
        # risk/contract = (4-2)*100 = 200; 2% of 10k = 200 -> 1 contract
        self.assertEqual(s.contracts, 1)
        self.assertAlmostEqual(s.dollar_risk, 200.0)
        self.assertAlmostEqual(s.risk_pct, 0.02, places=4)

    def test_scales_with_account(self):
        s = sizing.size_position(account_value=100_000, entry_price=4.0, stop_price=2.0)
        # risk budget 2000 / 200 = 10 contracts (cost cap 10% = 10k / 400 = 25)
        self.assertEqual(s.contracts, 10)
        self.assertAlmostEqual(s.cost_basis, 4000.0)

    def test_cost_cap_can_bind_before_risk_cap(self):
        # Tight stop -> risk cap would allow many, but 10% cost cap limits it.
        s = sizing.size_position(account_value=10_000, entry_price=5.0, stop_price=4.9)
        # cost cap: 1000 / 500 = 2 contracts
        self.assertEqual(s.contracts, 2)

    def test_stop_above_entry_returns_zero(self):
        s = sizing.size_position(account_value=10_000, entry_price=2.0, stop_price=3.0)
        self.assertEqual(s.contracts, 0)
        self.assertIn("stop", s.notes.lower())

    def test_zero_account_returns_zero(self):
        s = sizing.size_position(account_value=0, entry_price=4.0, stop_price=2.0)
        self.assertEqual(s.contracts, 0)

    def test_negative_kelly_edge_zeroes_position(self):
        # win_prob 0.5, payoff 1.0 -> Kelly f = 0 -> no position
        s = sizing.size_position(account_value=100_000, entry_price=4.0, stop_price=2.0,
                                 win_prob=0.5, payoff_ratio=1.0)
        self.assertEqual(s.contracts, 0)

    def test_kelly_caps_at_max_risk_pct(self):
        # Strong edge -> half-Kelly large, but max_risk_pct=2% caps it.
        s = sizing.size_position(account_value=100_000, entry_price=4.0, stop_price=2.0,
                                 win_prob=0.7, payoff_ratio=3.0)
        self.assertLessEqual(s.risk_pct, 0.02 + 1e-9)


# ── Exits ───────────────────────────────────────────────────────────────────

class TestExits(unittest.TestCase):
    def test_long_call_levels_from_defaults(self):
        e = exits.compute_exits(entry_price=4.0, expiration="2026-08-21",
                                today="2026-06-07", config={})
        # defaults: long.tp=1.0 -> 8.0 ; long.sl=-0.5 -> 2.0 ; time_exit_dte=21
        self.assertAlmostEqual(e.take_profit_price, 8.0)
        self.assertAlmostEqual(e.stop_price, 2.0)
        self.assertEqual(e.time_exit_dte, 21)
        # expiration 2026-08-21 minus 21 days = 2026-07-31
        self.assertEqual(e.time_exit_date, "2026-07-31")

    def test_respects_config_overrides(self):
        cfg = {"exit_rules": {"time_exit_dte": 14,
                              "long_option": {"take_profit": 0.5, "stop_loss": -0.3}}}
        e = exits.compute_exits(entry_price=10.0, expiration="2026-09-18",
                                today="2026-06-07", config=cfg)
        self.assertAlmostEqual(e.take_profit_price, 15.0)
        self.assertAlmostEqual(e.stop_price, 7.0)
        self.assertEqual(e.time_exit_dte, 14)


# ── Ticket (the hard switch) ────────────────────────────────────────────────

def _pick():
    return {"ticker": "AAPL", "strike": 180.0, "expiration": "2026-08-21",
            "option_type": "call", "bid": 4.10, "ask": 4.30}


class TestTicketGate(unittest.TestCase):
    def setUp(self):
        self.s = sizing.size_position(account_value=50_000, entry_price=4.20, stop_price=2.10)
        self.e = exits.compute_exits(entry_price=4.20, expiration="2026-08-21",
                                     today="2026-06-07", config={})

    def test_refuses_when_gate_not_ready(self):
        t = ticket.render_ticket(_pick(), self.s, self.e,
                                 gate_decision="GATHERING", live_enabled=True)
        self.assertEqual(t["mode"], "DRY_RUN")
        self.assertIn("DRY RUN", t["text"])
        self.assertNotIn("LIVE ORDER", t["text"])

    def test_refuses_when_live_flag_off(self):
        t = ticket.render_ticket(_pick(), self.s, self.e,
                                 gate_decision="READY", live_enabled=False)
        self.assertEqual(t["mode"], "DRY_RUN")
        self.assertIn("DRY RUN", t["text"])

    def test_emits_live_ticket_when_both_open(self):
        t = ticket.render_ticket(_pick(), self.s, self.e,
                                 gate_decision="READY", live_enabled=True)
        self.assertEqual(t["mode"], "LIVE")
        self.assertIn("LIVE ORDER", t["text"])
        self.assertIn("AAPL", t["text"])
        self.assertIn(str(self.s.contracts), t["text"])
        self.assertIn("4.2", t["text"])      # limit near mid
        self.assertGreater(t["limit_price"], 0)

    def test_zero_contracts_refuses_even_when_open(self):
        zero = sizing.size_position(account_value=0, entry_price=4.2, stop_price=2.1)
        t = ticket.render_ticket(_pick(), zero, self.e,
                                 gate_decision="READY", live_enabled=True)
        self.assertEqual(t["mode"], "DRY_RUN")
        self.assertIn("0 contracts", t["text"])


# ── Slippage ────────────────────────────────────────────────────────────────

class TestSlippage(unittest.TestCase):
    def test_record_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "fills.db")
            slippage.record_fill(db, ticker="AAPL", intended_price=4.20,
                                 actual_price=4.30, contracts=2, ticket_id="t1")
            slippage.record_fill(db, ticker="MSFT", intended_price=3.00,
                                 actual_price=3.00, contracts=1, ticket_id="t2")
            rep = slippage.slippage_report(db)
            self.assertEqual(rep["n_fills"], 2)
            # avg per-contract slippage = (0.10 + 0.00)/2 = 0.05
            self.assertAlmostEqual(rep["avg_slippage_per_contract"], 0.05, places=4)
            # total $ slippage = 0.10*2*100 + 0 = 20
            self.assertAlmostEqual(rep["total_slippage_usd"], 20.0, places=2)

    def test_empty_report(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "fills.db")
            rep = slippage.slippage_report(db)
            self.assertEqual(rep["n_fills"], 0)


if __name__ == "__main__":
    unittest.main()
