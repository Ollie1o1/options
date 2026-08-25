"""Position sizing for the paper book: the arithmetic and the two queries.

The decision itself (`size`) is pure, so every branch of it is asserted here
without a database. The two queries that feed it get a temp ledger.

unittest style on purpose — the options venv has no pytest, so these have to be
runnable locally as well as in CI.

Reference: docs/BOOK_SIZING_SPEC.md §6.
"""
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.book_sizing import (SizingDecision, book_equity, load_sizing_config,
                             open_risk, size)
from src.capital_risk import capital_at_risk

CFG = {
    "enabled": True,
    "opening_balance": 50_000.0,
    "equity_basis_date": "2026-08-05",
    "sizing_start_date": "2026-08-19",
    "max_risk_pct": 0.02,
    "max_open_risk_pct": 0.10,
}


def _cfg(**over):
    out = dict(CFG)
    out.update(over)
    return out


class PureDecision(unittest.TestCase):
    """`size` — spec tests 1-5."""

    def test_unbounded_risk_refuses(self):
        # A naked call's capital_at_risk is None. None is not zero and not
        # free: it is a position whose loss cannot be bounded, so no contract
        # count can be justified.
        d = size(None, equity=40_110.0, open_risk=0.0, cfg=_cfg())
        self.assertEqual(d.contracts, 0)
        self.assertEqual(d.reason, "unbounded_risk")

    def test_floor_of_budget_over_risk(self):
        # $40,110 x 2% = $802.20 budget; $392 per contract -> 2.04 -> 2.
        d = size(392.0, equity=40_110.0, open_risk=0.0, cfg=_cfg())
        self.assertEqual(d.contracts, 2)
        self.assertEqual(d.reason, "risk_capped")
        self.assertAlmostEqual(d.risk_per_contract, 392.0)
        self.assertAlmostEqual(d.equity, 40_110.0)

    def test_below_one_contract_refuses_rather_than_rounding_up(self):
        # CRM at $1,468 against an $802 budget. Rounding up to 1 would place a
        # position 83% larger than the rule allows; sizing is a gate.
        d = size(1_468.0, equity=40_110.0, open_risk=0.0, cfg=_cfg())
        self.assertEqual(d.contracts, 0)
        self.assertEqual(d.reason, "below_one_contract")

    def test_concurrent_cap_reduces_then_refuses(self):
        # Ceiling is 10% = $4,011. With $3,600 already at risk the headroom is
        # $411, which buys 1 contract at $392 even though the per-trade budget
        # would allow 2.
        reduced = size(392.0, equity=40_110.0, open_risk=3_600.0, cfg=_cfg())
        self.assertEqual(reduced.contracts, 1)
        self.assertEqual(reduced.reason, "concurrent_capped")
        # At the ceiling there is no headroom at all and the trade refuses —
        # with the reason naming the cap that bound, not "too small".
        at_ceiling = size(392.0, equity=40_110.0, open_risk=4_011.0, cfg=_cfg())
        self.assertEqual(at_ceiling.contracts, 0)
        self.assertEqual(at_ceiling.reason, "concurrent_capped")

    def test_disabled_yields_exactly_one_contract(self):
        # Today's behaviour, stated rather than implied.
        d = size(1_468.0, equity=40_110.0, open_risk=99_999.0,
                 cfg=_cfg(enabled=False))
        self.assertEqual(d.contracts, 1)
        self.assertEqual(d.reason, "disabled")

    def test_blown_account_refuses(self):
        # Equity at or below zero has no 2% to allocate. Reported as its own
        # reason so it can never be read as "the position was too big".
        d = size(392.0, equity=0.0, open_risk=0.0, cfg=_cfg())
        self.assertEqual(d.contracts, 0)
        self.assertEqual(d.reason, "no_equity")

    def test_zero_risk_per_contract_is_not_infinite_contracts(self):
        # A structure priced at zero risk would divide by zero. It is a broken
        # input, not a free trade.
        d = size(0.0, equity=40_110.0, open_risk=0.0, cfg=_cfg())
        self.assertEqual(d.contracts, 0)
        self.assertEqual(d.reason, "unbounded_risk")

    def test_decision_is_frozen(self):
        d = size(392.0, equity=40_110.0, open_risk=0.0, cfg=_cfg())
        self.assertIsInstance(d, SizingDecision)
        with self.assertRaises(Exception):
            d.contracts = 5  # type: ignore[misc]


class BullPutSizesFromWidthMinusCredit(unittest.TestCase):
    """Spec test 6 — the defect `src/execution/sizing.py` would have introduced.

    Its formulas are `(entry - stop) * 100` and `entry * 100`, both long-premium.
    A Bull Put receives its credit and risks `width - credit`; sizing it off the
    credit prices the trade at a fraction of its loss and buys several times too
    many contracts. Asserted as a DIFFERENCE so a future edit cannot quietly
    reintroduce premium-based sizing and still pass.
    """

    def test_the_two_bases_disagree_on_contract_count(self):
        # $5 wide, $1.08 credit -> risks $392, receives $108.
        risk = capital_at_risk(
            strategy_name="Bull Put Spread",
            entry_price=1.08, spread_width=5.0, net_credit=1.08,
            quantity=1.0, ticker="WMT",
        )
        self.assertAlmostEqual(risk, 392.0)

        premium_basis = 1.08 * 100  # what a long-premium sizer would use
        sized_correctly = size(risk, 40_110.0, 0.0, _cfg())
        sized_off_premium = size(premium_basis, 40_110.0, 0.0, _cfg())

        self.assertEqual(sized_correctly.contracts, 2)
        self.assertEqual(sized_off_premium.contracts, 7)
        self.assertNotEqual(sized_correctly.contracts,
                            sized_off_premium.contracts)

    def test_credit_is_never_the_risk_for_a_credit_structure(self):
        risk = capital_at_risk(
            strategy_name="Bull Put Spread",
            entry_price=1.08, spread_width=5.0, net_credit=1.08, ticker="WMT")
        self.assertGreater(risk, 1.08 * 100)


def _ledger(path):
    """A trades table with only the columns these two queries read."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trades ("
        " entry_id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT,"
        " strategy_name TEXT, status TEXT, pnl_usd REAL, capital_at_risk REAL,"
        " entry_price REAL, strike REAL, max_loss_usd REAL, spread_width REAL,"
        " net_credit REAL, quantity REAL DEFAULT 1.0)")
    return conn


def _row(conn, **kw):
    base = {"date": "2026-08-20 10:00:00", "ticker": "WMT",
            "strategy_name": "Bull Put Spread", "status": "OPEN",
            "pnl_usd": None, "capital_at_risk": None, "entry_price": 1.0,
            "strike": 100.0, "max_loss_usd": None, "spread_width": None,
            "net_credit": None, "quantity": 1.0}
    base.update(kw)
    cols = ", ".join(base)
    conn.execute(f"INSERT INTO trades ({cols}) VALUES "
                 f"({', '.join('?' * len(base))})", tuple(base.values()))
    conn.commit()


class Equity(unittest.TestCase):
    """Spec test 7 — equity compounds off the book's own realised P&L."""

    def setUp(self):
        self.conn = _ledger(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_opening_balance_plus_realised_since_the_basis_date(self):
        _row(self.conn, date="2026-08-10 09:00:00", status="CLOSED", pnl_usd=-500.0)
        _row(self.conn, date="2026-08-12 09:00:00", status="CLOSED", pnl_usd=+120.0)
        self.assertAlmostEqual(book_equity(self.conn, CFG), 49_620.0)

    def test_excludes_the_old_book(self):
        # Entered AND closed before the restart: another book's money.
        _row(self.conn, date="2026-07-01 09:00:00", status="CLOSED", pnl_usd=-9_000.0)
        # Entered before the restart, closed after it: still the old book. The
        # restart splits on ENTRY date everywhere else in this system.
        _row(self.conn, date="2026-08-01 09:00:00", status="CLOSED", pnl_usd=-4_000.0)
        _row(self.conn, date="2026-08-10 09:00:00", status="CLOSED", pnl_usd=-500.0)
        self.assertAlmostEqual(book_equity(self.conn, CFG), 49_500.0)

    def test_open_positions_do_not_move_equity(self):
        # An open position's P&L is not realised. Marking it here would size
        # new trades off an unrealised number that moves every minute.
        _row(self.conn, date="2026-08-10 09:00:00", status="OPEN", pnl_usd=None)
        _row(self.conn, date="2026-08-10 09:00:00", status="CLOSED", pnl_usd=-500.0)
        self.assertAlmostEqual(book_equity(self.conn, CFG), 49_500.0)

    def test_empty_book_is_the_opening_balance(self):
        self.assertAlmostEqual(book_equity(self.conn, CFG), 50_000.0)


class OpenRisk(unittest.TestCase):
    """Spec test 8 — the legacy book is grandfathered out of the cap."""

    def setUp(self):
        self.conn = _ledger(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_excludes_positions_opened_before_the_sized_era(self):
        # Legacy: opened 2026-08-18, one day before sizing shipped. Open risk
        # across the real legacy book is $176,323 against a $4,011 cap; counting
        # it would refuse every new trade for months.
        _row(self.conn, date="2026-08-18 09:00:00", capital_at_risk=176_323.0)
        _row(self.conn, date="2026-08-19 09:00:00", capital_at_risk=800.0)
        self.assertAlmostEqual(open_risk(self.conn, CFG), 800.0)

    def test_closed_positions_are_not_open_risk(self):
        _row(self.conn, date="2026-08-20 09:00:00", capital_at_risk=800.0)
        _row(self.conn, date="2026-08-20 09:00:00", capital_at_risk=900.0,
             status="CLOSED", pnl_usd=50.0)
        self.assertAlmostEqual(open_risk(self.conn, CFG), 800.0)

    def test_null_capital_at_risk_is_recomputed_not_treated_as_zero(self):
        # NULL means "not recorded", never zero — summing it as zero would
        # under-report exposure and let the cap admit trades it should refuse.
        _row(self.conn, date="2026-08-20 09:00:00", capital_at_risk=None,
             entry_price=1.08, spread_width=5.0, net_credit=1.08, quantity=2.0)
        self.assertAlmostEqual(open_risk(self.conn, CFG), 784.0)

    def test_null_sizing_start_date_grandfathers_everything(self):
        # The sized era has not started, so nothing counts against the cap.
        _row(self.conn, date="2026-08-20 09:00:00", capital_at_risk=800.0)
        self.assertAlmostEqual(
            open_risk(self.conn, _cfg(sizing_start_date=None)), 0.0)


class ConfigLoading(unittest.TestCase):

    def test_missing_block_is_disabled(self):
        # A config without the block is a config that never opted in. Sizing
        # off means quantity 1 — the behaviour every historical row carries.
        cfg = load_sizing_config({})
        self.assertFalse(cfg["enabled"])
        self.assertEqual(size(392.0, 40_110.0, 0.0, cfg).contracts, 1)

    def test_real_config_is_loadable_and_enabled(self):
        import json
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "config.json")) as f:
            cfg = load_sizing_config(json.load(f))
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["opening_balance"], 50_000.0)
        # 0.015 since 2026-08-25. Raised entry throughput at IDENTICAL total
        # risk: the book was pinned against the concurrent cap (5 of 5 slots,
        # 0.9 positions of headroom), and throughput is slots over hold time.
        # The same 10% now buys 6.7 slots instead of 5. Candidate supply was
        # never the constraint - 168 Bull Puts cleared the gates the day this
        # was measured and 2-3 were taken.
        self.assertEqual(cfg["max_risk_pct"], 0.015)
        self.assertEqual(cfg["max_open_risk_pct"], 0.10)
        self.assertEqual(cfg["equity_basis_date"], "2026-08-05")

    def test_garbage_values_fall_back_rather_than_crashing_the_ledger(self):
        cfg = load_sizing_config(
            {"position_sizing": {"enabled": True, "max_risk_pct": "banana"}})
        self.assertEqual(cfg["max_risk_pct"], 0.02)


if __name__ == "__main__":
    unittest.main()
