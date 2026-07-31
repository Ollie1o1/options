"""A mark that nobody traded must not be allowed to write an exit.

`update_positions` marks every open leg through a fallback ladder. Its last
rung is a Black-Scholes price, which used to be computed at a hardcoded 30 vol
for every name — so an 80-vol contract got a badly wrong number, and if that
number happened to cross a stop or take-profit, a *real* exit row went into the
ledger that the gate cohort, the track record and every P&L statistic then read
forever. See docs/MARK_TRUSTWORTHINESS_SPEC.md.

Three properties are pinned here:
  1. the model rung prices at the row's own stored entry IV (schema v16),
     falling back to 0.30 only when the row has none;
  2. a model-sourced mark can never fire a price-based exit — the row stays
     OPEN and a warning names it — while the deterministic expiry settlement
     (priced off spot, not off the mark) still fires and remains the guarantee
     that no row hangs OPEN forever;
  3. the ladder prefers a live, uncrossed bid/ask mid over a stale last trade.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta

import pandas as pd

from src import paper_manager as pm
from src.paper_manager import (
    MARK_CLOSE,
    MARK_LAST,
    MARK_MID,
    MARK_MODEL,
    PaperManager,
    _mid_from_quote,
    _model_sigma,
)

SPOT = 100.0


# ── Fake network ────────────────────────────────────────────────────────────
class _FakeFastInfo:
    def __init__(self, last_price=None):
        self.last_price = last_price


class _FakeTicker:
    """Stands in for the underlying's ticker: spot only, no option data."""

    def __init__(self, symbol, session=None):
        self.symbol = symbol
        self.fast_info = _FakeFastInfo(SPOT)

    def history(self, period="1d"):
        return pd.DataFrame()


class _FakeYF:
    Ticker = _FakeTicker


def _fake_yf_and_session():
    return _FakeYF, None


class _StubbedManager(PaperManager):
    """A PaperManager whose three mark rungs are dictated by the test.

    Overriding the seams (rather than the whole ladder) keeps the real
    preference order, the real provenance tagging and the real exit gating
    under test while removing every network hop.
    """

    chain_quotes: dict = {}
    traded_mark = (None, None)
    model_price = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain_calls = []
        self.traded_calls = []
        self.model_sigmas = []

    def _fetch_chain_quotes(self, ticker, expiration):
        self.chain_calls.append((ticker, expiration))
        return dict(self.chain_quotes)

    def _fetch_traded_mark(self, symbol):
        self.traded_calls.append(symbol)
        return self.traded_mark

    def _model_mark(self, option_type, spot, strike, expiration, sigma):
        self.model_sigmas.append(sigma)
        return self.model_price


class _MarkTestCase(unittest.TestCase):
    """Temp DB + config, with paper_manager's network entry points stubbed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "trades.db")
        self.cfg = os.path.join(self.tmp, "config.json")
        with open(self.cfg, "w") as f:
            json.dump(
                {
                    "exit_rules": {
                        "time_exit_dte": 21,
                        "min_days_held": 3,
                        "long_option": {"take_profit": 1.00, "take_profit_delta": 0.80,
                                        "stop_loss": -0.50},
                        "spread": {"take_profit": 0.50, "stop_loss": -1.00},
                    },
                    "paper_trading": {"commission_per_contract": 0.0,
                                      "slippage_per_share": 0.05,
                                      "fx_conversion_rate": 0.0},
                },
                f,
            )
        self._real_yf = pm._get_yf_and_session
        self._real_rfr = pm._get_rfr if pm._HAS_RFR else None
        pm._get_yf_and_session = _fake_yf_and_session
        if pm._HAS_RFR:
            pm._get_rfr = lambda: 0.045
        PaperManager(db_path=self.db, config_path=self.cfg)  # creates the schema

    def tearDown(self):
        pm._get_yf_and_session = self._real_yf
        if self._real_rfr is not None:
            pm._get_rfr = self._real_rfr
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _manager(self, **attrs):
        mgr = _StubbedManager(db_path=self.db, config_path=self.cfg)
        for k, v in attrs.items():
            setattr(mgr, k, v)
        return mgr

    def _insert_open_row(self, **overrides):
        """Insert a long-call row: 100 strike, $3.50 entry, 60 DTE, held 30d."""
        row = {
            "date": str(date.today() - timedelta(days=30)),
            "ticker": "TSTX",
            "expiration": str(date.today() + timedelta(days=60)),
            "strike": 100.0,
            "type": "call",
            "entry_price": 3.50,
            "quality_score": 0.70,
            "strategy_name": "Long Call",
            "status": "OPEN",
            "entry_iv": None,
        }
        row.update(overrides)
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with sqlite3.connect(self.db) as conn:
            cur = conn.execute(
                f"INSERT INTO trades ({cols}) VALUES ({marks})", tuple(row.values())
            )
            return cur.lastrowid

    def _row(self, entry_id):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM trades WHERE entry_id=?", (entry_id,)
            ).fetchone()


class TestModelMarkNeverFiresAnExit(_MarkTestCase):
    def test_model_mark_crossing_the_stop_leaves_the_row_open(self):
        # $1.00 against a $3.50 entry is -71%, well through the -50% stop.
        entry_id = self._insert_open_row()
        mgr = self._manager(model_price=1.00)
        with self.assertLogs("src.paper_manager", level="WARNING") as logs:
            mgr.update_positions()
        row = self._row(entry_id)
        self.assertEqual(row["status"], "OPEN")
        self.assertIsNone(row["exit_reason"])
        self.assertTrue(
            any("Exit checks skipped" in m and str(entry_id) in m and "TSTX" in m
                for m in logs.output),
            f"no skip warning naming the row: {logs.output}",
        )

    def test_a_market_mark_at_the_same_price_still_stops_out(self):
        # Regression: the gate is the mark's provenance, not its level.
        entry_id = self._insert_open_row()
        mgr = self._manager(traded_mark=(1.00, MARK_LAST))
        mgr.update_positions()
        row = self._row(entry_id)
        self.assertEqual(row["status"], "CLOSED")
        self.assertIn("Stop Loss", row["exit_reason"])
        self.assertNotIn("model mark", row["exit_reason"])

    def test_model_mark_does_not_set_the_high_water_mark(self):
        # max_price_seen is ledger data, so a fabricated price stays out of it.
        entry_id = self._insert_open_row()
        self._manager(model_price=99.0).update_positions()
        self.assertIsNone(self._row(entry_id)["max_price_seen"])

    def test_one_model_leg_gates_a_whole_spread(self):
        # The structure's cost-to-close is only as trustworthy as its worst leg:
        # the short leg is quoted, the long one is not, so no exit may fire.
        entry_id = self._insert_open_row(
            strike=100.0, type="put", strategy_name="Bull Put Spread",
            long_strike=95.0, net_credit=1.00,
        )
        mgr = self._manager(
            chain_quotes={(100.0, "put"): (0.01, 0.03)},   # short leg quoted
            traded_mark=(None, None),
            model_price=0.20,                              # long leg fabricated
        )
        with self.assertLogs("src.paper_manager", level="WARNING") as logs:
            mgr.update_positions()
        self.assertEqual(self._row(entry_id)["status"], "OPEN")
        self.assertTrue(
            any("Exit checks skipped" in m and "put $95" in m for m in logs.output),
            f"no skip warning naming the model leg: {logs.output}",
        )

    def test_time_exit_still_fires_on_a_model_mark_but_is_stamped(self):
        # A pure-DTE rule never reads the mark's level, so it may fire — the
        # exit price it records is model-sourced and says so in the ledger.
        entry_id = self._insert_open_row(
            expiration=str(date.today() + timedelta(days=10))
        )
        mgr = self._manager(model_price=1.00)
        mgr.update_positions()
        row = self._row(entry_id)
        self.assertEqual(row["status"], "CLOSED")
        self.assertIn("Time Exit", row["exit_reason"])
        self.assertIn("(model mark)", row["exit_reason"])


class TestModelFallbackSigma(_MarkTestCase):
    def test_stored_entry_iv_is_used(self):
        self._insert_open_row(entry_iv=0.80)
        mgr = self._manager(model_price=3.60)
        mgr.update_positions()
        self.assertEqual(mgr.model_sigmas, [0.80])

    def test_missing_entry_iv_falls_back_to_thirty_vol(self):
        self._insert_open_row(entry_iv=None)
        mgr = self._manager(model_price=3.60)
        mgr.update_positions()
        self.assertEqual(mgr.model_sigmas, [0.30])

    def test_model_sigma_helper_rejects_unusable_values(self):
        self.assertEqual(_model_sigma(0.80), 0.80)
        self.assertEqual(_model_sigma(None), 0.30)
        self.assertEqual(_model_sigma(0.0), 0.30)
        self.assertEqual(_model_sigma(-1.0), 0.30)
        self.assertEqual(_model_sigma("junk"), 0.30)
        self.assertEqual(_model_sigma(99.0), 0.30)   # 9900% vol is a data error

    def test_sigma_reaches_the_pricer(self):
        # The real _model_mark must hand its sigma straight to american_price.
        from src import utils

        seen = {}

        def _spy(option_type, S, K, T, r, sigma):
            seen["sigma"] = sigma
            return 4.20

        real = utils.american_price
        utils.american_price = _spy
        try:
            mgr = PaperManager(db_path=self.db, config_path=self.cfg)
            price = mgr._model_mark(
                "call", SPOT, 100.0, str(date.today() + timedelta(days=30)), 0.80
            )
        finally:
            utils.american_price = real
        self.assertEqual(price, 4.20)
        self.assertEqual(seen["sigma"], 0.80)


class TestMarkPreferenceOrder(_MarkTestCase):
    KEY = ("TSTX", "2030-01-18", 100.0, "call")

    def _mark(self, mgr, quotes):
        return mgr._mark_option_leg(self.KEY, "TSTXSYM", quotes, SPOT, 0.30)

    def test_a_sane_two_sided_book_gives_a_mid(self):
        mgr = self._manager(traded_mark=(2.00, MARK_LAST), model_price=9.99)
        self.assertEqual(self._mark(mgr, {(100.0, "call"): (1.00, 1.20)}), (1.10, MARK_MID))

    def test_a_crossed_book_falls_through_to_the_last_trade(self):
        mgr = self._manager(traded_mark=(2.00, MARK_LAST), model_price=9.99)
        self.assertEqual(self._mark(mgr, {(100.0, "call"): (1.50, 1.00)}), (2.00, MARK_LAST))

    def test_a_one_sided_book_falls_through_to_the_last_trade(self):
        mgr = self._manager(traded_mark=(2.00, MARK_LAST), model_price=9.99)
        self.assertEqual(self._mark(mgr, {(100.0, "call"): (1.00, 0.0)}), (2.00, MARK_LAST))
        self.assertEqual(self._mark(mgr, {(100.0, "call"): (None, 1.20)}), (2.00, MARK_LAST))

    def test_no_quote_and_no_trade_falls_through_to_close_then_model(self):
        mgr = self._manager(traded_mark=(1.80, MARK_CLOSE), model_price=9.99)
        self.assertEqual(self._mark(mgr, {}), (1.80, MARK_CLOSE))
        mgr.traded_mark = (None, None)
        self.assertEqual(self._mark(mgr, {}), (9.99, MARK_MODEL))

    def test_nothing_at_all_marks_nothing(self):
        mgr = self._manager(traded_mark=(None, None), model_price=None)
        self.assertEqual(self._mark(mgr, {}), (None, None))

    def test_mid_helper_guards(self):
        self.assertEqual(_mid_from_quote(1.0, 1.2), 1.1)
        self.assertIsNone(_mid_from_quote(1.2, 1.0))    # crossed
        self.assertIsNone(_mid_from_quote(0.0, 1.2))    # one-sided
        self.assertIsNone(_mid_from_quote(1.0, None))
        self.assertIsNone(_mid_from_quote(None, None))
        self.assertIsNone(_mid_from_quote("x", "y"))

    def test_traded_mark_prefers_the_last_trade_then_the_daily_close(self):
        # The two market rungs below the mid, exercised against a fake ticker.
        class _Tkr:
            def __init__(self, last, close):
                self.fast_info = _FakeFastInfo(last)
                self._close = close

            def history(self, period="1d"):
                if self._close is None:
                    return pd.DataFrame()
                return pd.DataFrame({"Close": [self._close]})

        mgr = PaperManager(db_path=self.db, config_path=self.cfg)
        for last, close, expected in (
            (2.50, 1.80, (2.50, MARK_LAST)),
            (None, 1.80, (1.80, MARK_CLOSE)),
            (0.0, 1.80, (1.80, MARK_CLOSE)),
            (None, None, (None, None)),
        ):
            tkr = _Tkr(last, close)
            pm._get_yf_and_session = lambda t=tkr: (type("Y", (), {"Ticker": staticmethod(lambda *a, **k: t)}), None)
            self.assertEqual(mgr._fetch_traded_mark("TSTXSYM"), expected)

    def test_one_chain_call_per_ticker_expiration_pair(self):
        # Two legs on the same pair must not cost two chain requests.
        self._insert_open_row(strike=100.0, strategy_name="Bull Put Spread",
                              type="put", long_strike=95.0, net_credit=1.00)
        mgr = self._manager(traded_mark=(0.40, MARK_LAST))
        mgr.update_positions()
        self.assertEqual(mgr.chain_calls.count(("TSTX", str(date.today() + timedelta(days=60)))), 1)
        self.assertEqual(len(mgr.chain_calls), 1)


class TestExpirySettlementUnaffected(_MarkTestCase):
    def test_expired_row_settles_at_intrinsic_with_only_a_model_mark(self):
        # The terminal guarantee: prices off spot, so no mark source can block it.
        entry_id = self._insert_open_row(
            expiration=str(date.today() - timedelta(days=1)), strike=90.0
        )
        mgr = self._manager(model_price=1.00)
        mgr.update_positions()
        row = self._row(entry_id)
        self.assertEqual(row["status"], "CLOSED")
        self.assertEqual(row["exit_reason"], "Expired (settled at intrinsic)")
        self.assertAlmostEqual(row["exit_price"], SPOT - 90.0, places=4)

    def test_expired_row_settles_with_no_mark_at_all(self):
        entry_id = self._insert_open_row(
            expiration=str(date.today() - timedelta(days=1)), strike=90.0
        )
        self._manager(traded_mark=(None, None), model_price=None).update_positions()
        self.assertEqual(self._row(entry_id)["status"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
