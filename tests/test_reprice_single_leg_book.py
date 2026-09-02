"""tests/test_reprice_single_leg_book.py

Pure computation only — no db file, no network. Mirrors the fixture style of
tests/test_spread_surface_report.py and tests/test_track_record_equal_weighted.py.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spread_surface import Cell, SpreadSurface  # noqa: E402

from scripts.reprice_single_leg_book import (  # noqa: E402
    cluster_bootstrap_pf,
    gross_pct,
    is_expired,
    new_friction_fraction,
)


class GrossPctTests(unittest.TestCase):
    def test_short_leg_profits_when_price_falls(self):
        # Sold at 2.00, bought back at 0.50: short profits as price falls.
        self.assertAlmostEqual(gross_pct(2.00, 0.50, short=True), 0.75)

    def test_long_leg_profits_when_price_rises(self):
        self.assertAlmostEqual(gross_pct(1.00, 2.50, short=False), 1.5)

    def test_matches_paper_manager_formula_sign(self):
        # paper_manager._evaluate_short_single_leg_exit:
        #   pnl_raw = (entry_price - current_price) / entry_price
        entry, exit_ = 3.20, 4.10
        self.assertAlmostEqual(
            gross_pct(entry, exit_, short=True),
            (entry - exit_) / entry,
        )


class IsExpiredTests(unittest.TestCase):
    def test_expired_reason_detected(self):
        self.assertTrue(is_expired("Expired (settled at intrinsic)"))

    def test_other_reasons_not_expired(self):
        self.assertFalse(is_expired("Take Profit (50% @ 12d)"))
        self.assertFalse(is_expired(None))
        self.assertFalse(is_expired(""))


class NewFrictionFractionTests(unittest.TestCase):
    def setUp(self):
        # Three OI buckets at the SAME (delta, dte) cell — bucket 0 is the
        # illiquid pin, higher buckets progressively cheaper. This mirrors
        # the real surface (a populated bucket 0 always exists), so
        # relative(..., open_interest=None) hits bucket 0's OWN exact cell
        # directly instead of falling through the collapse ladder — and
        # oi_collapsed_relative's median-across-buckets then genuinely
        # differs from it, which is the property being tested.
        self.surface = SpreadSurface(
            {
                (1, 1, 0): Cell(n=40, rel_half_spread=0.05, median_depth=5),
                (1, 1, 2): Cell(n=50, rel_half_spread=0.02, median_depth=20),
                (1, 1, 3): Cell(n=30, rel_half_spread=0.01, median_depth=50),
            },
            {"fit_date": "2026-09-01"},
        )

    def test_known_oi_hits_exact_cell(self):
        frac, prov = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=500.0, round_trip=True,
        )
        self.assertEqual(prov, "cell")
        # half=0.02*2.00=0.04; round trip both sides, no commission: 2*0.04=0.08
        # fraction of mid: 0.08/2.00 = 0.04
        self.assertAlmostEqual(frac, 0.04)

    def test_hold_to_expiry_charges_opening_side_only(self):
        frac, _ = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=500.0, round_trip=False,
        )
        self.assertAlmostEqual(frac, 0.02)  # one side: 0.04/2.00 = 0.02

    def test_unknown_oi_conservative_pins_bucket_zero(self):
        # open_interest=None resolves cell_key's oi dimension to bucket 0,
        # which is populated here — an exact "cell" hit at the illiquid pin,
        # not a fallback rung.
        frac, prov = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=None, round_trip=True, central=False,
        )
        self.assertEqual(prov, "cell")
        # half=0.05*2.00=0.10; round trip: 2*0.10=0.20; fraction 0.20/2.00=0.10
        self.assertAlmostEqual(frac, 0.10)

    def test_unknown_oi_central_uses_oi_collapsed_marginal(self):
        frac, prov = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=None, round_trip=True, central=True,
        )
        self.assertEqual(prov, "oi_collapsed")
        # median of [0.05, 0.02, 0.01] = 0.02; half=0.04; round trip 0.08;
        # fraction 0.08/2.00 = 0.04 — strictly less than the conservative
        # bucket-0 pin (0.10) computed above.
        self.assertAlmostEqual(frac, 0.04)


class ClusterBootstrapPfTests(unittest.TestCase):
    def test_point_estimate_matches_plain_profit_factor(self):
        rows = [
            {"date": "2026-08-01", "r": 0.10},
            {"date": "2026-08-01", "r": -0.05},
            {"date": "2026-08-02", "r": 0.20},
            {"date": "2026-08-03", "r": -0.10},
        ]
        point, lo, hi = cluster_bootstrap_pf(rows, "r", n_boot=200, seed=1)
        # wins 0.30, losses 0.15 -> PF 2.0
        self.assertAlmostEqual(point, 2.0)
        self.assertIsNotNone(lo)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)

    def test_clustering_widens_the_interval_vs_row_level(self):
        # 10 entry days, 4 rows each. Every row's sign is fully determined by
        # its day (5 "good" days all positive, 5 "bad" days all negative) —
        # the day is the true independent unit; the 4 rows inside it carry
        # almost no extra information beyond magnitude noise. A row-level
        # bootstrap that resamples all 40 rows as if independent treats one
        # day's outcome as 4 pieces of evidence, understating uncertainty —
        # the same overcounting trap that inflated the ranker test and the
        # catalyst bootstrap. The correct cluster bootstrap must be wider.
        rows = []
        good_vals = [0.28, 0.30, 0.32, 0.34]
        bad_vals = [-0.23, -0.25, -0.27, -0.29]
        for d in (0, 2, 4, 6, 8):
            for v in good_vals:
                rows.append({"date": f"day{d}", "r": v})
        for d in (1, 3, 5, 7, 9):
            for v in bad_vals:
                rows.append({"date": f"day{d}", "r": v})
        _, lo_cluster, hi_cluster = cluster_bootstrap_pf(
            rows, "r", n_boot=4000, seed=1)

        def _row_level(rows, n_boot, seed):
            import random
            from scripts.publish_track_record import profit_factor
            rnd = random.Random(seed)
            vals = [r["r"] for r in rows]
            draws = [profit_factor([rnd.choice(vals) for _ in vals])
                     for _ in range(n_boot)]
            draws = sorted(d for d in draws if d is not None)
            return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]

        lo_row, hi_row = _row_level(rows, 4000, 1)
        self.assertGreater((hi_cluster - lo_cluster), (hi_row - lo_row))

    def test_single_cluster_returns_no_interval(self):
        rows = [{"date": "2026-08-01", "r": 0.1}, {"date": "2026-08-01", "r": -0.1}]
        point, lo, hi = cluster_bootstrap_pf(rows, "r", n_boot=100, seed=1)
        self.assertIsNone(lo)
        self.assertIsNone(hi)


import sqlite3
import tempfile

from src.spread_surface import Cell, SpreadSurface

from scripts.reprice_single_leg_book import (
    count_multi_leg_refused,
    dollar_scale_factor,
    fetch_single_leg_rows,
    reprice_row,
)


def _make_ledger(path, trades):
    """trades: list of dicts with keys entry_id, ticker, strike, expiration,
    date, type, status, entry_delta, strategy_name, net_credit, entry_price,
    exit_price, exit_reason, pnl_pct, pnl_usd, capital_at_risk, quantity."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE trades (
        entry_id INTEGER, ticker TEXT, strike REAL, expiration TEXT,
        date TEXT, type TEXT, status TEXT, entry_delta REAL,
        strategy_name TEXT, net_credit REAL, entry_price REAL,
        exit_price REAL, exit_reason TEXT, pnl_pct REAL, pnl_usd REAL,
        capital_at_risk REAL, quantity REAL, duplicate_of INTEGER)""")
    for t in trades:
        con.execute(
            "INSERT INTO trades (entry_id, ticker, strike, expiration, date, "
            "type, status, entry_delta, strategy_name, net_credit, "
            "entry_price, exit_price, exit_reason, pnl_pct, pnl_usd, "
            "capital_at_risk, quantity, duplicate_of) VALUES "
            "(:entry_id,:ticker,:strike,:expiration,:date,:type,:status,"
            ":entry_delta,:strategy_name,:net_credit,:entry_price,"
            ":exit_price,:exit_reason,:pnl_pct,:pnl_usd,:capital_at_risk,"
            ":quantity,:duplicate_of)",
            {**{"duplicate_of": None}, **t},
        )
    con.commit()
    con.close()


def _make_archive(path, quotes):
    """quotes: list of dicts with keys symbol, strike, expiration, type,
    snap_date, bid, ask, open_interest."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE chain_snapshots (
        symbol TEXT, strike REAL, expiration TEXT, type TEXT,
        snap_date TEXT, bid REAL, ask REAL, open_interest REAL)""")
    for q in quotes:
        con.execute(
            "INSERT INTO chain_snapshots VALUES (:symbol,:strike,:expiration,"
            ":type,:snap_date,:bid,:ask,:open_interest)", q)
    con.commit()
    con.close()


_SHORT_PUT = dict(
    entry_id=1, ticker="AAPL", strike=150.0, expiration="2026-09-20",
    date="2026-09-01", type="put", status="CLOSED", entry_delta=-0.20,
    strategy_name="Short Put", net_credit=None, entry_price=2.00,
    exit_price=1.00, exit_reason="Take Profit (50% @ 12d)", pnl_pct=0.45,
    pnl_usd=90.0, capital_at_risk=800.0, quantity=1.0,
)

_LONG_CALL = dict(
    entry_id=2, ticker="MSFT", strike=400.0, expiration="2026-10-01",
    date="2026-09-02", type="call", status="CLOSED", entry_delta=0.30,
    strategy_name="Long Call", net_credit=None, entry_price=3.00,
    exit_price=4.50, exit_reason="Take Profit (50%)", pnl_pct=0.48,
    pnl_usd=144.0, capital_at_risk=300.0, quantity=1.0,
)

_BULL_PUT = dict(
    entry_id=3, ticker="SPY", strike=440.0, expiration="2026-09-20",
    date="2026-09-01", type="put", status="CLOSED", entry_delta=-0.20,
    strategy_name="Bull Put", net_credit=1.40, entry_price=1.40,
    exit_price=0.60, exit_reason="Take Profit (60%)", pnl_pct=0.55,
    pnl_usd=55.0, capital_at_risk=360.0, quantity=1.0,
)


class FetchSingleLegRowsTests(unittest.TestCase):
    def test_only_single_leg_closed_rows_returned(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = f"{d}/ledger.db"
            archive = f"{d}/archive.db"
            _make_ledger(ledger, [_SHORT_PUT, _LONG_CALL, _BULL_PUT])
            _make_archive(archive, [])
            rows = fetch_single_leg_rows(ledger, archive)
            self.assertEqual({r["entry_id"] for r in rows}, {1, 2})

    def test_real_open_interest_joined_when_archive_matches(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = f"{d}/ledger.db"
            archive = f"{d}/archive.db"
            _make_ledger(ledger, [_SHORT_PUT])
            _make_archive(archive, [{
                "symbol": "AAPL", "strike": 150.0, "expiration": "2026-09-20",
                "type": "put", "snap_date": "2026-09-01", "bid": 1.90,
                "ask": 2.10, "open_interest": 340.0,
            }])
            rows = fetch_single_leg_rows(ledger, archive)
            self.assertEqual(rows[0]["open_interest"], 340.0)

    def test_no_archive_match_leaves_open_interest_none(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = f"{d}/ledger.db"
            archive = f"{d}/archive.db"
            _make_ledger(ledger, [_SHORT_PUT])
            _make_archive(archive, [])
            rows = fetch_single_leg_rows(ledger, archive)
            self.assertIsNone(rows[0]["open_interest"])


class CountMultiLegRefusedTests(unittest.TestCase):
    def test_counts_only_net_credit_rows(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = f"{d}/ledger.db"
            _make_ledger(ledger, [_SHORT_PUT, _LONG_CALL, _BULL_PUT])
            self.assertEqual(count_multi_leg_refused(ledger), 1)


class DollarScaleFactorTests(unittest.TestCase):
    def test_recovers_scale_from_booked_row(self):
        # pnl_usd = entry_price * pnl_pct * scale  =>  scale = pnl_usd / (entry_price * pnl_pct)
        factor = dollar_scale_factor(entry_price=2.00, booked_pct=0.45,
                                     booked_pnl_usd=90.0, quantity=1.0)
        self.assertAlmostEqual(factor, 100.0)  # 90 / (2.00*0.45) = 100

    def test_falls_back_to_quantity_times_multiplier_when_pct_is_zero(self):
        factor = dollar_scale_factor(entry_price=2.00, booked_pct=0.0,
                                     booked_pnl_usd=0.0, quantity=2.0)
        self.assertAlmostEqual(factor, 200.0)


class RepriceRowTests(unittest.TestCase):
    def test_short_put_repriced_lower_than_gross(self):
        surface = SpreadSurface(
            {(0, 1, 2): Cell(n=50, rel_half_spread=0.03, median_depth=20)},
            {"fit_date": "2026-09-01"},
        )
        row = dict(_SHORT_PUT, open_interest=500.0,
                  dte=19.0, abs_delta=0.20)
        out = reprice_row(row, surface)
        self.assertTrue(out["oi_known"])
        self.assertLess(out["repriced_pct_central"], out["gross_pct"])
        self.assertEqual(out["repriced_pct_central"],
                         out["repriced_pct_conservative"])  # OI known: one number

    def test_unknown_oi_reports_both_central_and_conservative(self):
        surface = SpreadSurface(
            {(0, 1, 2): Cell(n=50, rel_half_spread=0.03, median_depth=20)},
            {"fit_date": "2026-09-01"},
        )
        row = dict(_LONG_CALL, open_interest=None, dte=29.0, abs_delta=0.30)
        out = reprice_row(row, surface)
        self.assertFalse(out["oi_known"])
        self.assertLessEqual(out["repriced_pct_conservative"],
                             out["repriced_pct_central"])


if __name__ == "__main__":
    unittest.main()
