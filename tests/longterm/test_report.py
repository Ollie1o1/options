"""Tests for the desk-kit holdings report (pure render over fixture sidecar)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import fills as F
from src.longterm import plan as P
from src.longterm import report as R
from src.longterm.zones import Snapshot


def sidecar():
    closes = [700.0 + i for i in range(252)]
    return {
        "meta": {"sidecar": "2026-07-20.json", "generated": "2026-07-20T14:00:00"},
        "cash": {"pool": 5000.0, "deployed": 1500.0, "remaining": 3500.0},
        "book_value": 1697.9, "unrealized_pnl": 197.9,
        "names": [{
            "ticker": "MU", "thesis": "<script>alert(1)</script>",
            "spot": 848.95, "state": "NEAR", "drawdown_pct": -32.4,
            "sigma_dist": 0.98, "next_level": 750.0,
            "ladder": [
                {"level": 750.0, "size_usd": 2000.0, "filled": False},
                {"level": 650.0, "size_usd": 1750.0, "filled": True},
            ],
            "held": {"shares": 2.0, "avg_price": 750.0, "cost": 1500.0},
            "closes": closes, "ma50": closes[-50:], "ma200": closes[-200:],
            "labels": [f"2025-{(i % 12) + 1:02d}-01" for i in range(252)],
        }],
    }


class TestRender(unittest.TestCase):
    def test_page_contains_the_essentials(self):
        html = R.render(sidecar())
        self.assertIn("MU", html)
        self.assertIn("NEAR", html)
        self.assertIn("750", html)
        self.assertIn("3,500", html)          # remaining cash KPI
        self.assertIn("<svg", html)           # price chart rendered

    def test_thesis_is_escaped(self):
        html = R.render(sidecar())
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_empty_plan_renders_placeholder(self):
        data = sidecar()
        data["names"] = []
        html = R.render(data)
        self.assertIn("empty", html.lower())


class TestWrite(unittest.TestCase):
    def test_write_report_writes_html_json_and_hub(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "reports", "holdings")
            html_path, json_path = R.write_report(out_dir=out_dir, data=sidecar())
            self.assertTrue(os.path.exists(html_path))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "latest.html")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "reports", "index.html")))


class TestBuildPartialFetchFailure(unittest.TestCase):
    """A held ticker whose snapshot fetch fails (realistic transient
    yfinance error — data.fetch_snapshots simply omits it from the returned
    dict) must be excluded from BOTH sides of the P&L calc, not just the
    market-value side. Regression for a bug where cost was summed
    unconditionally over every held ticker while book_value only summed
    tickers with a live snapshot, silently understating unrealized_pnl by
    the failed ticker's full cost basis."""

    def test_cost_excludes_positions_with_failed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            db_path = os.path.join(tmp, "longterm.db")
            plan = P.Plan(cash_pool_usd=1000.0, names=[
                P.PlanName("A", [P.Tranche(100.0, 1.0)], thesis="a"),
                P.PlanName("B", [P.Tranche(50.0, 1.0)], thesis="b"),
            ])
            P.save_plan(plan, plan_path)
            F.record_fill("A", 100.0, 2.0, 100.0, db_path=db_path)  # cost 200
            F.record_fill("B", 50.0, 3.0, 50.0, db_path=db_path)    # cost 150

            snap_a = Snapshot(ticker="A", spot=150.0, high_52w=160.0,
                              low_52w=90.0, ma200=120.0, daily_sigma=0.02,
                              closes=[100.0 + i for i in range(40)])
            # B "fails" — absent from the returned dict, exactly like
            # data.fetch_snapshots does on a per-ticker exception.
            with mock.patch("src.longterm.data.fetch_snapshots",
                            return_value={"A": snap_a}):
                data = R.build(plan_path=plan_path, db_path=db_path)

            self.assertEqual(data["book_value"], 300.0)      # A only: 2 * 150
            # A's unrealized P&L is 300 - 200 = 100. If B's $150 cost leaked
            # into the total with no matching market value, this would be
            # -50 instead — a phantom $150 loss on a position that simply
            # has no price data this run.
            self.assertEqual(data["unrealized_pnl"], 100.0)


if __name__ == "__main__":
    unittest.main()
