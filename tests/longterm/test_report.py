"""Tests for the desk-kit holdings report (pure render over fixture sidecar)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import report as R


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


if __name__ == "__main__":
    unittest.main()
