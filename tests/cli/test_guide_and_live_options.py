"""Offline unit tests for live 60-120d ETF spread scanner, plain-English coach, and HelpModal."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from src.cli.guide import (
    plain_english_action_plan,
    plain_english_traffic_light,
    plain_english_vehicle_verdict,
)
from src.cli.main import cli
from src.radar.live_options import scan_live_etf_wide_spreads
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView
from src.tui.app import DeskApp
from tests.tui.test_desk_app import FakeProvider


class TestGuideAndLiveOptions(unittest.IsolatedAsyncioTestCase):
    def test_scan_live_etf_wide_spreads_filters_friction_and_computes_limit(self) -> None:
        def _fake_fetch(sym: str):
            if sym == "SPY":
                # Tight spread: short 565P bid 6.20/ask 6.35, long 535P bid 2.10/ask 2.20
                # cr_mid = 4.125, cr_cross = 4.00, rt_spread = 0.25 -> 6.25% friction <= 25% -> PASS
                return {
                    "symbol": "SPY",
                    "spot": 600.0,
                    "expiration": "2027-02-19",
                    "dte": 87,
                    "puts": [
                        {"strike": 565.0, "bid": 6.20, "ask": 6.35},
                        {"strike": 535.0, "bid": 2.10, "ask": 2.20},
                    ],
                    "calls": [],
                }
            # Wide illiquid spread -> rejected (>25% friction)
            return {
                "symbol": sym,
                "spot": 100.0,
                "expiration": "2027-02-19",
                "dte": 87,
                "puts": [
                    {"strike": 94.0, "bid": 1.00, "ask": 1.60},
                    {"strike": 89.0, "bid": 0.40, "ask": 0.80},
                ],
                "calls": [],
            }

        opps, warns = scan_live_etf_wide_spreads(["SPY", "ILLIQ"], fetch_fn=_fake_fetch)
        self.assertEqual(len(warns), 0)
        self.assertEqual(len(opps), 1)
        self.assertEqual(opps[0].ticker, "SPY")
        self.assertEqual(opps[0].playbook_id, "etf_wide_spread")
        self.assertLess(opps[0].friction_value, 0.10)
        self.assertIn("Min Limit", opps[0].setup_summary)

    def test_plain_english_helpers_and_opt_guide_quit(self) -> None:
        prov = FakeProvider("/tmp/unused_picks.db")
        snap = prov.get_radar()
        plan = plain_english_action_plan(snap)
        self.assertEqual(len(plan), 3)
        self.assertIn("Core Portfolio", plan[1])

        dos = prov.get_dossier("AAPL")
        light, _reason = plain_english_traffic_light(dos)
        self.assertTrue(light.startswith("GREEN"))

        view = DetailedView(symbol="AAPL", direction="BULLISH", target_move_pct=0.08, horizon_days=30)
        rows = compare_vehicles(view, spot=225.0, iv=0.22)
        verdict = plain_english_vehicle_verdict(rows, view)
        self.assertTrue("BEST CHOICE" in verdict or "SKIP" in verdict)

        # Interactive opt guide handles ? then Q cleanly
        runner = CliRunner()
        res = runner.invoke(cli, ["--no-color", "guide"], input="?\nQ\n")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("PLAIN-ENGLISH CHEAT SHEET", res.output)

    async def test_tui_help_modal_and_verdict_banner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prov = FakeProvider(str(Path(td) / "picks.db"))
            app = DeskApp(provider=prov)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()
                # Open HelpModal with '?' and close with button
                await pilot.press("?")
                await pilot.pause()
                self.assertEqual(len(app.screen_stack), 2)
                await pilot.click("#help-close")
                await pilot.pause()
                self.assertEqual(len(app.screen_stack), 1)
                # Activate "Analyze a Stock" button on TODAY -> jumps to WORKBENCH with verdict box
                app.query_one("#act-workbench").press()
                await pilot.pause(0.3)
                self.assertEqual(app.query_one("#main-tabs").active, "workbench")
                verdict = str(app.query_one("#wb-verdict-box").render())
                self.assertTrue("BEST CHOICE" in verdict or "SKIP" in verdict)


if __name__ == "__main__":
    unittest.main()
