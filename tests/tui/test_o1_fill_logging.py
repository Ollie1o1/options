"""Pressing [f] on a live O1 spread pre-fills the SPREAD's quote and tags the fill 'O1'."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from textual.widgets import DataTable, Input, Static

from src.radar.schema import Opportunity, RadarSnapshot
from src.tui.app import DeskApp
from src.tui.widgets import LogFillModal
from tests.tui.test_desk_app import FakeProvider

O1_OPP = Opportunity(
    ticker="SPY", asset_class="option", playbook_id="etf_wide_spread",
    playbook_label="60-120d Wide Bull Put", evidence_tier="LEAD", direction="LONG",
    spot=697.0, chg_1d_pct=None, chg_5d_pct=None, horizon_label="93d",
    setup_summary="Bull Put 645/610 (5.0% wide) exp 2027-02-26", badges=("LIVE_CHAIN",),
    risk_flags=("LIMIT_ONLY_NO_MKT",), friction_label="6.8% cr", friction_value=0.068,
    extra={"expiration": "2027-02-26", "short_strike": 645.0, "long_strike": 610.0,
           "credit_cross": 4.60, "credit_mid": 4.85, "walk_away_limit": 4.69},
)


class O1Provider(FakeProvider):
    def get_radar(self, *, refresh: bool = False) -> RadarSnapshot:
        snap = super().get_radar(refresh=refresh)
        snap.opportunities = [O1_OPP] + list(snap.opportunities)
        return snap

    def get_health(self) -> Dict[str, Any]:
        h = super().get_health()
        h["o1_fills"] = {"attempts": 3, "filled": 2, "needed": 20, "target_k": 0.35, "fill_rate": 2 / 3,
                         "mean_k": 0.21, "fill_rate_flag": False, "verdict": "COLLECTING"}
        return h


class O1FillLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_fill_dialog_prefills_spread_quote_and_lab_shows_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = DeskApp(provider=O1Provider(str(Path(td) / "own_picks.db")))
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.pause()
                await pilot.press("5")
                await pilot.pause()
                lab = app.query_one("#lab-health-box", Static)
                from rich.console import Console

                con = Console(width=200, record=True)
                con.print(lab.render())
                text = con.export_text()
                self.assertIn("TEST O1 FILLS 2/20", text)
                self.assertIn("COLLECTING", text)

                await pilot.press("2")
                await pilot.pause()
                rt = app.query_one("#radar-table", DataTable)
                rt.focus()
                rt.move_cursor(row=0)
                await pilot.pause()
                self.assertEqual(app.focused_opp.playbook_id if app.focused_opp else None, "etf_wide_spread")
                app.action_log_fill()
                await pilot.pause()
                modal = app.screen
                self.assertIsInstance(modal, LogFillModal)
                self.assertTrue(modal.query_one("#lf-structure", Input).value.startswith("O1 Bull Put 645/610"))
                self.assertEqual(modal.query_one("#lf-bid", Input).value, "4.60")
                self.assertEqual(modal.query_one("#lf-ask", Input).value, "5.10")
                self.assertEqual(modal.query_one("#lf-limit", Input).value, "4.69")


if __name__ == "__main__":
    unittest.main()
