"""Deterministic offline Textual pilot tests for src.tui.DeskApp."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual.widgets import DataTable, Input, Static, Switch, TabbedContent

from src.radar.playbooks import PlaybookMeta, list_playbooks
from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView, VehicleComparison
from src.tui.app import DeskApp, filter_opportunities
from src.tui.state import DeskDataProvider
from src.tui.themes import THEME_ORDER


class FakeProvider(DeskDataProvider):
    """100% in-memory / temp-dir provider for deterministic Textual pilot tests."""

    def __init__(self, own_picks_db: str, *, worst_health: str = "STALE") -> None:
        super().__init__(own_picks_db=own_picks_db)
        self._worst = worst_health

    def get_radar(self, *, refresh: bool = False) -> RadarSnapshot:
        return RadarSnapshot(
            generated_at="2026-11-24T19:00:00Z",
            regime_posture="RISK_ON",
            vix=16.5,
            vix_3m=18.2,
            vix_term="CONTANGO",
            spy_last=595.40,
            spy_chg_1d_pct=0.45,
            qqq_chg_1d_pct=0.72,
            next_macro_event="NFP 2026-12-04",
            core_allocation=[
                {"ticker": "SMH", "rank": 1, "target_weight": 0.28, "vol_scale": 0.84, "spot": 250.0, "mom_12_1_pct": 24.5},
                {"ticker": "XLK", "rank": 2, "target_weight": 0.28, "vol_scale": 0.84, "spot": 230.0, "mom_12_1_pct": 19.2},
                {"ticker": "GLD", "rank": 3, "target_weight": 0.28, "vol_scale": 0.84, "spot": 245.0, "mom_12_1_pct": 21.0},
                {"ticker": "CASH", "rank": 0, "target_weight": 0.16, "vol_scale": 0.84, "spot": 1.0, "mom_12_1_pct": 0.0},
            ],
            opportunities=[
                Opportunity(
                    ticker="SMH", asset_class="etf", playbook_id="vol_managed_outlook",
                    playbook_label="Vol-Managed Sector Rotation", evidence_tier="VALIDATED",
                    direction="LONG", spot=250.0, chg_1d_pct=1.1, chg_5d_pct=2.4,
                    horizon_label="21-63d", setup_summary="Rank #1/16 · Weight 28%",
                    badges=("FAVOR:SMH", "TFSA_OK"), risk_flags=(),
                    friction_label="0.06% etf", friction_value=0.0006, confluence_count=2,
                ),
                Opportunity(
                    ticker="AAPL", asset_class="stock", playbook_id="equity_pullback",
                    playbook_label="Uptrend Pullback (Shares)", evidence_tier="LEAD",
                    direction="LONG", spot=225.0, chg_1d_pct=-1.8, chg_5d_pct=-3.5,
                    horizon_label="3-10d", setup_summary="RSI2=5.4 · +7.1% >200MA",
                    badges=(">200MA", "RSI2=5.4"), risk_flags=(),
                    friction_label="0.10% stk", friction_value=0.0010, confluence_count=1,
                ),
            ],
            degraded_scanners=[],
            from_cache=True,
        )

    def get_dossier(self, ticker: str, *, fetch_chain: bool = False) -> TickerDossier:
        tk = ticker.upper()
        spot = 225.0 if tk == "AAPL" else 250.0
        return TickerDossier(
            ticker=tk,
            as_of="2026-11-24T19:00:00Z",
            spot=spot,
            chg_1d_pct=-1.2,
            chg_5d_pct=-3.1,
            chg_21d_pct=2.5,
            mom_12_1_pct=18.4,
            dist_sma20_pct=-1.5,
            dist_sma50_pct=2.0,
            dist_sma200_pct=8.4,
            rsi_2=5.4,
            rsi_14=42.0,
            rv_20_ann=22.0,
            supports=[{"label": "50d MA", "level": spot * 0.97, "pct": -0.03}],
            resistances=[{"label": "20d high", "level": spot * 1.04, "pct": 0.04}],
            sector_etf="XLK",
            sector_posture="FAVOR",
            next_macro_event="NFP 2026-12-04",
            implied_move_1s_pct=6.3,
            rnd_prob_up_5pct=0.22,
            rnd_prob_dn_5pct=0.21,
            closes_60d=[spot * (0.95 + 0.001 * i) for i in range(60)],
        )

    def get_express(self, view: DetailedView, *, spot: float, iv: float = 0.25) -> List[VehicleComparison]:
        return compare_vehicles(view, spot=spot, iv=iv)

    def get_playbooks(self) -> List[PlaybookMeta]:
        return list_playbooks(include_closed=True)

    def get_health(self) -> Dict[str, Any]:
        return {
            "disk_free_gb": 5.5,
            "worst": self._worst,
            "live_execution": False,
            "fills": {"total_attempts": 0, "total_filled": 0},
            "readouts": ["2027-01-15: Revisions Arm B"],
        }


class TestDeskAppPilot(unittest.IsolatedAsyncioTestCase):
    async def test_wide_navigation_filter_workbench_quicklog_and_themes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            picks_db = str(Path(td) / "own_picks.db")
            provider = FakeProvider(picks_db, worst_health="STALE")
            app = DeskApp(provider=provider)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()

                # 1. Startup & Ribbon shows LAB STALE indicator
                ribbon = str(app.query_one("#market-ribbon", Static).render())
                self.assertIn("SPY", ribbon)
                self.assertIn("LAB STALE", ribbon)

                # 2. Tab navigation 1..5
                for key, expected in [("2", "radar"), ("3", "workbench"), ("4", "book"), ("5", "lab"), ("1", "today")]:
                    await pilot.press(key)
                    await pilot.pause()
                    self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, expected)

                # 3. Radar filtering & sorting
                await pilot.press("2")
                await pilot.pause()
                flt = app.query_one("#radar-filter-input", Input)
                flt.value = "pullback"
                await pilot.pause()
                rt = app.query_one("#radar-table", DataTable)
                self.assertEqual(rt.row_count, 1)
                self.assertEqual(app.filtered_opps[0].ticker, "AAPL")
                await pilot.press("s")
                await pilot.pause()

                # 4. Workbench drill-down + toggle 100sh switch adds Covered Call
                app.open_workbench("AAPL")
                await pilot.pause()
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "workbench")
                vt = app.query_one("#wb-vehicles-table", DataTable)
                n_before = vt.row_count
                sw = app.query_one("#wb-owns-switch", Switch)
                sw.value = True
                await pilot.pause()
                self.assertEqual(vt.row_count, n_before + 1)

                # 5. Quick-Log Modal writes 1 pick & updates Book
                await pilot.press("l")
                await pilot.pause()
                await pilot.click("#ql-confirm")
                await pilot.pause()
                book = provider.get_book()
                self.assertEqual(len(book["equity_picks"]), 1)
                self.assertEqual(book["equity_picks"][0]["ticker"], "AAPL")

                # 6. Theme switching across all 4 themes
                for name in THEME_ORDER:
                    app.apply_desk_theme(name)
                    await pilot.pause()
                    self.assertEqual(app.theme, name)

    async def test_narrow_terminal_reflow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provider = FakeProvider(str(Path(td) / "own_picks.db"), worst_health="OK")
            app = DeskApp(provider=provider, initial_tab="radar")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                self.assertTrue(app.screen.has_class("compact"))

    def test_micro_query_filter_tokens(self) -> None:
        opps = FakeProvider("/tmp/unused.db").get_radar().opportunities
        self.assertEqual(len(filter_opportunities(opps, "stock")), 1)
        self.assertEqual(len(filter_opportunities(opps, "validated")), 1)
        self.assertEqual(len(filter_opportunities(opps, "conf>=2")), 1)
        self.assertEqual(len(filter_opportunities(opps, "fric<0.08%")), 1)


if __name__ == "__main__":
    unittest.main()
