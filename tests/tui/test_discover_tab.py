"""Pilot tests for the [6] DISCOVER tab and the ④ Evidence Card screen — fully offline."""
from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

from textual.widgets import DataTable, Select, Static, TabbedContent

from src.radar.playbooks import PlaybookMeta, list_playbooks
from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.topdown.hub import build_hub
from src.topdown.synthetic import synthetic_ohlcv, write_cboe_dir
from src.tui.app import DeskApp
from src.tui.discover import EvidenceCardScreen
from src.tui.state import DeskDataProvider


def _snapshot() -> RadarSnapshot:
    return RadarSnapshot(
        generated_at="2026-11-24T21:00:00Z", regime_posture="NEUTRAL", vix=None, vix_3m=None,
        vix_term="UNKNOWN", spy_last=600.0, spy_chg_1d_pct=0.1, qqq_chg_1d_pct=0.2, next_macro_event="none",
        opportunities=[Opportunity(
            ticker="XOM", asset_class="stock", playbook_id="equity_pullback",
            playbook_label="Uptrend Pullback (Shares)", evidence_tier="LEAD", direction="LONG", spot=100.0,
            chg_1d_pct=-1.0, chg_5d_pct=-4.0, horizon_label="3-10d", setup_summary="RSI2=4",
        )],
        from_cache=True,
    )


def _plain(widget: Static) -> str:
    """What the reader sees in a Static, whether it holds markup or a Rich renderable."""
    import io

    from rich.console import Console

    content = widget.content
    if isinstance(content, str):
        from rich.text import Text

        content = Text.from_markup(content)
    buf = io.StringIO()
    Console(file=buf, width=140, no_color=True).print(content)
    return buf.getvalue()


class OfflineProvider(DeskDataProvider):
    def __init__(self, td: str) -> None:
        super().__init__(
            ohlcv_db=os.path.join(td, "eq.db"), cache_path=os.path.join(td, "radar.json"),
            own_picks_db=os.path.join(td, "picks.db"), fills_db=os.path.join(td, "fills.db"),
            chain_db=os.path.join(td, "chain.db"), paper_db=os.path.join(td, "paper.db"),
            longterm_db=os.path.join(td, "lt.db"), plan_path=os.path.join(td, "plan.json"),
        )
        self.td = td
        self.card_calls: List[str] = []

    def get_radar(self, *, refresh: bool = False) -> RadarSnapshot:
        return _snapshot()

    def get_health(self) -> Dict[str, Any]:
        return {"disk_free_gb": 50.0, "worst": "OK", "live_execution": False,
                "fills": {"total_attempts": 0, "total_filled": 0}, "readouts": []}

    def get_playbooks(self) -> List[PlaybookMeta]:
        return list_playbooks(include_closed=True)

    def get_dossier(self, ticker: str, *, fetch_chain: bool = False) -> TickerDossier:
        return TickerDossier(
            ticker=ticker.upper(), as_of="2026-11-24T21:00:00Z", spot=100.0, chg_1d_pct=-1.0, chg_5d_pct=-2.0,
            chg_21d_pct=1.0, mom_12_1_pct=15.0, dist_sma20_pct=-1.0, dist_sma50_pct=1.0, dist_sma200_pct=5.0,
            rsi_2=5.0, rsi_14=40.0, rv_20_ann=24.0, sector_etf="XLE",
            rnd_prob_up_5pct=0.2, rnd_prob_dn_5pct=0.2, implied_move_1s_pct=7.0,
        )

    def get_hub(self, *, refresh: bool = False) -> Any:
        if self._hub is None or refresh:
            cboe = write_cboe_dir(os.path.join(self.td, "cboe"))
            with mock.patch("src.radar.scanners._load_upcoming_earnings_map", return_value={"AAPL": 2}), \
                    mock.patch("src.radar.scanners._load_recent_post_earnings_crashes", return_value={}), \
                    mock.patch("src.radar.scanners._load_latest_rev_breadth", return_value={"XOM": 0.5}):
                self._hub = build_hub(ohlcv=synthetic_ohlcv(), snapshot=_snapshot(),
                                      fred_path=os.path.join(self.td, "fred.json"), cboe_dir=cboe,
                                      estimates_db=os.path.join(self.td, "est.db"))
        return self._hub

    def get_card(self, ticker: str) -> Any:
        self.card_calls.append(ticker)
        return self.get_hub().card(ticker, dossier=self.get_dossier(ticker), chain_db=self.chain_db,
                                   with_peer_iv=False)


class DiscoverTabPilot(unittest.IsolatedAsyncioTestCase):
    async def _wait_hub(self, app: DeskApp, pilot: Any) -> None:
        for _ in range(60):
            await pilot.pause(0.05)
            if app.hub is not None and not app._hub_loading:
                return
        self.fail("Discover hub never finished building")

    async def test_discover_funnel_board_lens_sector_and_card(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prov = OfflineProvider(td)
            app = DeskApp(provider=prov)
            async with app.run_test(size=(140, 44)) as pilot:
                await pilot.press("6")
                await self._wait_hub(app, pilot)
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "discover")

                market = _plain(app.query_one("#disc-market", Static))
                self.assertIn("MARKET", market)
                board = app.query_one("#disc-board", DataTable)
                self.assertEqual(board.row_count, 16)

                # default lens = PROVEN core rotation, rows = held ETFs
                lens_tbl = app.query_one("#disc-lens-table", DataTable)
                held = [r.etf for r in app.hub.board.rows if r.weight > 0]
                self.assertEqual(lens_tbl.row_count, len(held))
                self.assertIn("PROVEN", _plain(app.query_one("#disc-lens-head", Static)))

                # switch lens through the Select
                app.query_one("#disc-lens-select", Select).value = "landmines"
                await pilot.pause(0.1)
                self.assertEqual(app._disc_lens, "landmines")
                self.assertEqual(lens_tbl.row_count, 1)  # AAPL earnings in 2d
                self.assertIn("AVOID", _plain(app.query_one("#disc-lens-head", Static)))

                # drill into XLE from the board (Enter on its row)
                board.focus()
                xle_row = [r.etf for r in app.hub.board.rows].index("XLE")
                board.move_cursor(row=xle_row)
                await pilot.press("enter")
                await pilot.pause(0.1)
                self.assertEqual((app._disc_mode, app._disc_sector), ("sector", "XLE"))
                self.assertEqual(lens_tbl.row_count, 4)
                self.assertIn("XLE", _plain(app.query_one("#disc-lens-head", Static)))

                # Enter on a constituent opens the ④ Evidence Card full screen
                lens_tbl.focus()
                lens_tbl.move_cursor(row=0)
                await pilot.press("enter")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, EvidenceCardScreen) and prov.card_calls:
                        body = _plain(app.screen.query_one("#card-body", Static))
                        if "Verdict" in body:
                            break
                self.assertIsInstance(app.screen, EvidenceCardScreen)
                body = _plain(app.screen.query_one("#card-body", Static))
                self.assertIn("Where every number came from", body)
                await pilot.press("escape")
                await pilot.pause(0.05)
                self.assertNotIsInstance(app.screen, EvidenceCardScreen)

                # the lens picker was blanked on drill-down, so re-choosing the first lens returns to ③
                self.assertTrue(app.query_one("#disc-lens-select", Select).is_blank())
                app.query_one("#disc-lens-select", Select).value = "core_rotation"
                await pilot.pause(0.1)
                self.assertEqual(app._disc_mode, "lens")
                self.assertEqual(lens_tbl.row_count, len(held))

    async def test_card_w_key_opens_workbench(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prov = OfflineProvider(td)
            app = DeskApp(provider=prov)
            async with app.run_test(size=(140, 44)) as pilot:
                await pilot.press("6")
                await self._wait_hub(app, pilot)
                app.open_evidence_card("XOM")
                await pilot.pause(0.3)
                await pilot.press("w")
                await pilot.pause(0.2)
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "workbench")

    async def test_compact_terminal_still_renders_discover(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = DeskApp(provider=OfflineProvider(td))
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("6")
                await self._wait_hub(app, pilot)
                self.assertEqual(app.query_one("#disc-board", DataTable).row_count, 16)


if __name__ == "__main__":
    unittest.main()
