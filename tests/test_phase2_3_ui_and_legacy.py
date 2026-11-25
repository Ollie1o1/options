"""Deterministic offline unit & pilot tests for Phase 2 (TUI) and Phase 3 (Legacy CLI & Web Dashboard)."""
from __future__ import annotations

import datetime as dt
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from textual.widgets import DataTable, Input, Static, TabbedContent

from src import cli_display as cd
from src import dashboard as dash
from src import settings as user_settings
from src.radar.live_options import refresh_local_prices
from src.radar.playbooks import PlaybookMeta, list_playbooks
from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView, VehicleComparison
from src.tui.app import DeskApp, filter_opportunities
from src.tui.state import DeskDataProvider
from src.tui.widgets import format_workbench_visual_markup


def _seed_ohlcv_db(db_path: str, symbol: str, closes: List[float], end_date: str = "2026-11-24") -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """CREATE TABLE IF NOT EXISTS ohlcv (
                ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, adj_close REAL, volume REAL,
                PRIMARY KEY (ticker, date)
            )"""
        )
        end = dt.date.fromisoformat(end_date)
        start = end - dt.timedelta(days=len(closes) - 1)
        rows = []
        for i, c in enumerate(closes):
            d = (start + dt.timedelta(days=i)).isoformat()
            rows.append((symbol, d, c, c * 1.01, c * 0.99, c, c, 1_000_000.0))
        con.executemany(
            "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def _seed_paper_trades_db(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """CREATE TABLE trades (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, ticker TEXT, strategy_name TEXT, expiration TEXT,
                strike REAL, type TEXT, entry_price REAL, quantity INTEGER, status TEXT
            )"""
        )
        con.execute(
            "INSERT INTO trades (date, ticker, strategy_name, expiration, strike, type, entry_price, quantity, status) "
            "VALUES ('2026-11-20', 'SPY', 'Bull Put', '2027-02-19', 560.0, 'put', 3.25, 1, 'OPEN')"
        )
        con.commit()
    finally:
        con.close()


def _seed_longterm_plan(plan_path: str, lt_db: str) -> None:
    from src.longterm.fills import record_fill

    plan = {
        "cash_pool_usd": 1000.0,
        "names": [
            {
                "ticker": "RMBS",
                "thesis": "Memory interface leader",
                "tranches": [
                    {"level": 50.0, "weight": 0.5},
                    {"level": 42.0, "weight": 0.5},
                ],
            }
        ],
    }
    Path(plan_path).write_text(json.dumps(plan))
    record_fill("RMBS", 50.0, 10.0, 49.80, fill_date="2026-11-10", note="tranche 1", db_path=lt_db)


class Phase2FakeProvider(DeskDataProvider):
    """Offline DeskDataProvider backed by temp DBs for Book/Fill/Lab + deterministic Radar."""

    def get_radar(self, *, refresh: bool = False) -> RadarSnapshot:
        return RadarSnapshot(
            generated_at="2026-11-24T20:00:00Z",
            regime_posture="RISK_ON",
            vix=16.0,
            vix_3m=18.0,
            vix_term="CONTANGO",
            spy_last=595.0,
            spy_chg_1d_pct=0.4,
            qqq_chg_1d_pct=0.6,
            next_macro_event="NFP 2026-12-04",
            kalshi_macro=[{"label": "Fed Hold", "prob_pct": 82.0, "ticker": "KXFEDDECISION"}],
            core_allocation=[
                {"ticker": "SMH", "rank": 1, "target_weight": 0.30, "vol_scale": 0.90, "spot": 250.0, "mom_12_1_pct": 25.0},
            ],
            opportunities=[
                Opportunity(
                    ticker="SMH", asset_class="etf", playbook_id="vol_managed_outlook",
                    playbook_label="Vol-Managed Sector Rotation", evidence_tier="VALIDATED",
                    direction="LONG", spot=250.0, chg_1d_pct=1.0, chg_5d_pct=2.0,
                    horizon_label="21-63d", setup_summary="Rank #1 · Weight 30%",
                    badges=("TFSA_OK",), risk_flags=(),
                    friction_label="0.06% etf", friction_value=0.0006, confluence_count=2,
                ),
                Opportunity(
                    ticker="AAPL", asset_class="stock", playbook_id="equity_pullback",
                    playbook_label="Uptrend Pullback (Shares)", evidence_tier="LEAD",
                    direction="LONG", spot=225.0, chg_1d_pct=-1.5, chg_5d_pct=-3.0,
                    horizon_label="3-10d", setup_summary="RSI2=4.2 · >200MA",
                    badges=(">200MA",), risk_flags=("EARNINGS_4D",),
                    friction_label="0.10% stk", friction_value=0.0010, confluence_count=1,
                ),
                Opportunity(
                    ticker="SPY", asset_class="option", playbook_id="etf_wide_spread",
                    playbook_label="ETF Wide Spread", evidence_tier="LEAD",
                    direction="LONG", spot=595.0, chg_1d_pct=0.4, chg_5d_pct=1.1,
                    horizon_label="60-120d", setup_summary="Bull Put 560/530 · cr $4.20",
                    badges=("TEST_O1",), risk_flags=(),
                    friction_label="8.0% of cr", friction_value=0.08, confluence_count=1,
                ),
                Opportunity(
                    ticker="RMBS", asset_class="stock", playbook_id="holdings_buy_zone",
                    playbook_label="Holdings Buy Zone", evidence_tier="CONTEXT",
                    direction="LONG", spot=43.0, chg_1d_pct=-2.0, chg_5d_pct=-5.0,
                    horizon_label="long-term", setup_summary="Near tranche $42",
                    badges=("PLAN",), risk_flags=(),
                    friction_label="0.10% stk", friction_value=0.0010, confluence_count=1,
                ),
            ],
            degraded_scanners=[],
            from_cache=not refresh,
        )

    def get_dossier(self, ticker: str, *, fetch_chain: bool = False) -> TickerDossier:
        tk = ticker.upper()
        spot = 595.0 if tk == "SPY" else 225.0
        return TickerDossier(
            ticker=tk,
            as_of="2026-11-24T20:00:00Z",
            spot=spot,
            chg_1d_pct=0.4,
            chg_5d_pct=1.2,
            chg_21d_pct=3.5,
            mom_12_1_pct=20.0,
            dist_sma20_pct=1.0,
            dist_sma50_pct=3.0,
            dist_sma200_pct=9.0,
            rsi_2=35.0,
            rsi_14=55.0,
            rv_20_ann=18.0,
            atm_iv=20.0,
            iv_vs_rv_ratio=1.11,
            skew_25d_vp=4.2,
            chain_source="archive",
            rnd_source="svi",
            supports=[{"label": "50d MA", "level": spot * 0.96, "pct": -0.04}],
            resistances=[{"label": "20d high", "level": spot * 1.03, "pct": 0.03}],
            sector_etf="XLK",
            sector_posture="FAVOR",
            next_macro_event="NFP 2026-12-04",
            implied_move_1s_pct=5.1,
            rnd_prob_up_5pct=0.24,
            rnd_prob_dn_5pct=0.19,
            closes_60d=[spot * (0.94 + 0.001 * i) for i in range(60)],
        )

    def get_live_etf_options(self) -> Tuple[List[Opportunity], List[str]]:
        return (
            [
                Opportunity(
                    ticker="SPY", asset_class="option", playbook_id="etf_wide_spread",
                    playbook_label="60-120d Wide Bull Put", evidence_tier="LEAD",
                    direction="LONG", spot=595.0, chg_1d_pct=0.4, chg_5d_pct=1.1,
                    horizon_label="93d", setup_summary="LIVE Bull Put 560/530 · Mid $4.40",
                    badges=("LIVE_CHAIN", "TEST_O1"), risk_flags=("LIMIT_ONLY_NO_MKT",),
                    friction_label="3.1% cr", friction_value=0.031, confluence_count=1,
                ),
                Opportunity(
                    ticker="QQQ", asset_class="option", playbook_id="etf_wide_spread",
                    playbook_label="60-120d Wide Bull Put", evidence_tier="LEAD",
                    direction="LONG", spot=510.0, chg_1d_pct=0.6, chg_5d_pct=1.2,
                    horizon_label="90d", setup_summary="LIVE Bull Put 480/455 · Mid $3.80",
                    badges=("LIVE_CHAIN", "TEST_O1"), risk_flags=("LIMIT_ONLY_NO_MKT",),
                    friction_label="7.2% cr", friction_value=0.072, confluence_count=1,
                ),
            ],
            [],
        )

    def get_express(self, view: DetailedView, *, spot: float, iv: float = 0.25, allow_live: bool = False) -> List[VehicleComparison]:
        return compare_vehicles(view, spot=spot, iv=iv, chain_db="/nonexistent/chain.db", allow_live=False)

    def get_playbooks(self) -> List[PlaybookMeta]:
        return list_playbooks(include_closed=True)


class TestPhase2TuiCockpit(unittest.IsolatedAsyncioTestCase):
    async def test_filter_presets_visual_logfill_book_and_lab(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ohlcv_db = str(Path(td) / "ohlcv.db")
            picks_db = str(Path(td) / "own_picks.db")
            fills_db = str(Path(td) / "fills.db")
            paper_db = str(Path(td) / "paper.db")
            lt_db = str(Path(td) / "longterm.db")
            plan_path = str(Path(td) / "longterm_plan.json")

            # Seed AAPL closing above its 5-day SMA so equity_pullback triggers EXIT >SMA5
            _seed_ohlcv_db(ohlcv_db, "AAPL", [200.0, 201.0, 202.0, 203.0, 204.0, 215.0])
            _seed_ohlcv_db(ohlcv_db, "RMBS", [45.0, 44.5, 44.0, 43.8, 43.5, 43.0])
            _seed_paper_trades_db(paper_db)
            _seed_longterm_plan(plan_path, lt_db)

            provider = Phase2FakeProvider(
                ohlcv_db=ohlcv_db,
                own_picks_db=picks_db,
                fills_db=fills_db,
                paper_db=paper_db,
                longterm_db=lt_db,
                plan_path=plan_path,
            )
            # Pre-log one pullback pick on AAPL @ $205 so Book shows live P&L + EXIT >SMA5
            provider.log_stock_pick(
                ticker="AAPL",
                price=205.0,
                shares=5,
                playbook_id="equity_pullback",
                notes="test pullback",
            )

            app = DeskApp(provider=provider, initial_tab="radar")
            async with app.run_test(size=(130, 40)) as pilot:
                await pilot.pause()

                # 1. Ribbon includes Kalshi macro tag
                ribbon = str(app.query_one("#market-ribbon", Static).render())
                self.assertIn("Kalshi Fed Hold 82%", ribbon)

                # 2. Clickable Radar Filter Presets
                rt = app.query_one("#radar-table", DataTable)
                self.assertEqual(rt.row_count, 4)

                await pilot.click("#preset-cln")
                await pilot.pause()
                # AAPL has EARNINGS_4D risk flag, so Clean keeps 3 rows
                self.assertEqual(rt.row_count, 3)
                self.assertNotIn("AAPL", [o.ticker for o in app.filtered_opps])

                await pilot.click("#preset-act")
                await pilot.pause()
                # Proven+Lead excludes RMBS (CONTEXT), keeping 3 rows
                self.assertEqual(rt.row_count, 3)
                self.assertNotIn("RMBS", [o.ticker for o in app.filtered_opps])

                await pilot.click("#preset-opt")
                await pilot.pause()
                self.assertEqual(rt.row_count, 1)
                self.assertEqual(app.filtered_opps[0].ticker, "SPY")

                await pilot.click("#preset-all")
                await pilot.pause()
                self.assertEqual(rt.row_count, 4)

                # 3. Workbench visual bar chart rendered
                app.open_workbench("SPY")
                await pilot.pause()
                vis = str(app.query_one("#wb-visual-box", Static).render())
                self.assertIn("Market RND (30d ±5%)", vis)
                self.assertIn("Edge vs Cost", vis)

                # 4. Log Fill Modal (`f`) saves real fill attempt to temp fills.db
                await pilot.press("f")
                await pilot.pause()
                await pilot.click("#lf-confirm")
                await pilot.pause()
                health_after = provider.get_health()
                self.assertEqual(health_after["fills"]["total_attempts"], 1)
                self.assertEqual(health_after["fills"]["total_filled"], 1)

                # 5. [4] BOOK shows live EXIT >SMA5 alert, paper options, and long-term plan
                await pilot.press("4")
                await pilot.pause()
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "book")
                sc_txt = str(app.query_one("#book-scorecard-box", Static).render())
                self.assertIn("EXIT ALERTS:", sc_txt)
                self.assertIn("EXIT >SMA5", sc_txt)
                extra_txt = str(app.query_one("#book-extra-box", Static).render())
                self.assertIn("OPEN PAPER OPTIONS (1)", extra_txt)
                self.assertIn("LONG-TERM PLAN:", extra_txt)
                self.assertIn("RMBS", extra_txt)

                # 6. [5] LAB shows OHLCV count & fill attempt count
                await pilot.press("5")
                await pilot.pause()
                lab_txt = str(app.query_one("#lab-health-box", Static).render())
                self.assertIn("OHLCV: 2 syms", lab_txt)
                self.assertIn("1/1", lab_txt)

                # 7. Live ETF Spreads (`o`) replaces archived SPY without DuplicateKey crash
                app.action_live_options()
                await pilot.pause(0.2)
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "radar")
                self.assertEqual(app.query_one("#radar-filter-input", Input).value, "option")
                opt_tickers = [o.ticker for o in app.filtered_opps]
                self.assertEqual(opt_tickers.count("SPY"), 1)
                self.assertIn("QQQ", opt_tickers)
                spy_opt = next(o for o in app.filtered_opps if o.ticker == "SPY")
                self.assertIn("LIVE_CHAIN", spy_opt.badges)

    def test_filter_tokens_actionable_and_clean(self) -> None:
        opps = Phase2FakeProvider().get_radar().opportunities
        self.assertEqual(len(filter_opportunities(opps, "actionable")), 3)
        self.assertEqual(len(filter_opportunities(opps, "clean")), 3)
        self.assertEqual(len(filter_opportunities(opps, "actionable clean")), 2)


class TestPhase3LegacyAndDashboard(unittest.TestCase):
    def test_print_report_numbers_board_order_not_quality_score(self) -> None:
        # Row 0 has lower quality_score (0.50) than Row 1 (0.95), but arrives first on the gated board.
        import sys as _sys

        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in _sys.path:
            _sys.path.insert(0, scripts_dir)
        from ui_preview import FIXTURE_PICK

        df = pd.DataFrame(
            [
                dict(FIXTURE_PICK, symbol="FIRST", quality_score=0.50, ev_per_contract=42.0),
                dict(FIXTURE_PICK, symbol="SECOND", quality_score=0.95, ev_per_contract=25.0),
            ]
        )
        captured: List[pd.DataFrame] = []

        def _capture_table(frame, *args, **kwargs):
            captured.append(frame.copy())

        buf = io.StringIO()
        with patch.object(cd, "print_comparison_table", side_effect=_capture_table), \
             patch.object(cd, "print_order_ticket"), \
             patch("sys.stdin.isatty", return_value=False), \
             redirect_stdout(buf):
            cd.print_report(df, 182.40, 0.045, 2, 30, 60, mode="Discovery scan")

        self.assertEqual(len(captured), 1)
        tbl = captured[0]
        self.assertEqual(list(tbl["symbol"]), ["FIRST", "SECOND"])
        self.assertEqual(list(tbl["_rank"]), [1, 2])

    def test_format_decision_zone_includes_walk_away_limit_line(self) -> None:
        debit_row = pd.Series(
            {
                "symbol": "SPY",
                "type": "call",
                "strike": 600.0,
                "premium": 5.00,
                "bid": 4.90,
                "ask": 5.10,
                "ev_per_contract": 65.0,
                "strategy_name": "Long Call",
            }
        )
        lines = cd.format_decision_zone(debit_row)
        joined = "\n".join(lines)
        self.assertIn("LIMIT", joined)
        self.assertIn("mid $5.00", joined)
        self.assertIn("abort $", joined)

        credit_row = pd.Series(
            {
                "symbol": "SPY",
                "type": "put",
                "strike": 560.0,
                "premium": 3.50,
                "bid": 3.40,
                "ask": 3.60,
                "ev_per_contract": 40.0,
                "strategy_name": "Short Put",
            }
        )
        clines = cd.format_decision_zone(credit_row)
        cjoined = "\n".join(clines)
        self.assertIn("LIMIT", cjoined)
        self.assertIn("mid $3.50", cjoined)

    def test_settings_persistence_and_unknown_key_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "settings.json")
            with patch.object(user_settings, "_SETTINGS_PATH", p):
                self.assertIsNone(user_settings.get_setting("account_size"))
                user_settings.set_setting("account_size", 12500.0)
                user_settings.set_setting("default_dte_preset", "wide_etf")
                user_settings.set_setting("unknown_key", "ignored")
                self.assertEqual(user_settings.get_setting("account_size"), 12500.0)
                self.assertEqual(user_settings.get_setting("default_dte_preset"), "wide_etf")
                self.assertIsNone(user_settings.get_setting("unknown_key"))

    def test_dashboard_plotly_builders_and_payoff(self) -> None:
        dos = Phase2FakeProvider().get_dossier("SPY")
        fig_px = dash.build_dossier_price_figure(dos)
        self.assertIsInstance(fig_px, go.Figure)
        self.assertEqual(len(fig_px.data), 1)

        view = DetailedView(
            symbol="SPY",
            direction="BULLISH",
            target_move_pct=0.06,
            horizon_days=30,
            risk_budget_usd=500.0,
            account_type="tfsa",
        )
        fig_rnd, src_lbl = dash.build_rnd_comparison_figure(
            595.0, 0.20, view, chain_db="/nonexistent/chain.db"
        )
        self.assertIsInstance(fig_rnd, go.Figure)
        self.assertEqual(len(fig_rnd.data), 2)
        self.assertIn("lognormal", src_lbl)

        vehicles = compare_vehicles(view, spot=595.0, iv=0.20, chain_db="/nonexistent/chain.db")
        fig_ev = dash.build_vehicle_edge_figure(vehicles, 500.0)
        self.assertIsInstance(fig_ev, go.Figure)
        self.assertEqual(len(fig_ev.data), 1)

        vis_txt = format_workbench_visual_markup(vehicles, dos, view)
        self.assertIn("Market RND", vis_txt)
        self.assertIn("Edge vs Cost", vis_txt)

        xs_l, pnl_l = dash.calculate_payoff({"strike": 100.0, "premium": 5.0, "type": "call", "strategy_name": "Long Call"}, 100.0)
        self.assertEqual(len(xs_l), 100)
        self.assertAlmostEqual(min(pnl_l), -500.0, places=1)
        self.assertGreater(max(pnl_l), 1000.0)

        xs_s, pnl_s = dash.calculate_payoff({"strike": 100.0, "premium": 3.0, "type": "put", "strategy_name": "Short Put"}, 100.0)
        self.assertAlmostEqual(max(pnl_s), 300.0, places=1)
        self.assertLess(min(pnl_s), -1000.0)

    def test_refresh_local_prices_skips_current_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "ohlcv.db")
            _seed_ohlcv_db(db_path, "SPY", [590.0, 595.0], end_date="2026-11-23")
            _seed_ohlcv_db(db_path, "AAPL", [220.0, 222.0], end_date="2026-11-20")

            calls: List[List[str]] = []

            def _fake_refresh(full_db, tickers, *, through=None, pause=0.15):
                calls.append(list(tickers))
                return {
                    "symbols": {t: {"status": "updated", "inserted": 2} for t in tickers},
                    "inserted": 2 * len(tickers),
                }

            extras = [f"SYM{i:02d}" for i in range(30)]
            with patch("src.breakout.refresh.refresh_universe", side_effect=_fake_refresh):
                res = refresh_local_prices(
                    db_path,
                    expand_universe=False,
                    extra_symbols=extras,
                    through="2026-11-23",
                )

            # SPY is already at 2026-11-23 -> skipped; AAPL + 30 extras = 31 symbols -> 2 chunks (25 + 6)
            self.assertEqual([len(c) for c in calls], [25, 6])
            self.assertNotIn("SPY", [t for c in calls for t in c])
            self.assertEqual(res["current"], 1)
            self.assertEqual(res["updated"], 31)
            self.assertEqual(res["failed"], 0)
            self.assertEqual(res["inserted_bars"], 62)

    def test_phase4_closed_playbook_and_launcher_web_entry(self) -> None:
        from src import launcher
        from src.radar.playbooks import list_playbooks

        all_pbs = {p.id: p for p in list_playbooks(include_closed=True)}
        active_pbs = {p.id: p for p in list_playbooks(include_closed=False)}
        self.assertIn("non_earnings_breakout", all_pbs)
        self.assertEqual(all_pbs["non_earnings_breakout"].evidence_tier, "CLOSED")
        self.assertNotIn("non_earnings_breakout", active_pbs)

        rows = launcher._menu_rows()
        self.assertTrue(any("[W]" in r and "WEB" in r for r in rows))
        self.assertIn("HELP", rows[-2])
        self.assertIn("QUIT", rows[-1])

        # Verify zero deprecated Streamlit use_container_width calls remain
        dash_src = (Path(__file__).resolve().parent.parent / "src" / "dashboard.py").read_text()
        self.assertNotIn("use_container_width", dash_src)

        # Verify Ctrl+C exiting [W] WEB returns cleanly to the launcher menu
        class _FakeProc:
            def __init__(self):
                self.terminated = False

            def wait(self, timeout=None):
                if not self.terminated:
                    raise KeyboardInterrupt()
                return 0

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True

        fake_proc = _FakeProc()
        with patch("sys.argv", ["run.py"]), \
             patch.object(launcher, "_show_menu", side_effect=["W", "Q"]) as show_mock, \
             patch("subprocess.Popen", return_value=fake_proc):
            launcher.main()
        self.assertTrue(fake_proc.terminated)
        self.assertEqual(show_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
