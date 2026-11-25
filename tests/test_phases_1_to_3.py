"""Offline unit, CLI & TUI pilot tests covering Phases 1, 2, and 3.

Zero network calls; all writes stay inside `tempfile.TemporaryDirectory()`.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import numpy as np
from click.testing import CliRunner
from textual.widgets import Input, Static, TabbedContent

from src.alpha.edgar_events import (
    compute_xbrl_buyback_quality,
    load_cached_edgar_events,
    parse_efts_hits,
    refresh_edgar_events,
    save_edgar_events,
)
from src.alpha.macro_plumbing import (
    MacroPlumbingSnapshot,
    compute_fed_net_liquidity,
    load_macro_plumbing,
    parse_cftc_cot,
    parse_fred_csv,
    parse_polymarket_macro,
    refresh_macro_plumbing,
)
from src.alpha.options_for_shares import (
    compute_dealer_gex,
    scan_option_signals_for_shares,
)
from src.alpha.premortem import build_premortem
from src.cli.main import cli
from src.equity_playbooks.ghost_race import compute_ghost_race
from src.own_picks import close_equity_pick, list_equity_picks, log_equity_pick
from src.profile import (
    UserProfile,
    apply_rebalance_to_profile,
    estimate_cash_yield_annual,
    import_wealthsimple_csv,
    load_profile,
    rebalance_due_status,
    save_profile,
)
from src.radar.schema import TickerDossier
from src.structure.types import DetailedView
from src.topdown.lenses import LENS_BY_ID, LensContext, run_lens
from src.topdown.sectors import build_sector_board
from src.topdown.synthetic import synthetic_ohlcv
from src.tui.app import DeskApp
from src.tui.charts import (
    braille_ghost_race,
    braille_payoff_and_rnd,
    braille_price_chart,
    braille_sector_rrg,
    theme_palette,
    vix_curve_strip,
)
from src.tui.widgets import AlphaViewerModal, BloombergCommandModal, RebalanceModal
from src.web_terminal import TERMINAL_HTML


def _make_dossier(ticker: str, **kw: Any) -> TickerDossier:
    base: Dict[str, Any] = dict(
        ticker=ticker,
        as_of="2026-11-24",
        spot=100.0,
        chg_1d_pct=0.2,
        chg_5d_pct=0.8,
        chg_21d_pct=2.0,
        mom_12_1_pct=15.0,
        dist_sma20_pct=1.0,
        dist_sma50_pct=2.5,
        dist_sma200_pct=6.0,
        rsi_2=45.0,
        rsi_14=54.0,
        rv_20_ann=20.0,
    )
    base.update(kw)
    return TickerDossier(**base)


def _seed_vm_ohlcv_db(db_path: Path, n_days: int = 340) -> None:
    """Populate a self-contained SQLite OHLCV DB with SPY + all 16 Outlook ETFs + AAPL."""
    from src.outlook.vm_rotation import UNIVERSE as VM_UNIVERSE

    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE ohlcv (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, "
            "close REAL, adj_close REAL, volume REAL, PRIMARY KEY (ticker, date))"
        )
        d0 = dt.date(2025, 6, 1)
        dates: List[str] = []
        cur = d0
        while len(dates) < n_days:
            if cur.weekday() < 5:
                dates.append(cur.isoformat())
            cur += dt.timedelta(days=1)

        rng = np.random.default_rng(42)
        rows = []
        syms = ["SPY", "AAPL"] + list(VM_UNIVERSE)
        for idx, sym in enumerate(syms):
            # Give first few ETFs steady uptrends so VM_OUTLOOK_TOP3 stays invested
            drift = 0.0010 if idx in (2, 3, 4) else (0.0004 if sym in ("SPY", "AAPL") else -0.0002)
            px = 100.0 + idx * 5.0
            for d in dates:
                shock = float(rng.normal(drift, 0.009))
                px = max(10.0, px * (1.0 + shock))
                rows.append((sym, d, px, px * 1.01, px * 0.99, px, px, 2_000_000.0))
        con.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.commit()
    finally:
        con.close()


def _seed_chain_archive_db(db_path: Path) -> None:
    """Create a synthetic chain_archive.db with one BULL_FLOW and one CRASH_SMIRK symbol."""
    today = dt.date.today()
    snap = today.isoformat()
    exp = (today + dt.timedelta(days=35)).isoformat()
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            """CREATE TABLE chain_snapshots (
                symbol TEXT, snap_date TEXT, type TEXT, strike REAL, expiration TEXT,
                bid REAL, ask REAL, iv REAL, delta REAL, gamma REAL, open_interest REAL, spot REAL
            )"""
        )
        rows = [
            # BULL: ATM call IV (0.28) >> ATM put IV (0.24) => CW spread +4.0vp; positive GEX (heavy call OI)
            ("BULL", snap, "call", 100.0, exp, 4.20, 4.40, 0.28, 0.52, 0.04, 15000, 100.0),
            ("BULL", snap, "put", 100.0, exp, 3.80, 4.00, 0.24, -0.48, 0.04, 3000, 100.0),
            ("BULL", snap, "call", 105.0, exp, 2.00, 2.15, 0.27, 0.30, 0.03, 25000, 100.0),
            ("BULL", snap, "put", 93.0, exp, 1.50, 1.65, 0.31, -0.25, 0.03, 4000, 100.0),
            ("BULL", snap, "put", 90.0, exp, 0.90, 1.05, 0.32, -0.18, 0.02, 2000, 100.0),
            ("BULL", snap, "call", 110.0, exp, 0.90, 1.05, 0.26, 0.18, 0.02, 12000, 100.0),
            # BEAR: Put smirk = 0.39 - 0.23 = +16.0vp (>= 10vp) + negative GEX (heavy put OI)
            ("BEAR", snap, "call", 200.0, exp, 6.00, 6.20, 0.23, 0.51, 0.02, 2000, 200.0),
            ("BEAR", snap, "put", 200.0, exp, 7.00, 7.20, 0.27, -0.49, 0.02, 18000, 200.0),
            ("BEAR", snap, "put", 185.0, exp, 3.20, 3.40, 0.39, -0.25, 0.02, 30000, 200.0),
            ("BEAR", snap, "call", 215.0, exp, 2.10, 2.30, 0.22, 0.24, 0.015, 3000, 200.0),
            ("BEAR", snap, "put", 180.0, exp, 2.40, 2.60, 0.41, -0.19, 0.015, 14000, 200.0),
            ("BEAR", snap, "call", 220.0, exp, 1.20, 1.40, 0.22, 0.16, 0.01, 1000, 200.0),
        ]
        con.executemany(
            "INSERT INTO chain_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


class TestPhase1TfsaLoop(unittest.TestCase):
    def test_profile_save_load_cash_yield_and_due_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prof_p = str(Path(td) / "tfsa_profile.json")
            rates_p = str(Path(td) / "macro_rates_cache.json")
            Path(rates_p).write_text(json.dumps({"dgs3mo": 4.55, "as_of": {"DGS3MO": "2026-11-20"}}))

            p = UserProfile(tfsa_usd=15000.0, cash_etf="SGOV", drift_threshold_pct=5.0)
            save_profile(p, path=prof_p)
            loaded = load_profile(prof_p)
            self.assertEqual(loaded.tfsa_usd, 15000.0)
            self.assertEqual(loaded.cash_etf, "SGOV")

            y, lbl = estimate_cash_yield_annual(rates_cache=rates_p)
            self.assertAlmostEqual(y, 0.0455, places=4)
            self.assertIn("4.55%", lbl)

            due0, msg0 = rebalance_due_status(loaded, max_drift_pct=0.0, rebalance_recommended=False)
            self.assertTrue(due0)
            self.assertIn("Initial", msg0)

            apply_rebalance_to_profile(
                loaded,
                {"XLE": 10.5, "XLV": 5.0, "CASH/SGOV": 99.0},
                rebalance_date="2026-11-15",
                path=prof_p,
            )
            reloaded = load_profile(prof_p)
            self.assertEqual(reloaded.core_holdings, {"XLE": 10.5, "XLV": 5.0})
            self.assertEqual(reloaded.last_rebalance_date, "2026-11-15")

            due_ok, _ = rebalance_due_status(
                reloaded, max_drift_pct=2.0, rebalance_recommended=False, today=dt.date(2026, 11, 25)
            )
            self.assertFalse(due_ok)

            due_drift, msg_drift = rebalance_due_status(
                reloaded, max_drift_pct=7.2, rebalance_recommended=True, today=dt.date(2026, 11, 25)
            )
            self.assertTrue(due_drift)
            self.assertIn("7.2%", msg_drift)

    def test_wealthsimple_csv_import_idempotent_and_routing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prof_p = str(Path(td) / "tfsa_profile.json")
            picks_db = str(Path(td) / "own_picks.db")
            csv_p = Path(td) / "ws_activity.csv"
            csv_p.write_text(
                "date,transaction,symbol,quantity,price,amount\n"
                "2026-11-01,BUY,XLE,10,90.50,905.00\n"
                "2026-11-05,SELL,XLE,2,92.00,184.00\n"
                "2026-11-06,BUY,SGOV,15,100.40,1506.00\n"
                "2026-11-10,BUY,NVDA,3,140.00,420.00\n"
                "2026-11-12,DIVIDEND,XLE,0,0,5.25\n",
                encoding="utf-8",
            )
            res1 = import_wealthsimple_csv(str(csv_p), profile_path=prof_p, own_picks_db=picks_db)
            self.assertEqual(res1["core_updates"], 3)
            self.assertEqual(res1["picks_logged"], 1)
            self.assertEqual(res1["skipped_duplicates"], 0)
            self.assertAlmostEqual(res1["core_holdings"]["XLE"], 8.0, places=4)
            self.assertAlmostEqual(res1["core_holdings"]["SGOV"], 15.0, places=4)

            picks = list_equity_picks(db_path=picks_db)
            self.assertEqual(len(picks), 1)
            self.assertEqual(picks[0]["ticker"], "NVDA")
            self.assertEqual(picks[0]["playbook_id"], "wealthsimple_import")

            # Re-importing same statement must be 100% idempotent
            res2 = import_wealthsimple_csv(str(csv_p), profile_path=prof_p, own_picks_db=picks_db)
            self.assertEqual(res2["core_updates"], 0)
            self.assertEqual(res2["picks_logged"], 0)
            self.assertEqual(res2["skipped_duplicates"], 4)

    def test_ghost_race_exact_vm_parity_and_override_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "eq.db"
            picks_db = str(Path(td) / "own_picks.db")
            _seed_vm_ohlcv_db(db, n_days=340)

            # Log 1 closed winner + 1 open position inside the race window
            con = sqlite3.connect(str(db))
            dates = [r[0] for r in con.execute("SELECT date FROM ohlcv WHERE ticker='AAPL' ORDER BY date").fetchall()]
            con.close()
            d_ent = dates[-40]
            d_ex = dates[-15]
            pid1 = log_equity_pick("AAPL", 100.0, shares=10, entry_date=d_ent, playbook_id="equity_pullback", db_path=picks_db)
            close_equity_pick(pid1, 112.0, exit_date=d_ex, db_path=picks_db)
            log_equity_pick("AAPL", 105.0, shares=5, entry_date=dates[-10], playbook_id="equity_pullback", db_path=picks_db)

            race = compute_ghost_race(
                ohlcv_db=str(db),
                own_picks_db=picks_db,
                lookback_days=60,
                cash_yield_ann=0.05,
                account_usd=5000.0,
            )
            self.assertEqual(len(race.dates), 60)
            self.assertEqual(len(race.spy), 60)
            self.assertEqual(len(race.vm_outlook_sgov), 60)
            self.assertEqual(len(race.real_book), 60)
            self.assertEqual(race.spy[0], 100.0)
            self.assertEqual(race.vm_outlook_sgov[0], 100.0)
            self.assertGreaterEqual(race.vm_cash_only_boost_pct, 0.0)
            self.assertIsNotNone(race.real_ret_pct)
            self.assertIsNotNone(race.override_vs_core_pct)
            self.assertEqual(race.n_picks_total, 2)
            self.assertEqual(race.n_picks_closed, 1)
            self.assertEqual(race.win_rate_pct, 100.0)

    def test_cli_profile_and_rebalance_apply(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "eq.db"
            cache = Path(td) / "radar.json"
            prof_p = str(Path(td) / "tfsa_profile.json")
            picks_db = str(Path(td) / "own_picks.db")
            _seed_vm_ohlcv_db(db, n_days=300)
            obj = {
                "ohlcv_db": str(db),
                "cache_path": str(cache),
                "profile_path": prof_p,
                "own_picks_db": picks_db,
            }
            r1 = runner.invoke(cli, ["--no-color", "profile", "--tfsa", "10000", "--cash-etf", "SGOV", "--json"], obj=dict(obj))
            self.assertEqual(r1.exit_code, 0, r1.output)
            p1 = json.loads(r1.output)
            self.assertEqual(p1["tfsa_usd"], 10000.0)
            self.assertEqual(p1["cash_etf"], "SGOV")

            r2 = runner.invoke(cli, ["--no-color", "rebalance", "--refresh", "--apply", "--json"], obj=dict(obj))
            self.assertEqual(r2.exit_code, 0, r2.output)
            p2 = json.loads(r2.output)
            self.assertTrue(p2["applied"])
            self.assertEqual(p2["cash_etf"], "SGOV")
            self.assertGreater(len(p2["orders"]), 0)
            saved = load_profile(prof_p)
            self.assertIsNotNone(saved.last_rebalance_date)


class TestPhase2BloombergChartsAndWeb(unittest.TestCase):
    def test_braille_charts_and_theme_palettes(self) -> None:
        pal = theme_palette("amber_crt")
        self.assertEqual(pal["primary"], "#ffb03b")

        c_strip = vix_curve_strip(vix9d=14.0, vix=16.0, vix3m=19.0, vix6m=21.0, rich=False)
        self.assertIn("CONTANGO", c_strip)
        b_strip = vix_curve_strip(vix9d=32.0, vix=29.0, vix3m=25.0, vix6m=23.0, rich=False)
        self.assertIn("BACKWARDATION", b_strip)

        px_chart = braille_price_chart(
            [100.0 + i * 0.5 for i in range(40)],
            supports=[{"level": 105.0}],
            resistances=[{"level": 119.0}],
            sma200_level=102.0,
            rich=False,
        )
        self.assertEqual(len(px_chart.splitlines()), 5)
        self.assertIn("$", px_chart)

        board = build_sector_board(synthetic_ohlcv())
        rrg = braille_sector_rrg(board.rows, rich=False)
        self.assertIn("LEADING", rrg)
        self.assertIn("LAGGING", rrg)
        self.assertIn("★", rrg)

        view = DetailedView(symbol="SPY", direction="BULLISH", target_move_pct=0.06, horizon_days=30, risk_budget_usd=500.0)
        rnd = braille_payoff_and_rnd(595.0, 0.20, view, rich=False)
        self.assertIn("RND Curve", rnd)

        from src.equity_playbooks.ghost_race import GhostRaceResult

        fake_race = GhostRaceResult(
            dates=["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"],
            spy=[100.0, 101.0, 100.5, 102.0],
            vm_outlook_sgov=[100.0, 100.8, 101.2, 103.5],
            real_book=[100.0, 100.5, 101.0, 104.0],
            spy_ret_pct=2.0,
            vm_ret_pct=3.5,
            vm_cash_only_boost_pct=0.4,
            real_ret_pct=4.0,
            override_vs_core_pct=0.5,
        )
        g_txt = braille_ghost_race(fake_race, rich=False)
        self.assertIn("VM+SGOV +3.5%", g_txt)
        self.assertIn("Real Book +4.0%", g_txt)

    def test_web_terminal_html_and_api_helpers(self) -> None:
        self.assertIn('data-theme="amber_crt"', TERMINAL_HTML)
        self.assertIn("/desk/api/overview", TERMINAL_HTML)
        self.assertIn("/desk/api/card/", TERMINAL_HTML)
        self.assertIn("renderRRG", TERMINAL_HTML)
        self.assertIn("renderGhostSVG", TERMINAL_HTML)

        from src.web_terminal import build_desk_card, build_desk_overview
        from tests.test_phase2_3_ui_and_legacy import Phase2FakeProvider

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "eq.db"
            _seed_vm_ohlcv_db(db, n_days=280)
            prov = Phase2FakeProvider(
                ohlcv_db=str(db),
                own_picks_db=str(Path(td) / "picks.db"),
                fills_db=str(Path(td) / "fills.db"),
                chain_db=str(Path(td) / "chain.db"),
                paper_db=str(Path(td) / "paper.db"),
                longterm_db=str(Path(td) / "lt.db"),
                plan_path=str(Path(td) / "plan.json"),
            )
            ov = build_desk_overview(refresh=False, provider=prov)
            json.dumps(ov)
            for k in ("snapshot", "action_plan", "profile", "alpha", "ghost_race", "book", "rebalance", "health", "playbooks"):
                self.assertIn(k, ov)

            card = build_desk_card("SPY", provider=prov)
            json.dumps(card)
            self.assertEqual(card["ticker"], "SPY")
            self.assertIn("verdict", card)
            self.assertIn("premortem", card)
            self.assertIn("kill_switches", card["premortem"])


class TestPhase2TuiBloombergFeatures(unittest.IsolatedAsyncioTestCase):
    async def test_command_bar_rebalance_modal_rrg_and_nudges(self) -> None:
        from tests.test_phase2_3_ui_and_legacy import Phase2FakeProvider

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "eq.db"
            _seed_vm_ohlcv_db(db, n_days=280)
            prov = Phase2FakeProvider(
                ohlcv_db=str(db),
                own_picks_db=str(Path(td) / "picks.db"),
                fills_db=str(Path(td) / "fills.db"),
                chain_db=str(Path(td) / "chain.db"),
                paper_db=str(Path(td) / "paper.db"),
                longterm_db=str(Path(td) / "lt.db"),
                plan_path=str(Path(td) / "plan.json"),
            )
            app = DeskApp(provider=prov, initial_tab="today")
            async with app.run_test(size=(136, 42)) as pilot:
                await pilot.pause()

                # 1. Bloomberg `:` Command Bar -> `MACRO` opens AlphaViewerModal
                await pilot.press(":")
                await pilot.pause()
                self.assertIsInstance(app.screen, BloombergCommandModal)
                cmd_in = app.screen.query_one("#bcmd-input", Input)
                cmd_in.value = "MACRO"
                await pilot.press("enter")
                await pilot.pause(0.2)
                self.assertIsInstance(app.screen, AlphaViewerModal)
                await pilot.press("escape")
                await pilot.pause()

                # 2. `b` opens RebalanceModal showing SGOV reserve & orders
                await pilot.press("b")
                await pilot.pause(0.2)
                self.assertIsInstance(app.screen, RebalanceModal)
                await pilot.press("escape")
                await pilot.pause()

                # 3. Workbench `[` / `]` and `-` / `+` keyboard nudges
                app.open_workbench("SPY")
                await pilot.pause()
                t0 = app.query_one("#wb-target-input", Input).value
                await pilot.press("]")
                await pilot.pause()
                t1 = app.query_one("#wb-target-input", Input).value
                self.assertNotEqual(t0, t1)

                d0 = app.query_one("#wb-days-input", Input).value
                await pilot.press("+")
                await pilot.pause()
                d1 = app.query_one("#wb-days-input", Input).value
                self.assertNotEqual(d0, d1)

                # 4. Book tab shows Ghost Benchmark Race box
                await pilot.press("4")
                await pilot.pause(0.2)
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "book")
                race_txt = str(app.query_one("#book-race-box", Static).render())
                self.assertIn("GHOST BENCHMARK RACE", race_txt)


class TestPhase3AlphaLabsAndLenses(unittest.TestCase):
    def test_options_for_shares_cw_spread_put_smirk_and_dealer_gex(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdb = Path(td) / "chain_archive.db"
            _seed_chain_archive_db(cdb)
            sigs = scan_option_signals_for_shares(chain_db=str(cdb))
            self.assertEqual(len(sigs), 2)
            by_tk = {s.ticker: s for s in sigs}

            bull = by_tk["BULL"]
            self.assertEqual(bull.signal_label, "BULL_FLOW_SHARES")
            self.assertGreater(bull.cw_iv_spread_vp or 0.0, 3.0)
            self.assertEqual(bull.gex_regime, "POSITIVE_GEX")
            self.assertGreater(bull.net_gex_m or 0.0, 0.0)
            self.assertEqual(bull.call_wall, 105.0)
            self.assertEqual(bull.evidence_label, "WATCHLIST")

            bear = by_tk["BEAR"]
            self.assertEqual(bear.signal_label, "CRASH_SMIRK_AVOID")
            self.assertGreaterEqual(bear.put_smirk_vp or 0.0, 10.0)
            self.assertEqual(bear.gex_regime, "NEGATIVE_GEX")
            self.assertLess(bear.net_gex_m or 0.0, 0.0)
            self.assertEqual(bear.put_wall, 185.0)

            # Direct gamma-flip check when put GEX sits below spot and call GEX sits above spot
            gex = compute_dealer_gex(
                [
                    {"type": "put", "strike": 95.0, "gamma": 0.04, "open_interest": 10000},
                    {"type": "call", "strike": 105.0, "gamma": 0.04, "open_interest": 15000},
                ],
                spot=100.0,
            )
            self.assertIsNotNone(gex["gamma_flip"])
            self.assertGreater(gex["gamma_flip"], 95.0)
            self.assertLess(gex["gamma_flip"], 105.0)

    def test_edgar_events_efts_xbrl_and_sqlite_roundtrip(self) -> None:
        efts_payload = {
            "hits": {
                "hits": [
                    {
                        "_id": "adsh-001",
                        "_source": {
                            "adsh": "0001-26-0001",
                            "file_date": "2026-11-20",
                            "form": "10-12B",
                            "display_names": ["SPINCO CORP (SPNC) (CIK 0001234567)"],
                        },
                    }
                ]
            }
        }
        evs = parse_efts_hits(efts_payload, event_type="SPINOFF_FORM10", source_label="SEC EFTS")
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].ticker, "SPNC")

        xbrl_payload = {
            "entityName": "APPLE INC",
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"end": "2025-10-01", "val": 15_500_000_000},
                                {"end": "2026-10-01", "val": 14_880_000_000},
                            ]
                        }
                    }
                },
                "us-gaap": {
                    "GrossProfit": {"units": {"USD": [{"end": "2026-10-01", "form": "10-K", "val": 180_000_000_000}]}},
                    "Assets": {"units": {"USD": [{"end": "2026-10-01", "form": "10-K", "val": 360_000_000_000}]}},
                },
            },
        }
        bb = compute_xbrl_buyback_quality("AAPL", xbrl_payload)
        self.assertIsNotNone(bb)
        assert bb is not None
        self.assertAlmostEqual(bb.metric_value or 0.0, 4.0, places=1)
        self.assertIn("GP/A 50.0%", bb.summary)

        with tempfile.TemporaryDirectory() as td:
            edb = str(Path(td) / "edgar_events.db")
            self.assertEqual(save_edgar_events([evs[0], bb], db_path=edb), 2)
            loaded = load_cached_edgar_events(db_path=edb)
            self.assertEqual(len(loaded), 2)
            sp = load_cached_edgar_events(db_path=edb, event_type="SPINOFF_FORM10")
            self.assertEqual(len(sp), 1)
            self.assertEqual(sp[0].ticker, "SPNC")

            # Refresh with mocked getter (no network)
            def _fake_get(url: str) -> Dict[str, Any]:
                return efts_payload if "search-index" in url else xbrl_payload

            with mock.patch("src.insider.edgar.cik_for", return_value=320193):
                out = refresh_edgar_events(db_path=edb, buyback_tickers=("AAPL",), getter=_fake_get)
            self.assertGreaterEqual(out["saved"], 2)

    def test_macro_plumbing_fred_liquidity_cot_and_polymarket_honesty(self) -> None:
        walcl = parse_fred_csv(
            "DATE,WALCL\n"
            + "\n".join(f"2026-08-{i:02d},{7_000_000 + i * 50_000}" for i in range(1, 20))
        )
        tga = parse_fred_csv("DATE,WTREGEN\n2026-08-01,700000\n2026-08-19,650000\n")
        rrp = parse_fred_csv("DATE,RRPONTSYD\n2026-08-01,200.0\n2026-08-19,150.0\n")
        liq = compute_fed_net_liquidity(walcl, tga, rrp)
        self.assertIsNotNone(liq["fed_net_liquidity_b"])
        self.assertGreater(liq["fed_liq_chg_4w_b"], 0.0)

        # Polymarket strict spread guard: reject wide >0.05 or 0-bid markets!
        pm_rows = parse_polymarket_macro(
            [
                {"question": "Will the Fed cut rates in December?", "bestBid": 0.62, "bestAsk": 0.65, "volume24hr": 500000},
                {"question": "Will CPI beat 3.0%?", "bestBid": 0.30, "bestAsk": 0.45, "volume24hr": 900000},  # spread 0.15 -> REJECT
                {"question": "Will Fed hike?", "bestBid": 0.0, "bestAsk": 0.02, "volume24hr": 800000},  # 0 bid -> REJECT
                {"question": "Who wins the NBA finals?", "bestBid": 0.50, "bestAsk": 0.52, "volume24hr": 999999},  # non-macro -> REJECT
            ]
        )
        self.assertEqual(len(pm_rows), 1)
        self.assertAlmostEqual(pm_rows[0]["prob_pct"], 63.5, places=1)

        cot = parse_cftc_cot(
            [
                {
                    "market_and_exchange_names": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
                    "report_date_as_yyyy_mm_dd": "2026-11-18T00:00:00.000",
                    "noncomm_positions_long_all": "350000",
                    "noncomm_positions_short_all": "100000",
                    "open_interest_all": "1000000",
                }
            ]
        )
        self.assertEqual(len(cot), 1)
        self.assertEqual(cot[0]["asset"], "SPY/ES")
        self.assertEqual(cot[0]["bias"], "CROWDED LONG")

        with tempfile.TemporaryDirectory() as td:
            cpath = str(Path(td) / "macro_plumbing.json")

            def _fake_fetch(url: str) -> str:
                if "VIX" in url:
                    return "DATE,OPEN,HIGH,LOW,CLOSE\n2026-11-24,18,19,17,18.5\n"
                if "WALCL" in url:
                    return "DATE,WALCL\n2026-11-01,7000000\n2026-11-20,7100000\n"
                if "WTREGEN" in url:
                    return "DATE,WTREGEN\n2026-11-01,700000\n"
                if "RRPONTSYD" in url:
                    return "DATE,RRPONTSYD\n2026-11-01,100\n"
                if "polymarket" in url:
                    return json.dumps([{"question": "Will the Fed cut rates?", "bestBid": 0.70, "bestAsk": 0.72, "volume24hr": 100000}])
                return json.dumps([])

            snap = refresh_macro_plumbing(cache_path=cpath, fetcher=_fake_fetch)
            self.assertEqual(snap.vix_curve["vix"], 18.5)
            reloaded = load_macro_plumbing(cache_path=cpath)
            self.assertEqual(reloaded.fed_net_liquidity_b, snap.fed_net_liquidity_b)
            self.assertEqual(len(reloaded.polymarket_macro), 1)

    def test_premortem_and_topdown_phase3_lenses(self) -> None:
        clean_dos = _make_dossier(
            "XOM", spot=120.0, dist_sma200_pct=8.0,
            sector_etf="XLE", sector_posture="FAVOR",
            supports=[{"label": "50d MA", "level": 114.0, "pct": -0.05}],
        )
        pm_clean = build_premortem("XOM", clean_dos, account_type="tfsa")
        self.assertEqual(pm_clean.risk_grade, "CLEAN")
        self.assertTrue(any("Wealthsimple TFSA" in n for n in pm_clean.tfsa_notes))

        bad_dos = _make_dossier(
            "BAD", spot=50.0, dist_sma200_pct=-12.0,
            days_to_earnings=3, earnings_date="2026-11-28",
            risk_flags=["POST_EARN_CRASH(-9.0%)"], sector_etf="XLY", sector_posture="AVOID",
        )
        mac = MacroPlumbingSnapshot(vix_curve={"vix": 30.0, "vix3m": 25.0}, vix_term_label="BACKWARDATION")
        pm_bad = build_premortem("BAD", bad_dos, macro=mac, account_type="tfsa")
        self.assertEqual(pm_bad.risk_grade, "HIGH_RISK")
        self.assertTrue(any("earnings" in p.lower() for p in pm_bad.bear_thesis_points))
        self.assertTrue(any("VIX curve inverted" in p for p in pm_bad.bear_thesis_points))

        # Verify Phase 3 lenses in src/topdown/lenses.py
        self.assertEqual(LENS_BY_ID["opt_skew"].label, "WATCHLIST")
        self.assertEqual(LENS_BY_ID["edgar_events"].label, "WATCHLIST")
        with tempfile.TemporaryDirectory() as td:
            cdb = Path(td) / "chain.db"
            edb = str(Path(td) / "edgar.db")
            _seed_chain_archive_db(cdb)
            evs = parse_efts_hits(
                {"hits": {"hits": [{"_id": "1", "_source": {"file_date": "2026-11-21", "form": "SC 13D", "display_names": ["ACME (ACME)"]}}]}},
                event_type="ACTIVIST_13D",
                source_label="SEC EFTS",
            )
            save_edgar_events(evs, db_path=edb)
            o = synthetic_ohlcv()
            ctx = LensContext(ohlcv=o, board=build_sector_board(o), chain_db=str(cdb), edgar_db=edb)

            res_sk = run_lens("opt_skew", ctx)
            self.assertEqual({r.ticker for r in res_sk.rows}, {"BULL", "BEAR"})
            res_ed = run_lens("edgar_events", ctx)
            self.assertEqual([r.ticker for r in res_ed.rows], ["ACME"])

    def test_cli_alpha_subcommands_json_and_text(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            cdb = Path(td) / "chain.db"
            edb = str(Path(td) / "edgar.db")
            mpath = str(Path(td) / "macro.json")
            _seed_chain_archive_db(cdb)
            Path(mpath).write_text(json.dumps({"vix_curve": {"vix": 16.5, "vix3m": 19.0}, "vix_term_label": "CONTANGO"}))

            with mock.patch("src.cli.alpha.scan_option_signals_for_shares", side_effect=lambda **kw: scan_option_signals_for_shares(chain_db=str(cdb), **kw)), \
                 mock.patch("src.cli.alpha.load_cached_edgar_events", side_effect=lambda **kw: load_cached_edgar_events(db_path=edb, **kw)), \
                 mock.patch("src.cli.alpha.load_macro_plumbing", side_effect=lambda: load_macro_plumbing(cache_path=mpath)):
                r1 = runner.invoke(cli, ["--no-color", "alpha", "skew", "--json"])
                self.assertEqual(r1.exit_code, 0, r1.output)
                self.assertEqual(len(json.loads(r1.output)), 2)

                r2 = runner.invoke(cli, ["--no-color", "alpha", "gex", "BULL", "--json"])
                self.assertEqual(r2.exit_code, 0, r2.output)
                self.assertEqual(json.loads(r2.output)["gex_regime"], "POSITIVE_GEX")

                r3 = runner.invoke(cli, ["--no-color", "alpha", "edgar", "--json"])
                self.assertEqual(r3.exit_code, 0, r3.output)
                self.assertEqual(json.loads(r3.output), [])

                r4 = runner.invoke(cli, ["--no-color", "alpha", "macro", "--json"])
                self.assertEqual(r4.exit_code, 0, r4.output)
                self.assertEqual(json.loads(r4.output)["vix_term_label"], "CONTANGO")

                fake_dos = _make_dossier("BULL", spot=100.0, dist_sma200_pct=5.0)
                with mock.patch("src.cli.alpha.build_dossier", return_value=fake_dos):
                    r5 = runner.invoke(cli, ["--no-color", "alpha", "premortem", "BULL", "--json"])
                self.assertEqual(r5.exit_code, 0, r5.output)
                self.assertEqual(json.loads(r5.output)["ticker"], "BULL")


if __name__ == "__main__":
    unittest.main()
