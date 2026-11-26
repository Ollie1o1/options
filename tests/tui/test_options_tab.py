"""Comprehensive offline unit & Textual pilot tests for `[7] OPTIONS`, Chain Matrix, Leg Composer, Risk Surface & Paper Log."""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from textual.widgets import DataTable, Input, Static, TabbedContent

from src.chain_archive import ensure_db
from src.tui.app import DeskApp
from src.tui.charts import (
    braille_iv_smile_and_term,
    braille_multileg_payoff,
    flow_split_bar,
    gex_profile_bars,
    risk_heatmap_grid,
)
from src.tui.options_engine import (
    OptionLeg,
    OptionsIntelBundle,
    analyze_custom_legs,
    classify_legs,
    filter_option_rows,
    load_chain_matrix,
    scan_options_desk,
)
from src.tui.widgets import PaperOptionModal
from tests.tui.test_desk_app import FakeProvider


def _seed_chain_archive(db_path: str) -> str:
    """Create a realistic synthetic multi-expiry `chain_archive.db` for SPY & AAPL."""
    ensure_db(db_path)
    today = dt.date.today()
    snap = today.isoformat()
    exp_35 = (today + dt.timedelta(days=35)).isoformat()
    exp_85 = (today + dt.timedelta(days=85)).isoformat()

    rows = []
    from src.utils import bs_delta, bs_gamma, bs_price, bs_theta

    for sym, spot, base_iv in (("SPY", 590.0, 0.18), ("AAPL", 225.0, 0.28)):
        step = 5.0 if sym == "SPY" else 2.5
        strikes = [round(spot + i * step, 2) for i in range(-12, 13)]
        for exp_s, dte in ((exp_35, 35), (exp_85, 85)):
            T = dte / 365.0
            for k in strikes:
                moneyness = (k - spot) / spot
                # Realistic smirk: higher IV on downside puts
                iv = max(0.10, base_iv + (-0.28 * moneyness if moneyness < 0 else 0.08 * moneyness) + (0.015 if dte == 35 else 0.0))
                for ot in ("call", "put"):
                    theo = max(0.15, float(bs_price(ot, spot, k, T, 0.045, iv)))
                    half = max(0.02, round(theo * 0.025, 2))
                    bid = max(0.05, round(theo - half, 2))
                    ask = round(theo + half, 2)
                    dl = float(bs_delta(ot, spot, k, T, 0.045, iv))
                    gm = float(bs_gamma(spot, k, T, 0.045, iv))
                    th = float(bs_theta(ot, spot, k, T, 0.045, iv))
                    oi = 4500 if abs(moneyness) < 0.04 else 1200
                    vol = 900 if abs(moneyness) < 0.04 else 250
                    contract = f"{sym}{exp_s.replace('-', '')}{ot[0].upper()}{int(k * 1000):08d}"
                    rows.append(
                        (
                            sym, snap, contract, ot, k, exp_s,
                            bid, ask, 20.0, 20.0, iv, dl, gm, th, 0.15, 0.01,
                            float(oi), float(vol), snap, spot, snap, "synthetic",
                        )
                    )

    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "INSERT OR REPLACE INTO chain_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        con.commit()
    finally:
        con.close()
    return exp_35


class OptionsDeskProvider(FakeProvider):
    """Offline provider backed by synthetic chain_archive.db + temp paper_trades.db."""

    def __init__(self, td: str) -> None:
        super().__init__(own_picks_db=os.path.join(td, "own_picks.db"), worst_health="OK")
        self.chain_db = os.path.join(td, "chain_archive.db")
        self.paper_db = os.path.join(td, "paper_trades.db")
        self.exp_35 = _seed_chain_archive(self.chain_db)

    def get_options_intel(self) -> OptionsIntelBundle:
        return OptionsIntelBundle(
            uoa_rows=[
                {
                    "symbol": "SPY",
                    "unusual_count": 3,
                    "net_call_share": 0.74,
                    "call_oi_added": 12400,
                    "put_oi_added": 4100,
                    "call_premium_added": 3_200_000.0,
                    "put_premium_added": 1_100_000.0,
                    "unusual": [
                        {
                            "type": "call",
                            "strike": 600.0,
                            "expiration": self.exp_35,
                            "d_oi": 5400,
                            "volume": 8200,
                            "why": "oi_jump",
                        }
                    ],
                }
            ],
            occ_rows=[
                {
                    "symbol": "AAPL",
                    "trade_date": dt.date.today().isoformat(),
                    "total_vol": 285000,
                    "total_oi": 410000,
                    "vol_to_oi": 0.70,
                    "oi_delta_pct": 14.5,
                    "signal_label": "OI_SURGE",
                }
            ],
            vol_rows=[
                {
                    "symbol": "SPY",
                    "iv_pct": 18.5,
                    "rv_pct": 13.0,
                    "vrp_vp": 5.5,
                    "iv_rv_ratio": 1.42,
                    "iv_chg_vp": 1.1,
                    "rv_pctile": 38.0,
                }
            ],
            skew_rows=[
                {
                    "ticker": "AAPL",
                    "spot": 225.0,
                    "snap_date": dt.date.today().isoformat(),
                    "cw_iv_spread_vp": 1.8,
                    "put_smirk_vp": 4.2,
                    "net_gex_m": 42.5,
                    "gex_regime": "POSITIVE_GEX",
                    "gamma_flip": 220.0,
                    "call_wall": 235.0,
                    "put_wall": 215.0,
                    "signal_label": "BULL_FLOW_SHARES",
                }
            ],
            as_of=dt.date.today().isoformat(),
        )


class TestOptionsEngineAndCharts(unittest.TestCase):
    def test_scanner_chain_matrix_and_multileg_math(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ch_db = os.path.join(td, "chain_archive.db")
            exp_35 = _seed_chain_archive(ch_db)

            rows, note = scan_options_desk(strategy="all", chain_db=ch_db, ohlcv_db=os.path.join(td, "none.db"))
            self.assertGreater(len(rows), 4)
            self.assertIn("Archive scan", note)

            bp = filter_option_rows(rows, "bull_put")
            tfsa = filter_option_rows(rows, "tfsa")
            self.assertGreater(len(bp), 0)
            self.assertTrue(all(r.strategy_key == "bull_put" and not r.tfsa_legal for r in bp))
            self.assertGreater(len(tfsa), 0)
            self.assertTrue(all(r.tfsa_legal for r in tfsa))

            mat = load_chain_matrix("SPY", expiration=exp_35, chain_db=ch_db, ohlcv_db=os.path.join(td, "none.db"))
            self.assertIsNotNone(mat)
            assert mat is not None
            self.assertEqual(mat.ticker, "SPY")
            self.assertGreater(len(mat.rows), 8)
            self.assertTrue(any("SPOT" in r.tags for r in mat.rows))
            self.assertGreaterEqual(len(mat.term_points), 2)
            self.assertIsNotNone(mat.net_gex_m)

            # Multi-leg Bull Put spread analysis
            legs = [
                OptionLeg("put", "SELL", 575.0, exp_35, 35, 4.10, 4.30, 4.20, 0.20),
                OptionLeg("put", "BUY", 550.0, exp_35, 35, 1.30, 1.50, 1.40, 0.23),
            ]
            sname, legal = classify_legs(legs)
            self.assertEqual(sname, "Bull Put")
            self.assertFalse(legal)
            an = analyze_custom_legs("SPY", 590.0, legs, days_elapsed=10, iv_shock_pct=-10.0)
            self.assertTrue(an.is_credit)
            self.assertAlmostEqual(an.net_mid, 2.80, places=2)
            self.assertEqual(len(an.grid_pnl), 5)
            self.assertEqual(len(an.grid_pnl[0]), 9)
            self.assertGreater(len(an.breakevens), 0)

            # Visual renderers produce clean non-empty output
            self.assertIn("IV SMILE", braille_iv_smile_and_term(mat.smile_points, mat.term_points, spot=mat.spot, rich=False))
            self.assertIn("DEALER GEX", gex_profile_bars(mat.gex_by_strike, spot=mat.spot, rich=False))
            self.assertIn("2D RISK", risk_heatmap_grid(an.grid_price_shocks, an.grid_iv_shocks, an.grid_pnl, rich=False))
            self.assertIn("PAYOFF", braille_multileg_payoff(an.prices, an.expiry_pnl, an.shocked_pnl, spot=590.0, rich=False))
            self.assertIn("%C", flow_split_bar(0.75, rich=False))

    def test_web_terminal_options_endpoints_and_html(self) -> None:
        from src.web_terminal import (
            TERMINAL_HTML,
            build_desk_options_chain,
            build_desk_options_scan,
            run_desk_options_analyze,
        )

        self.assertIn("OPTIONS HUB", TERMINAL_HTML)
        self.assertIn("/desk/api/options/scan", TERMINAL_HTML)
        self.assertIn("/desk/api/options/chain", TERMINAL_HTML)
        self.assertIn("/desk/api/options/analyze", TERMINAL_HTML)

        with tempfile.TemporaryDirectory() as td:
            prov = OptionsDeskProvider(td)
            scan = build_desk_options_scan(strategy="bull_put", provider=prov)
            self.assertGreater(scan["count"], 0)
            ch = build_desk_options_chain(ticker="SPY", expiration=prov.exp_35, provider=prov)
            self.assertTrue(ch["found"])
            self.assertEqual(ch["chain"]["ticker"], "SPY")
            first_row = scan["rows"][0]
            an = run_desk_options_analyze(
                {
                    "ticker": first_row["ticker"],
                    "spot": first_row["spot"],
                    "legs": first_row["legs"],
                    "days_elapsed": 5,
                    "iv_shock_pct": -10,
                }
            )
            self.assertTrue(an["ok"])
            self.assertEqual(an["analysis"]["days_elapsed"], 5)
            self.assertEqual(len(an["analysis"]["grid_pnl"]), 5)

    def test_discord_options_and_ticket_embeds(self) -> None:
        from src.notifications import build_option_ticket_embed, build_options_embed

        with tempfile.TemporaryDirectory() as td:
            prov = OptionsDeskProvider(td)
            rows, note = prov.scan_options(strategy="all", live=False)
            emb = build_options_embed(rows, note=note, strategy="all")
            self.assertIn("Options Desk", emb["title"])
            self.assertGreaterEqual(len(emb["fields"]), 1)

            r = rows[0]
            t_emb = build_option_ticket_embed(
                r.ticker,
                r.strategy,
                spot=r.spot,
                legs_summary=r.summary,
                net_label=f"${r.mid_price:.2f}",
                friction_pct=r.friction_pct,
                max_risk_usd=r.max_risk_usd,
                tfsa_legal=r.tfsa_legal,
                ladder_lines=("Step 1", "Step 2", "Abort"),
            )
            self.assertIn(r.ticker, t_emb["title"])
            self.assertGreaterEqual(len(t_emb["fields"]), 3)

    def test_only_if_good_filter_and_bot_api_embeds(self) -> None:
        from click.testing import CliRunner
        from unittest.mock import patch

        from src.cli.main import cli
        from src.notifications import (
            build_chain_api_embed,
            build_morning_api_embed,
            build_options_scan_api_embed,
            filter_good_option_alerts,
        )
        from src.tui.options_engine import OptionScanRow
        from src.web_terminal import build_desk_options_chain, build_desk_options_scan

        good_bp = OptionScanRow(
            ticker="SPY",
            strategy="Bull Put",
            strategy_key="bull_put",
            expiration="2026-12-18",
            dte=35,
            strikes_label="575/550P",
            spot=590.0,
            short_delta=0.25,
            mid_price=2.80,
            is_credit=True,
            walk_away_limit=2.65,
            friction_pct=0.08,
            max_risk_usd=2220.0,
            max_profit_usd=280.0,
            ror_pct=12.6,
            iv_pct=19.0,
            iv_vs_rv=1.15,
            skew_vp=3.5,
            min_oi=400,
            total_vol=200,
            tfsa_legal=False,
            evidence_badge="PROMISING",
            risk_flags=("TAXABLE_ONLY",),
        )
        noisy_bp = OptionScanRow(
            ticker="NVDA",
            strategy="Bull Put",
            strategy_key="bull_put",
            expiration="2026-12-18",
            dte=35,
            strikes_label="130/120P",
            spot=140.0,
            short_delta=0.25,
            mid_price=1.20,
            is_credit=True,
            walk_away_limit=1.05,
            friction_pct=0.09,
            max_risk_usd=880.0,
            max_profit_usd=120.0,
            ror_pct=13.6,
            iv_pct=48.0,
            iv_vs_rv=1.20,
            skew_vp=4.0,
            min_oi=500,
            total_vol=300,
            tfsa_legal=False,
            evidence_badge="PROMISING",
            risk_flags=("TAXABLE_ONLY", "EARNINGS_IN_5D"),
        )
        good_tfsa = OptionScanRow(
            ticker="QQQ",
            strategy="Long Call",
            strategy_key="long_call",
            expiration="2026-12-18",
            dte=40,
            strikes_label="$510C",
            spot=505.0,
            short_delta=0.45,
            mid_price=11.50,
            is_credit=False,
            walk_away_limit=11.60,
            friction_pct=0.04,
            max_risk_usd=1150.0,
            max_profit_usd=10000.0,
            ror_pct=None,
            iv_pct=18.5,
            iv_vs_rv=0.88,
            skew_vp=2.1,
            min_oi=350,
            total_vol=150,
            tfsa_legal=True,
            evidence_badge="WATCHLIST",
            risk_flags=(),
        )
        rich_tfsa = OptionScanRow(
            ticker="TSLA",
            strategy="Long Call",
            strategy_key="long_call",
            expiration="2026-12-18",
            dte=40,
            strikes_label="$260C",
            spot=255.0,
            short_delta=0.45,
            mid_price=15.00,
            is_credit=False,
            walk_away_limit=15.20,
            friction_pct=0.05,
            max_risk_usd=1500.0,
            max_profit_usd=5000.0,
            ror_pct=None,
            iv_pct=65.0,
            iv_vs_rv=1.35,
            skew_vp=2.0,
            min_oi=350,
            total_vol=150,
            tfsa_legal=True,
            evidence_badge="WATCHLIST",
            risk_flags=("IV_RICH_FOR_LONG",),
        )

        filtered = filter_good_option_alerts([good_bp, noisy_bp, good_tfsa, rich_tfsa])
        self.assertEqual([r.ticker for r in filtered], ["SPY", "QQQ"])

        # CLI --only-if-good skips Discord quietly when nothing qualifies
        runner = CliRunner()
        with patch("src.tui.options_engine.scan_options_desk", return_value=([noisy_bp, rich_tfsa], "mock scan")):
            res_quiet = runner.invoke(cli, ["notify", "--options", "--only-if-good", "--dry-run"])
            self.assertEqual(res_quiet.exit_code, 0, res_quiet.output)
            self.assertIn('"skipped_quiet": true', res_quiet.output)

        with patch("src.tui.options_engine.scan_options_desk", return_value=([good_bp, noisy_bp], "mock scan")):
            res_hit = runner.invoke(cli, ["notify", "--options", "--only-if-good", "--dry-run"])
            self.assertEqual(res_hit.exit_code, 0, res_hit.output)
            self.assertIn("High-Conviction Alert", res_hit.output)
            self.assertIn("SPY", res_hit.output)
            self.assertNotIn("NVDA", res_hit.output)

        # Discord slash command embed builders (/options, /chain, /morning)
        with tempfile.TemporaryDirectory() as td:
            prov = OptionsDeskProvider(td)
            scan_json = build_desk_options_scan(strategy="all", provider=prov)
            opt_emb = build_options_scan_api_embed(scan_json, strategy="all")
            self.assertIn("Options Hub Scan", opt_emb["title"])
            self.assertGreaterEqual(len(opt_emb["fields"]), 1)

            ch_json = build_desk_options_chain(ticker="SPY", expiration=prov.exp_35, provider=prov)
            ch_emb = build_chain_api_embed(ch_json)
            self.assertIn("SPY", ch_emb["title"])
            self.assertTrue(any("Near-ATM" in f["name"] for f in ch_emb["fields"]))

            morn_emb = build_morning_api_embed(
                {
                    "snapshot": {
                        "regime_posture": "RISK_ON",
                        "spy_last": 590.0,
                        "spy_chg_1d_pct": 0.6,
                        "qqq_chg_1d_pct": 0.9,
                        "vix": 15.2,
                        "vix_term": "CONTANGO",
                        "spy_rv20": 13.4,
                        "next_macro_event": "FOMC",
                        "core_allocation": [{"ticker": "QQQ", "target_weight": 0.5}],
                        "opportunities": [],
                    },
                    "book": {"equity_picks": [], "exit_alerts": []},
                }
            )
            self.assertIn("RISK_ON", morn_emb["title"])
            self.assertIn("QQQ", morn_emb["fields"][1]["value"])


class TestOptionsTabPilot(unittest.IsolatedAsyncioTestCase):
    async def test_options_scanner_chain_composer_intel_heatmap_and_paper_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prov = OptionsDeskProvider(td)
            app = DeskApp(provider=prov, offline=True)
            async with app.run_test(size=(140, 42)) as pilot:
                await pilot.pause()

                # 1. Switch to [7] OPTIONS — scanner auto-loads from archive
                await pilot.press("7")
                for _ in range(30):
                    await pilot.pause(0.05)
                    if app._opt_scan_loaded:
                        break
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "options")
                scan_tbl = app.query_one("#opt-scan-table", DataTable)
                self.assertGreater(scan_tbl.row_count, 3)

                # 2. Micro-query filter works live
                flt = app.query_one("#opt-filter-input", Input)
                flt.value = "tfsa"
                await pilot.pause(0.1)
                self.assertGreater(scan_tbl.row_count, 0)
                self.assertTrue(all(r.tfsa_legal for r in app._opt_scan_filtered))
                flt.value = "bull_put"
                await pilot.pause(0.1)
                self.assertGreater(scan_tbl.row_count, 0)
                self.assertTrue(all(r.strategy_key == "bull_put" for r in app._opt_scan_filtered))

                # 3. Paper-log modal (`p`) writes to paper_trades.db and populates BOOK options table
                await pilot.press("p")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, PaperOptionModal)
                await pilot.click("#po-confirm")
                await pilot.pause(0.15)
                bot = app.query_one("#book-options-table", DataTable)
                self.assertEqual(bot.row_count, 1)

                # 4. Enter on scanner row jumps to T-shaped Chain Matrix
                scan_tbl.focus()
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause(0.05)
                    if app._opt_view == "chain" and app._opt_chain_matrix is not None:
                        break
                self.assertEqual(app._opt_view, "chain")
                chain_tbl = app.query_one("#opt-chain-table", DataTable)
                self.assertGreater(chain_tbl.row_count, 5)

                # 5. 1-click BullPut 5% template & time/IV scrubbers on Chain Matrix
                await pilot.click("#opt-leg-bp5")
                await pilot.pause(0.1)
                self.assertEqual(len(app._opt_staged_legs), 2)
                await pilot.press(".")
                await pilot.pause(0.05)
                self.assertEqual(app._days_elapsed, 5)
                insp = str(app.query_one("#opt-chain-inspector", Static).render())
                self.assertIn("LEG COMPOSER", insp)
                self.assertIn("2D P&L SURFACE", insp)
                self.assertIn("T+5d", insp)

                # 6. Flow & Vol Intel sub-view
                await pilot.click("#opt-nav-intel")
                for _ in range(30):
                    await pilot.pause(0.05)
                    if app._opt_intel is not None:
                        break
                self.assertEqual(app._opt_view, "intel")
                intel_tbl = app.query_one("#opt-intel-table", DataTable)
                self.assertGreaterEqual(intel_tbl.row_count, 4)

                # 7. Workbench 2D Risk Surface toggle (`h`) + IV shock nudge (`{`)
                app.open_workbench("AAPL")
                await pilot.pause(0.15)
                await pilot.press("h")
                await pilot.pause(0.1)
                self.assertTrue(app._wb_show_heatmap)
                await pilot.press("{")
                await pilot.pause(0.1)
                self.assertEqual(app._iv_shock_pct, -5.0)

                # 8. Bloomberg `:` mnemonic `CHAIN SPY` & `FLOW`
                app._execute_bloomberg_mnemonic("CHAIN SPY")
                for _ in range(30):
                    await pilot.pause(0.05)
                    if app._opt_chain_matrix is not None and app._opt_chain_matrix.ticker == "SPY":
                        break
                self.assertEqual(app._opt_view, "chain")
                self.assertEqual(app._opt_chain_matrix.ticker, "SPY")

    async def test_compact_options_tab_renders_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prov = OptionsDeskProvider(td)
            app = DeskApp(provider=prov, initial_tab="options", offline=True)
            async with app.run_test(size=(80, 24)) as pilot:
                for _ in range(30):
                    await pilot.pause(0.05)
                    if app._opt_scan_loaded:
                        break
                self.assertTrue(app.screen.has_class("compact"))
                self.assertGreater(app.query_one("#opt-scan-table", DataTable).row_count, 0)


if __name__ == "__main__":
    unittest.main()
