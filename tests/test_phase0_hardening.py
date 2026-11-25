"""Phase 0 Hardening Regression Tests: CRA TFSA guard, Radar parity, pick closing, env loader, and notifications."""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from click.testing import CliRunner

from src.cli.main import cli
from src.notifications import (
    AlertEvent,
    build_events_embed,
    build_morning_embed,
    detect_action_events,
    filter_unsent_events,
    format_status_pill,
    send_macos_notification,
    summarize_for_macos,
)
from src.own_picks import list_equity_picks, log_equity_pick
from src.paths import load_project_env
from src.radar.engine import enrich_with_portfolio_and_iv, merge_confluence
from src.radar.playbooks import load_playbooks
from src.radar.scanners import scan_archived_etf_wide_spreads
from src.radar.schema import Opportunity, RadarSnapshot
from src.structure.express import _bs_px_delta, compare_vehicles
from src.structure.types import DetailedView
from src.tui.app import DeskApp
from tests.tui.test_desk_app import FakeProvider


def _make_opp(
    ticker: str,
    playbook_id: str,
    direction: str,
    *,
    tier: str = "LEAD",
    asset_class: str = "option",
    friction: float = 0.05,
) -> Opportunity:
    return Opportunity(
        ticker=ticker,
        asset_class=asset_class,
        playbook_id=playbook_id,
        playbook_label=playbook_id,
        evidence_tier=tier,
        direction=direction,
        spot=500.0,
        chg_1d_pct=0.0,
        chg_5d_pct=0.0,
        horizon_label="90d",
        setup_summary="test",
        badges=(),
        risk_flags=(),
        friction_label="5.0%",
        friction_value=friction,
    )


class TestPhase0CraTfsaAndRadarParity(unittest.TestCase):
    def test_playbooks_json_fences_credit_spreads_out_of_tfsa(self) -> None:
        reg = load_playbooks("configs/playbooks.json")
        self.assertEqual(reg["etf_wide_spread"].accounts, ("taxable",))
        self.assertEqual(reg["etf_bear_call"].accounts, ("taxable",))
        self.assertIn("tfsa", reg["vol_managed_outlook"].accounts)
        self.assertIn("tfsa", reg["equity_pullback"].accounts)
        self.assertIn("tfsa", reg["covered_call_holdings"].accounts)

    def test_merge_confluence_rejects_opposing_directions(self) -> None:
        bull_put = _make_opp("SPY", "etf_wide_spread", "LONG")
        bear_call = _make_opp("SPY", "etf_bear_call", "SHORT")
        merged = merge_confluence([bull_put, bear_call])
        for row in merged:
            self.assertEqual(row.confluence_count, 1)
            self.assertIn("OPPOSING_SIGNAL", row.risk_flags)

    def test_short_vol_gets_risk_flags_in_backwardation_and_cheap_iv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdb = Path(td) / "chains.db"
            con = sqlite3.connect(str(cdb))
            con.execute(
                "CREATE TABLE chain_snapshots (symbol TEXT, snap_date TEXT, contract TEXT, type TEXT, "
                "strike REAL, expiration TEXT, bid REAL, ask REAL, bid_size REAL, ask_size REAL, "
                "iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL, open_interest REAL, volume REAL)"
            )
            today = dt.date.today()
            exp = (today + dt.timedelta(days=30)).isoformat()
            # Low ATM IV (12%) while realized vol is high (~35%) -> IV/RV << 0.90
            con.execute(
                "INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("SPY", today.isoformat(), "SPY_C_500", "call", 500.0, exp, 5.0, 5.2, 10, 10, 0.12, 0.5, 0.01, -0.05, 0.2, 0.01, 500, 100),
            )
            con.commit()
            con.close()

            # Build high-RV20 price series (~35% annualized)
            rng = np.random.default_rng(7)
            closes = 500.0 * np.cumprod(1.0 + rng.normal(0.0, 0.024, 40))
            closes[-1] = 500.0
            ohlcv = {"SPY": ([f"2026-10-{i:02d}" for i in range(1, 41)], closes, np.ones(40) * 1e7)}

            short_opp = _make_opp("SPY", "etf_wide_spread", "LONG")
            enriched = enrich_with_portfolio_and_iv(
                [short_opp],
                ohlcv,
                chain_db=str(cdb),
                own_picks_db=str(Path(td) / "none.db"),
                paper_db=str(Path(td) / "none2.db"),
                longterm_db=str(Path(td) / "none3.db"),
                vix_term="BACKWARDATION",
            )
            self.assertIn("VOL_CHEAP_FOR_SHORT", enriched[0].risk_flags)
            self.assertIn("VIX_BACKWARDATION", enriched[0].risk_flags)
            self.assertNotIn("IV_CHEAP", enriched[0].badges)

    def test_archived_o1_uses_delta_rule_and_flags_non_spy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdb = Path(td) / "chains.db"
            con = sqlite3.connect(str(cdb))
            con.execute(
                "CREATE TABLE chain_snapshots (symbol TEXT, snap_date TEXT, contract TEXT, type TEXT, "
                "strike REAL, expiration TEXT, bid REAL, ask REAL, bid_size REAL, ask_size REAL, "
                "iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL, open_interest REAL, volume REAL)"
            )
            today = dt.date.today()
            exp = (today + dt.timedelta(days=90)).isoformat()
            spot = 500.0
            t_yr = 90.0 / 365.0
            rows = []
            # Low-vol 12% chain: 6% OTM (470) is ~-0.12 delta, whereas -0.25 delta sits near 485!
            for k in range(430, 510, 5):
                px, dl, th = _bs_px_delta(spot, float(k), t_yr, 0.04, 0.14, "put")
                half = max(0.02, px * 0.01)
                rows.append(
                    ("QQQ", today.isoformat(), f"QQQ_P_{k}", "put", float(k), exp,
                     max(0.05, round(px - half, 2)), round(px + half, 2),
                     50, 50, 0.14, round(dl, 4), 0.01, round(th, 4), 0.2, 0.01, 2000, 300)
                )
            con.executemany("INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            con.commit()
            con.close()

            ohlcv = {"QQQ": ([today.isoformat()], np.array([spot]), np.array([1e7]))}
            reg = load_playbooks("configs/playbooks.json")
            opps = scan_archived_etf_wide_spreads(reg, ohlcv, symbols=("QQQ",), chain_db=str(cdb))
            self.assertEqual(len(opps), 1)
            opp = opps[0]
            # Short leg must follow -0.25 delta (480 or 485), NOT fixed 6% OTM (470)!
            self.assertGreaterEqual(opp.extra["short_strike"], 480.0)
            self.assertIn("TESTED_ON_SPY_ONLY", opp.risk_flags)
            self.assertIn("TAXABLE_ONLY", opp.risk_flags)
            self.assertEqual(len(opp.extra["walk_away_steps"]), 3)

    def test_compare_vehicles_never_recommends_margin_spread_in_tfsa(self) -> None:
        view = DetailedView(
            symbol="SPY",
            direction="NEUTRAL",
            target_move_pct=0.0,
            horizon_days=30,
            risk_budget_usd=5000.0,
            account_type="tfsa",
            owns_100_shares=False,
        )
        rows = compare_vehicles(view, spot=500.0, iv=0.20)
        self.assertTrue(all(not r.recommended for r in rows if not r.tfsa_legal))


class TestPhase0EnvAndCliLifecycle(unittest.TestCase):
    def test_load_project_env_parses_quotes_and_respects_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ef = Path(td) / ".env"
            ef.write_text(
                '# comment\n'
                'export TEST_P0_A="hello_world"\n'
                "TEST_P0_B='single'\n"
                'TEST_P0_C=plain\n',
                encoding="utf-8",
            )
            os.environ["TEST_P0_C"] = "existing"
            try:
                os.environ.pop("TEST_P0_A", None)
                os.environ.pop("TEST_P0_B", None)
                n = load_project_env(ef, override=False)
                self.assertEqual(n, 2)
                self.assertEqual(os.environ["TEST_P0_A"], "hello_world")
                self.assertEqual(os.environ["TEST_P0_B"], "single")
                self.assertEqual(os.environ["TEST_P0_C"], "existing")
                load_project_env(ef, override=True)
                self.assertEqual(os.environ["TEST_P0_C"], "plain")
            finally:
                for k in ("TEST_P0_A", "TEST_P0_B", "TEST_P0_C"):
                    os.environ.pop(k, None)

    def test_cli_opt_close_by_ticker_and_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "own_picks.db")
            pid = log_equity_pick("AAPL", 200.0, shares=10.0, playbook_id="equity_pullback", db_path=db)
            runner = CliRunner()
            res = runner.invoke(
                cli,
                ["--json", "close", "AAPL", "--price", "212.50", "--note", "hit >SMA5"],
                obj={"own_picks_db": db, "json": True},
            )
            self.assertEqual(res.exit_code, 0, res.output)
            data = json.loads(res.output)
            self.assertTrue(data["ok"])
            self.assertEqual(data["pick_id"], pid)
            self.assertAlmostEqual(data["pnl_usd"], 125.0, places=2)
            rows = list_equity_picks(db_path=db)
            self.assertEqual(rows[0]["status"], "CLOSED")
            self.assertAlmostEqual(float(rows[0]["exit_price"]), 212.50, places=2)


class TestPhase0Notifications(unittest.TestCase):
    def _sample_snap(self) -> RadarSnapshot:
        return RadarSnapshot(
            generated_at="2026-11-25T14:00:00Z",
            regime_posture="RISK_OFF",
            vix=26.6,
            vix_3m=25.5,
            vix_term="BACKWARDATION",
            spy_last=718.78,
            spy_chg_1d_pct=-0.5,
            qqq_chg_1d_pct=-2.0,
            next_macro_event="NFP 2026-12-04 (9d)",
            core_allocation=[
                {"ticker": "XLE", "target_weight": 0.26},
                {"ticker": "XLV", "target_weight": 0.26},
                {"ticker": "XLP", "target_weight": 0.26},
                {"ticker": "CASH", "target_weight": 0.22},
            ],
            opportunities=[
                _make_opp("MSFT", "equity_pullback", "LONG", asset_class="stock"),
            ],
            degraded_scanners=[],
        )

    def test_morning_embed_and_action_events_and_zero_clutter_macos(self) -> None:
        snap = self._sample_snap()
        book = {"equity_picks": [], "exit_alerts": ["AAPL (#1): EXIT >SMA5 @ $215.00"]}
        health = {"disk_free_gb": 23.0, "worst": "WARN", "pipes": {"ohlcv": {"status": "OK", "detail": "375 syms"}, "chains": {"status": "STALE", "detail": "6d old"}}}

        embed = build_morning_embed(snap, book, health)
        self.assertIn("RISK_OFF", embed["title"])
        self.assertEqual(len(embed["fields"]), 5)

        pill = format_status_pill(snap, book, health)
        self.assertIn("1 EXIT", pill)

        events = detect_action_events(snap, book, health)
        cats = {e.category for e in events}
        self.assertIn("EXIT_ALERT", cats)
        self.assertIn("ALPHA_SETUP", cats)
        self.assertIn("PIPE_RED", cats)

        # Dedupe filters already-sent keys on the same day
        state = {"sent": {events[0].key: "2026-11-25"}}
        unsent = filter_unsent_events(events, state, today="2026-11-25")
        self.assertEqual(len(unsent), len(events) - 1)

        # macOS zero-clutter: suppresses INFO-only events, coalesces CRITICAL/WARN into ONE banner
        info_only = [AlertEvent("k", "INFO", "ALPHA_SETUP", "T", "B")]
        self.assertIsNone(summarize_for_macos(info_only))
        mac = summarize_for_macos(events)
        assert mac is not None
        self.assertIn("1 Exit Alert", mac[0])
        self.assertTrue(mac[2])  # sound enabled for critical exit alert

        # Events embed bundles cleanly
        ev_emb = build_events_embed(events)
        self.assertEqual(len(ev_emb["fields"]), len(events))

        # Mock runner verifies command construction without firing real OS alert
        seen_cmds = []

        def _fake_runner(cmd):
            seen_cmds.append(cmd)
            m = MagicMock()
            m.returncode = 0
            return m

        ok, backend = send_macos_notification("Options Desk", "Test msg", subtitle="Sub", runner=_fake_runner)
        self.assertTrue(ok)
        self.assertIn(backend, ("terminal-notifier", "osascript"))
        self.assertEqual(len(seen_cmds), 1)


class TestPhase0TuiPickCloseAndAccountToggle(unittest.IsolatedAsyncioTestCase):
    async def test_tui_log_and_close_pick_and_account_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            picks_db = str(Path(td) / "own_picks.db")
            provider = FakeProvider(picks_db, worst_health="OK")
            app = DeskApp(provider=provider, initial_tab="workbench")
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()
                # Toggle account TFSA <-> Taxable via `a`
                await pilot.press("a")
                await pilot.pause()
                # Log a pick with `l`
                await pilot.press("l")
                await pilot.pause()
                await pilot.click("#ql-confirm")
                await pilot.pause()
                self.assertEqual(len(provider.get_book()["equity_picks"]), 1)
                self.assertEqual(provider.get_book()["equity_picks"][0]["status"], "OPEN")

                # Close the pick via `x` -> ClosePickModal
                await pilot.press("x")
                await pilot.pause()
                await pilot.click("#cp-confirm")
                await pilot.pause()
                picks_after = provider.get_book()["equity_picks"]
                self.assertEqual(picks_after[0]["status"], "CLOSED")
                self.assertIsNotNone(picks_after[0]["exit_price"])


if __name__ == "__main__":
    unittest.main()
