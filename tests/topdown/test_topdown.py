"""Offline tests for src.topdown (Discover: ① market, ② sectors, ③ lenses, ④ evidence card).

Everything runs on `src.topdown.synthetic` data and temp directories; the
loaders that would read the real dolt/estimates archives are patched out.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest import mock

import numpy as np
from rich.console import Console

from src.outlook.vm_rotation import live_target
from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.topdown import render as R
from src.topdown.card import build_evidence_card, probability_label
from src.topdown.hub import build_hub
from src.topdown.lenses import LENSES, LensContext, build_context, run_lens, tags_by_ticker
from src.topdown.macro import fred_readings, ratio_reading, refresh_fred_cache, vol_regime
from src.topdown.peers import peer_strip
from src.topdown.sectors import build_sector_board, sector_members
from src.topdown.synthetic import synthetic_ohlcv, write_cboe_dir

_NO_EARN = mock.patch("src.radar.scanners._load_upcoming_earnings_map", return_value={})
_NO_CRASH = mock.patch("src.radar.scanners._load_recent_post_earnings_crashes", return_value={})
_NO_REV = mock.patch("src.radar.scanners._load_latest_rev_breadth", return_value={})


def _snapshot(opps=()) -> RadarSnapshot:
    return RadarSnapshot(
        generated_at="2026-11-24T21:00:00Z", regime_posture="NEUTRAL", vix=None, vix_3m=None,
        vix_term="UNKNOWN", spy_last=None, spy_chg_1d_pct=None, qqq_chg_1d_pct=None,
        next_macro_event="none", opportunities=list(opps),
    )


def _opp(ticker: str, playbook: str = "equity_pullback", tier: str = "LEAD") -> Opportunity:
    return Opportunity(
        ticker=ticker, asset_class="stock", playbook_id=playbook, playbook_label="Uptrend Pullback (Shares)",
        evidence_tier=tier, direction="LONG", spot=100.0, chg_1d_pct=-1.0, chg_5d_pct=-4.0,
        horizon_label="3-10d", setup_summary="RSI2=4 · +6% >200MA", as_of="2026-11-24T21:00:00Z",
    )


def _dossier(ticker: str, **kw: Any) -> TickerDossier:
    base: Dict[str, Any] = dict(
        ticker=ticker, as_of="2026-11-24T21:00:00Z", spot=100.0, chg_1d_pct=-1.0, chg_5d_pct=-3.0,
        chg_21d_pct=2.0, mom_12_1_pct=20.0, dist_sma20_pct=-1.0, dist_sma50_pct=1.0, dist_sma200_pct=6.0,
        rsi_2=4.0, rsi_14=40.0, rv_20_ann=25.0, sector_etf="XLE", rnd_prob_up_5pct=0.2,
        rnd_prob_dn_5pct=0.2, implied_move_1s_pct=7.0,
    )
    base.update(kw)
    return TickerDossier(**base)


class SectorBoardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ohlcv = synthetic_ohlcv()

    def test_board_is_ordered_by_the_validated_rank_and_matches_the_sleeve(self) -> None:
        board = build_sector_board(self.ohlcv)
        self.assertEqual(len(board.rows), 16)
        self.assertEqual([r.rank for r in board.rows], list(range(1, 17)))
        tgt = live_target(self.ohlcv)
        assert tgt is not None
        self.assertEqual({r.etf: r.weight for r in board.rows if r.weight > 0},
                         {k: round(v, 4) for k, v in tgt.weights.items()})
        for r in board.rows:
            self.assertEqual(r.posture == "FAVOR", r.weight > 0)
        self.assertAlmostEqual(board.sleeve_cash or 0.0, round(tgt.cash, 4))

    def test_breadth_counts_members_and_excludes_stale_ones(self) -> None:
        board = build_sector_board(self.ohlcv)
        xle = board.row("XLE")
        assert xle is not None
        self.assertEqual(set(sector_members("XLE", self.ohlcv)), {"XOM", "CVX", "COP", "OXY"})
        self.assertEqual(xle.n_members, 4)
        self.assertIsNotNone(xle.pct_above_200d)
        stale = synthetic_ohlcv(lag={"OXY": 10})
        xle2 = build_sector_board(stale).row("XLE")
        assert xle2 is not None
        self.assertEqual((xle2.n_members, xle2.n_stale), (3, 1))

    def test_xlk_breadth_includes_semis_and_smh_is_the_semis_subset(self) -> None:
        self.assertTrue({"NVDA", "AMD", "MU"} <= set(sector_members("XLK", self.ohlcv)))
        self.assertEqual(set(sector_members("SMH", self.ohlcv)), {"NVDA", "AMD", "MU"})
        self.assertEqual(sector_members("GLD", self.ohlcv), [])

    def test_relative_strength_is_etf_minus_spy(self) -> None:
        board = build_sector_board(self.ohlcv)
        r = board.row("XLK")
        assert r is not None and r.ret_63d_pct is not None and r.rs_63d_pct is not None
        spy = self.ohlcv["SPY"][1]
        spy63 = (spy[-1] / spy[-64] - 1.0) * 100.0
        self.assertAlmostEqual(r.rs_63d_pct, round(r.ret_63d_pct - spy63, 2), delta=0.02)


class MacroTests(unittest.TestCase):
    def test_vix_term_structure_and_staleness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            write_cboe_dir(td, vix=30.0, vix3m=25.0)
            import datetime as dt

            v = vol_regime(base_dir=td, today=dt.date(2026, 11, 25))
            self.assertEqual(v["vix"], 30.0)
            self.assertEqual(v["term"], "BACKWARDATION")
            self.assertAlmostEqual(v["term_ratio"], 1.2, places=3)
            self.assertFalse(v["stale"])
            self.assertTrue(vol_regime(base_dir=td, today=dt.date(2026, 12, 15))["stale"])
            self.assertIsNone(vol_regime(base_dir=os.path.join(td, "nope"))["vix"])

    def test_ratio_reading_math_and_missing_symbol(self) -> None:
        o = synthetic_ohlcv()
        r = ratio_reading(o, "RSP", "SPY", "x", "y")
        n, d = o["RSP"][1], o["SPY"][1]
        ratio = n / d
        self.assertAlmostEqual(r.chg_63d_pct or 0.0, round((ratio[-1] / ratio[-64] - 1) * 100, 2), places=6)
        miss = ratio_reading(o, "SPHB", "SPLV", "x", "y")
        self.assertIsNone(miss.chg_63d_pct)
        self.assertIn("not in price archive", miss.missing or "")

    def test_fred_cache_refresh_is_atomic_and_keeps_old_entries_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "fred.json")
            obs = [("2026-10-01", 3.1), ("2026-11-20", 3.4)]
            with mock.patch("src.topdown.macro.fetch_fred_series", return_value=obs):
                st = refresh_fred_cache(path)
            self.assertTrue(all(v == "ok" for v in st.values()))
            with mock.patch("src.topdown.macro.fetch_fred_series", side_effect=OSError("offline")):
                st2 = refresh_fred_cache(path)
            self.assertTrue(all(v.startswith("error") for v in st2.values()))
            with open(path) as fh:
                cache = json.load(fh)
            rows = fred_readings(cache)
            self.assertEqual(rows[0]["value"], 3.4)
            self.assertAlmostEqual(rows[0]["chg_1m"], 0.3, places=6)
            self.assertFalse(os.path.exists(path + ".tmp"))


class PeerTests(unittest.TestCase):
    def test_peers_are_same_sector_ordered_by_correlation(self) -> None:
        o = synthetic_ohlcv()
        # make CVX a near-clone of XOM so it must rank first
        ds, cs, vs = o["XOM"]
        o["CVX"] = (ds, cs * (1.0 + 0.0005 * np.sin(np.arange(len(cs)))), vs)
        strip = peer_strip("XOM", o, k=3, with_iv=False)
        self.assertEqual(strip.rows[0].ticker, "XOM")
        self.assertTrue(strip.rows[0].is_subject)
        self.assertEqual(strip.rows[1].ticker, "CVX")
        corrs = [r.corr_126d for r in strip.rows[1:]]
        self.assertEqual(corrs, sorted(corrs, reverse=True))
        self.assertIn("XLE", strip.method)
        self.assertIn("ret_63d_pct", strip.standing)
        self.assertRegex(strip.standing["ret_63d_pct"], r"^\d/\d$")

    def test_thin_sector_is_topped_up_and_says_so(self) -> None:
        o = synthetic_ohlcv()
        strip = peer_strip("LLY", o, k=4, with_iv=False)  # XLV has only LLY, PFE here
        self.assertEqual(len(strip.rows), 5)
        self.assertIn("outside the sector", strip.method)
        self.assertTrue(any(not r.in_sector for r in strip.rows[1:]))


class LensTests(unittest.TestCase):
    def setUp(self) -> None:
        self.o = synthetic_ohlcv()
        self.board = build_sector_board(self.o)

    def _ctx(self, **kw: Any) -> LensContext:
        return LensContext(ohlcv=self.o, board=self.board, **kw)

    def test_every_lens_declares_label_evidence_and_sort(self) -> None:
        labels = {"PROVEN", "PROMISING", "TESTING", "WATCHLIST", "CONTEXT", "AVOID"}
        for lz in LENSES:
            self.assertIn(lz.label, labels)
            self.assertTrue(lz.evidence and lz.source and lz.sort_note and lz.question)

    def test_core_rotation_lists_exactly_the_held_etfs(self) -> None:
        res = run_lens("core_rotation", self._ctx())
        self.assertEqual({r.ticker for r in res.rows}, {r.etf for r in self.board.rows if r.weight > 0})

    def test_leaders_are_uptrending_and_sorted_by_12_1(self) -> None:
        res = run_lens("leaders", self._ctx())
        for row in res.rows:
            c = self.o[row.ticker][1]
            self.assertGreater(c[-1], float(np.mean(c[-200:])))
            self.assertGreater(c[-22] / c[-253] - 1.0, 0.0)
        ms = [r.metric for r in res.rows]
        self.assertEqual(ms, sorted(ms, reverse=True))

    def test_revisions_threshold_and_landmines(self) -> None:
        ctx = self._ctx(rev_breadth={"XOM": 0.6, "CVX": 0.2, "XLE": 0.9}, earnings_days={"AAPL": 3, "MSFT": 30},
                        crash_flags={"ORCL": "POST_EARN_CRASH(-9.0%)"})
        self.assertEqual([r.ticker for r in run_lens("revisions", ctx).rows], ["XOM"])  # funds excluded
        lm = run_lens("landmines", ctx)
        self.assertEqual([r.ticker for r in lm.rows], ["ORCL", "AAPL"])

    def test_pullbacks_come_from_the_snapshot_and_carry_sector_posture(self) -> None:
        ctx = self._ctx(snapshot=_snapshot([_opp("XOM"), _opp("SPY", "vol_managed_outlook", "VALIDATED")]))
        res = run_lens("pullbacks", ctx)
        self.assertEqual([r.ticker for r in res.rows], ["XOM"])
        self.assertEqual(run_lens("pullbacks", self._ctx()).note, "No Radar snapshot cached — press Refresh.")

    def test_unknown_lens_raises_and_broken_inputs_are_isolated(self) -> None:
        with self.assertRaises(KeyError):
            run_lens("nope", self._ctx())
        bad = self._ctx()
        bad.board = None  # type: ignore[assignment]
        res = run_lens("core_rotation", bad)
        self.assertIn("unavailable", res.note)

    def test_tags_by_ticker(self) -> None:
        ctx = self._ctx(rev_breadth={"XOM": 0.6}, earnings_days={"XOM": 2})
        tags = tags_by_ticker([run_lens("revisions", ctx), run_lens("landmines", ctx)])
        self.assertEqual(tags["XOM"], [("REV", "TESTING"), ("MINE", "AVOID")])


class EvidenceCardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.o = synthetic_ohlcv()
        self.board = build_sector_board(self.o)

    def _card(self, dossier: TickerDossier, opps=()):
        return build_evidence_card(dossier.ticker, ohlcv=self.o, board=self.board, snapshot=_snapshot(opps),
                                   dossier=dossier, chain_db="/nonexistent.db", with_peer_iv=False)

    def test_verdict_vocabulary(self) -> None:
        sig = _opp("XOM")
        self.assertEqual(self._card(_dossier("XOM", active_signals=[sig])).verdict, "TESTED SETUP")
        warn = self._card(_dossier("XOM", active_signals=[sig], days_to_earnings=3, earnings_date="2026-11-27"))
        self.assertEqual(warn.verdict, "SETUP + WARNINGS")
        self.assertTrue(any("Earnings in 3d" in w for w in warn.warnings))
        self.assertEqual(self._card(_dossier("XOM", days_to_earnings=2)).verdict, "AVOID FOR NOW")
        self.assertEqual(self._card(_dossier("XOM")).verdict, "NO TESTED SETUP")
        ctx_sig = _opp("XOM", "insider_cluster", "CONTEXT")
        self.assertEqual(self._card(_dossier("XOM", active_signals=[ctx_sig])).verdict, "NO TESTED SETUP")

    def test_held_sector_is_pointed_out_when_no_setup(self) -> None:
        held = next(r.etf for r in self.board.rows if r.weight > 0)
        card = self._card(_dossier("XOM", sector_etf=held))
        self.assertTrue(any(f"{held} IS held" in r for r in card.reasons))

    def test_provenance_and_probability_labels(self) -> None:
        card = self._card(_dossier("XOM"))
        sections = [p.section for p in card.provenance]
        for s in ("Price, trend, levels, RV", "Sector posture & rank", "Peers", "Options IV / skew", "±5% odds (30d)"):
            self.assertIn(s, sections)
        self.assertTrue(probability_label(_dossier("XOM")).startswith("MODEL"))
        mkt = _dossier("XOM", chain_source="chain_archive", rnd_source="svi", chain_as_of="2026-11-17")
        self.assertTrue(probability_label(mkt).startswith("market-implied"))
        self.assertIn(card.verdict_style, ("green", "amber", "red", "grey"))
        json.dumps(card.to_dict(), default=str)


class HubAndRenderingTests(unittest.TestCase):
    """Render to a real Rich console and assert on what a reader would see."""

    def _hub(self, td: str, snapshot=None):
        write_cboe_dir(os.path.join(td, "cboe"))
        with _NO_EARN, _NO_CRASH, _NO_REV:
            return build_hub(ohlcv=synthetic_ohlcv(), snapshot=snapshot or _snapshot([_opp("XOM")]),
                             fred_path=os.path.join(td, "fred.json"), cboe_dir=os.path.join(td, "cboe"),
                             estimates_db=os.path.join(td, "est.db"),
                             own_picks_db=os.path.join(td, "picks.db"), longterm_db=os.path.join(td, "lt.db"))

    @staticmethod
    def _text(renderable: Any, width: int = 150) -> str:
        buf = io.StringIO()
        Console(file=buf, width=width, no_color=True, force_terminal=False).print(renderable)
        return buf.getvalue()

    def test_overview_shows_the_four_numbered_steps_labels_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._hub(td)
            out = self._text(R.hub_overview(hub))
        for needle in ("① MARKET", "② SECTORS", "③ IDEAS", "Read:", "VIX", "This month's TFSA core sleeve",
                       "PROVEN", "PROMISING", "TESTING", "WATCHLIST", "AVOID", "Source ·", "context only",
                       "Data:", "opt card"):
            self.assertIn(needle, out)
        self.assertEqual(out.count("Source ·"), 2)  # market + sectors panels each end with a source line

    def test_lens_sector_and_ledger_panels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._hub(td)
            for res in hub.lenses:
                out = self._text(R.lens_panel(res))
                self.assertIn(res.lens.label, out)
                self.assertIn("Evidence:", out)
                self.assertIn("Sorted by:", out)
            sec = self._text(R.sector_members_panel(hub, "XLE"))
            self.assertIn("XLE Energy — constituents", sec)
            for t in ("XOM", "CVX", "COP", "OXY"):
                self.assertIn(t, sec)
        ledger = self._text(R.null_ledger_panel())
        self.assertIn("does NOT work here", ledger)
        self.assertIn("Analyst upgrades/downgrades", ledger)

    def test_card_renders_six_sections_with_ages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._hub(td)
            card = hub.card("XOM", dossier=_dossier("XOM", active_signals=[_opp("XOM")]),
                            chain_db="/nonexistent.db", with_peer_iv=False)
            out = self._text(R.card_renderable(card))
        for needle in ("1 Verdict", "2 Price structure", "3 Peers", "4 Options", "5 Events",
                       "7 Where every number came from", "6 Portfolio fit", "TESTED SETUP", "Corr w/ XOM", "Beta vs SPY",
                       "MODEL, not market-implied"):
            self.assertIn(needle, out)

    def test_hub_json_is_serialisable_and_reports_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._hub(td)
            d = hub.to_dict()
        json.dumps(d, default=str)
        self.assertEqual(d["freshness"]["behind"], 0)
        self.assertEqual(len(d["board"]["rows"]), 16)
        self.assertEqual(len(d["lenses"]), len(LENSES))

    def test_build_context_is_failure_isolated(self) -> None:
        o = synthetic_ohlcv()
        board = build_sector_board(o)
        boom = mock.patch("src.radar.scanners._load_upcoming_earnings_map", side_effect=RuntimeError("x"))
        with boom, _NO_CRASH, _NO_REV:
            ctx = build_context(o, board, None, estimates_db="/nonexistent.db")
        self.assertEqual(ctx.earnings_days, {})


if __name__ == "__main__":
    unittest.main()


class WebFramesAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        write_cboe_dir(os.path.join(self.td.name, "cboe"))
        with _NO_EARN, _NO_CRASH, _NO_REV:
            self.hub = build_hub(ohlcv=synthetic_ohlcv(), snapshot=_snapshot([_opp("XOM")]),
                                 fred_path=os.path.join(self.td.name, "fred.json"),
                                 cboe_dir=os.path.join(self.td.name, "cboe"),
                                 estimates_db=os.path.join(self.td.name, "est.db"))

    def tearDown(self) -> None:
        self.td.cleanup()

    def test_web_frames_have_reader_facing_columns(self) -> None:
        from src.topdown import web

        bf = web.board_frame(self.hub)
        self.assertEqual(len(bf), 16)
        for col in ("Rank", "ETF", "Call", "Sleeve %", "RS vs SPY 3m %", "Stocks >200d %"):
            self.assertIn(col, bf.columns)
        self.assertEqual(list(bf["Rank"]), list(range(1, 17)))
        mf = web.members_frame(self.hub, "XLE")
        self.assertEqual(set(mf["Ticker"]), {"XOM", "CVX", "COP", "OXY"})
        self.assertTrue(mf["12-1m %"].is_monotonic_decreasing)
        card = self.hub.card("XOM", dossier=_dossier("XOM"), chain_db="/nonexistent.db", with_peer_iv=False)
        self.assertIn("Corr w/ XOM", web.peers_frame(card).columns)
        self.assertGreater(len(web.provenance_frame(card)), 5)
        self.assertEqual(len(web.null_ledger_frame()), len(__import__("src.topdown.lenses", fromlist=["x"]).NULL_LEDGER))
        self.assertIn("PROVEN", web.chip_html("PROVEN"))

    def test_cli_discover_and_card(self) -> None:
        from click.testing import CliRunner

        from src.cli.main import cli

        runner = CliRunner()
        with mock.patch("src.cli.discover._hub", return_value=self.hub):
            out = runner.invoke(cli, ["discover", "--json"])
            self.assertEqual(out.exit_code, 0, out.output)
            payload = json.loads(out.output)
            self.assertEqual(len(payload["board"]["rows"]), 16)
            for args, needle in ((["discover"], "② SECTORS"), (["discover", "--sector", "XLE"], "constituents"),
                                 (["discover", "--lens", "leaders"], "Sector leaders"),
                                 (["discover", "--ledger"], "does NOT work")):
                res = runner.invoke(cli, ["--no-color"] + args)
                self.assertEqual(res.exit_code, 0, res.output)
                self.assertIn(needle, res.output)
            bad = runner.invoke(cli, ["discover", "--lens", "nope"])
            self.assertNotEqual(bad.exit_code, 0)
            with mock.patch("src.radar.dossier.build_dossier", return_value=_dossier("XOM")):
                card = runner.invoke(cli, ["--no-color", "card", "XOM", "--no-peer-iv"])
            self.assertEqual(card.exit_code, 0, card.output)
            self.assertIn("Where every number came from", card.output)
