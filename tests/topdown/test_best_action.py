"""Offline tests for `0 · BEST ACTION TODAY`, DTE-sorted options strip (0DTE->240D), and calibrated odds."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from rich.console import Console

from src.fill_experiment import log_real_fill
from src.notifications import build_morning_embed
from src.profile import UserProfile, save_profile
from src.radar.schema import RadarSnapshot
from src.topdown.best_action import build_best_action, format_morning_brief_lines, format_prob_badge
from src.topdown.lenses import LENS_BY_ID, LensResult, LensRow
from src.topdown.render import best_action_panel, dte_strip_panel
from src.topdown.web import dte_strip_frame


def _fake_hub(*, spy_above_200: bool = True, pullback_rows=()):
    board_rows = [
        SimpleNamespace(etf="XLE", name="Energy", rank=1, posture="FAVOR", weight=0.26, rs_21d_pct=3.0, rs_63d_pct=8.0, pct_above_200d=80.0, n_members=20),
        SimpleNamespace(etf="XLV", name="Healthcare", rank=2, posture="FAVOR", weight=0.26, rs_21d_pct=2.0, rs_63d_pct=5.0, pct_above_200d=70.0, n_members=25),
        SimpleNamespace(etf="GLD", name="Gold", rank=3, posture="FAVOR", weight=0.26, rs_21d_pct=4.0, rs_63d_pct=10.0, pct_above_200d=None, n_members=0),
    ]
    board = SimpleNamespace(as_of="2026-11-27", rows=board_rows, sleeve_cash=0.22, sleeve_scale=0.78, basket_rv20=0.18)
    macro = SimpleNamespace(spy={"above_200d": spy_above_200, "as_of": "2026-11-27"}, vol={})
    pb_res = LensResult(lens=LENS_BY_ID["pullbacks"], rows=list(pullback_rows))

    def _lens(lid: str):
        return pb_res if lid == "pullbacks" else None

    return SimpleNamespace(board=board, macro=macro, ohlcv={}, lens=_lens)


def _sample_evidence(path: str) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": "2026-11-27",
        "vm_outlook_top3": {
            "calibrated": True,
            "horizons": {
                "3m_elevated": {"n": 110, "p_win": 0.72, "ci95": [0.63, 0.80], "mean_ret": 0.025, "pf": 2.4, "avg_loss": -0.032},
                "12m_all": {"n": 239, "p_win": 0.87, "ci95": [0.82, 0.91], "mean_ret": 0.10, "pf": 5.2, "avg_loss": -0.045},
            },
        },
        "equity_pullback": {
            "regimes": {
                "spy_above_200d": {"n": 800, "p_win": 0.62, "ci95": [0.59, 0.65], "mean_ret": 0.0055, "pf": 1.45, "avg_loss": -0.021},
                "spy_below_200d": {"n": 180, "p_win": 0.48, "ci95": [0.41, 0.55], "mean_ret": -0.004, "pf": 0.88, "avg_loss": -0.038},
            }
        },
        "options_dte": {
            "buckets": {
                "0DTE_BP": {"n": 150, "p_win": 0.81, "ci95": [0.74, 0.86], "mean_roc": -0.01, "avg_loss": -0.22, "pf": 0.92, "excess_entry_limit": -0.015, "verdict": "CLOSED"},
                "1_7D": {"n": 400, "p_win": 0.79, "ci95": [0.75, 0.83], "mean_roc": 0.01, "avg_loss": -0.19, "pf": 1.15, "excess_entry_limit": -0.005, "verdict": "CLOSED"},
                "8_30D": {"n": 600, "p_win": 0.84, "ci95": [0.81, 0.87], "mean_roc": 0.03, "avg_loss": -0.17, "pf": 1.65, "excess_entry_limit": -0.002, "verdict": "CLOSED"},
                "31_60D": {"n": 650, "p_win": 0.86, "ci95": [0.83, 0.88], "mean_roc": 0.045, "avg_loss": -0.16, "pf": 1.95, "excess_entry_limit": 0.001, "verdict": "CLOSED"},
                "60_120D": {"n": 740, "p_win": 0.88, "ci95": [0.85, 0.90], "mean_roc": 0.058, "avg_loss": -0.14, "pf": 2.31, "excess_entry_limit": 0.0018, "verdict": "LEAD_REFERENCE"},
                "120_240D": {"n": 500, "p_win": 0.87, "ci95": [0.84, 0.90], "mean_roc": 0.05, "avg_loss": -0.15, "pf": 2.05, "excess_entry_limit": 0.0005, "verdict": "CLOSED"},
            }
        },
        "implied_vs_realized": {
            "90d": {"20-30d": {"n": 600, "implied_otm_prob": 0.75, "realized_otm_prob": 0.83, "vrp_bias_pp": 8.0}}
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


class BestActionTests(unittest.TestCase):
    def test_prob_badge_requires_sample_and_shows_ci_pf_and_tail_loss(self) -> None:
        self.assertIn("Unproven", format_prob_badge(None, None, 0))
        txt = format_prob_badge(0.88, [0.85, 0.90], 740, pf=2.31, avg_loss=-0.14, mean_ret=0.058, horizon_tag="60–120d")
        self.assertIn("88%", txt)
        self.assertIn("85–90%", txt)
        self.assertIn("n=740", txt)
        self.assertIn("PF 2.31", txt)
        self.assertIn("AvgLoss -14.0%", txt)

    def test_best_action_dte_sort_and_bear_regime_pause(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ev_p = os.path.join(td, "ev.json")
            _sample_evidence(ev_p)
            hub_bull = _fake_hub(spy_above_200=True, pullback_rows=[LensRow(ticker="NVDA", sector_etf="XLK", why="pullback")])
            ba_dte = build_best_action(
                hub_bull, chain_db=os.path.join(td, "missing.db"), evidence_path=ev_p,
                sort_dte_by="dte", now_et=dt.datetime(2026, 11, 25, 12, 0),
            )
            self.assertTrue(ba_dte.evidence_loaded)
            self.assertEqual(ba_dte.execution_window, "OPTIMAL_OPRA_WINDOW")
            self.assertEqual(ba_dte.dte_rows[0].bucket_key, "0DTE_BP")
            self.assertEqual(ba_dte.dte_rows[0].evidence_label, "CLOSED")
            self.assertEqual(ba_dte.dte_rows[-1].bucket_key, "120_240D")
            self.assertEqual(ba_dte.actions[0].evidence_label, "PROVEN")
            self.assertIn("NVDA", ba_dte.actions[1].headline)

            # Evidence sort puts PROMISING 60_120D first
            ba_ev = build_best_action(hub_bull, chain_db=os.path.join(td, "missing.db"), evidence_path=ev_p, sort_dte_by="evidence")
            self.assertEqual(ba_ev.dte_rows[0].bucket_key, "60_120D")
            self.assertEqual(ba_ev.dte_rows[0].evidence_label, "PROMISING")

            # Bear regime (SPY below 200d) pauses stock pullbacks explicitly
            hub_bear = _fake_hub(spy_above_200=False, pullback_rows=[LensRow(ticker="NVDA", sector_etf="XLK", why="pullback")])
            ba_bear = build_best_action(hub_bear, chain_db=os.path.join(td, "missing.db"), evidence_path=ev_p)
            self.assertEqual(ba_bear.actions[1].evidence_label, "WAIT")
            self.assertIn("SPY below 200d", ba_bear.actions[1].headline)

            # Renderers & Streamlit DataFrame produce clean output
            con = Console(record=True, width=120)
            con.print(best_action_panel(ba_dte))
            con.print(dte_strip_panel(ba_dte))
            txt = con.export_text()
            self.assertIn("BEST ACTION TODAY", txt)
            self.assertIn("0DTE", txt)
            self.assertIn("60–120d (O1)", txt)
            df = dte_strip_frame(ba_dte)
            self.assertEqual(len(df), 6)
            self.assertEqual(df.iloc[0]["DTE Bucket"], "0DTE")

    def test_exact_sizing_on_target_o1_fill_meter_and_morning_brief(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ev_p = os.path.join(td, "ev.json")
            _sample_evidence(ev_p)
            prof_p = os.path.join(td, "profile.json")
            fills_p = os.path.join(td, "fills.db")

            # Custom $10,000 TFSA profile with core holdings already on target -> DO NOTHING TODAY
            save_profile(
                UserProfile(
                    tfsa_usd=10000.0,
                    risk_budget_usd=400.0,
                    cash_etf="SGOV",
                    last_rebalance_date=dt.date.today().isoformat(),
                    core_holdings={"XLE": 26.0, "XLV": 26.0, "GLD": 26.0},
                ),
                path=prof_p,
            )
            hub = _fake_hub(spy_above_200=True, pullback_rows=[LensRow(ticker="NVDA", sector_etf="XLK", why="pullback")])
            hub.ohlcv = {
                "XLE": (["2026-11-27"], [100.0], [1e6]),
                "XLV": (["2026-11-27"], [100.0], [1e6]),
                "GLD": (["2026-11-27"], [100.0], [1e6]),
                "NVDA": (["2026-11-27"], [125.0], [5e7]),
            }
            ba = build_best_action(
                hub,
                chain_db=os.path.join(td, "missing.db"),
                evidence_path=ev_p,
                profile_path=prof_p,
                fills_db=fills_p,
                now_et=dt.datetime(2026, 11, 25, 12, 0),
            )
            self.assertIn("DO NOTHING TODAY", ba.actions[0].headline)
            self.assertIn("$10,000", ba.actions[0].sizing_text)
            self.assertIn("NVDA $2,000", ba.actions[1].sizing_text)
            self.assertIn("TEST O1 FILLS 0/20", ba.o1_fill_meter)

            # 5-line morning brief + Discord morning embed integration
            lines = format_morning_brief_lines(ba)
            self.assertEqual(len(lines), 5)
            self.assertIn("BEST ACTION TODAY", lines[0])
            self.assertIn("TEST O1 FILLS", lines[-1])
            snap = RadarSnapshot(
                generated_at="2026-11-27",
                regime_posture="RISK_ON",
                vix=16.5,
                vix_3m=18.2,
                vix_term="CONTANGO",
                spy_last=590.0,
                spy_chg_1d_pct=0.4,
                qqq_chg_1d_pct=0.6,
                next_macro_event="CPI",
            )
            embed = build_morning_embed(snap, {"exit_alerts": [], "equity_picks": []}, {"pipes": {}}, best_action=ba)
            self.assertIn("0 · Best Action Today", embed["fields"][0]["name"])

            # Log 20 bad O1 fills (k = 0.80 > 0.35) -> O1 action escalates to AVOID / PAUSE O1
            for _ in range(20):
                log_real_fill(
                    broker="ibkr",
                    ticker="SPY",
                    structure="O1_bull_put",
                    side="SELL",
                    bid=4.00,
                    ask=6.00,
                    limit_posted=4.20,
                    filled=True,
                    fill_price=4.20,
                    db_path=fills_p,
                )
            ba_fail = build_best_action(
                hub,
                chain_db=os.path.join(td, "missing.db"),
                evidence_path=ev_p,
                profile_path=prof_p,
                fills_db=fills_p,
            )
            self.assertEqual(ba_fail.o1_fill_verdict.get("verdict"), "FAIL")
            self.assertEqual(ba_fail.actions[2].evidence_label, "AVOID")
            self.assertIn("PAUSE O1", ba_fail.actions[2].headline)


if __name__ == "__main__":
    unittest.main()
