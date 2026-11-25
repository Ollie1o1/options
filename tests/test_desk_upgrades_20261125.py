"""Unit tests for live desk fixes, OCC OI archiver, TFSA tax/CRA guard, post-mortem tags, Morning Ritual & Sandbox."""
from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path

from src.occ_archive import archive_occ_daily, parse_occ_volume_csv, scan_occ_uoa
from src.own_picks import build_mistake_cost_report, close_equity_pick, log_equity_pick
from src.profile import UserProfile, find_latest_wealthsimple_csv, import_wealthsimple_csv_text
from src.tfsa_tax import compute_tfsa_safety_report, estimate_tfsa_tax_drag


class TestOccArchive(unittest.TestCase):
    def test_parse_and_uoa_scan(self) -> None:
        csv_day1 = (
            "Underlying,Call/Put,Volume,Open Interest\n"
            "SPY,C,100000,500000\n"
            "SPY,P,80000,450000\n"
            "NVDA,C,50000,80000\n"
            "NVDA,P,10000,20000\n"
        )
        csv_day2 = (
            "Underlying,Call/Put,Volume,Open Interest\n"
            "SPY,C,110000,510000\n"
            "SPY,P,90000,460000\n"
            "NVDA,C,90000,120000\n"
            "NVDA,P,10000,25000\n"
        )
        rows = parse_occ_volume_csv(csv_day1, "2026-11-23")
        self.assertEqual(len(rows), 2)
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "occ.db")
            r1 = archive_occ_daily(db_path=db, trade_date="2026-11-23", fetcher=lambda _u: csv_day1)
            r2 = archive_occ_daily(db_path=db, trade_date="2026-11-24", fetcher=lambda _u: csv_day2)
            self.assertEqual(r1["saved"], 2)
            self.assertEqual(r2["saved"], 2)
            sigs = scan_occ_uoa(db_path=db, min_total_vol=100)
            nvda = next(s for s in sigs if s.symbol == "NVDA")
            self.assertEqual(nvda.signal_label, "CALL_OI_SURGE")
            self.assertGreater(nvda.oi_delta_pct or 0, 40.0)


class TestTfsaTaxAndSafety(unittest.TestCase):
    def test_dividend_drag_and_cra_status(self) -> None:
        rows, total_drag, bps = estimate_tfsa_tax_drag({"XLE": 10000.0, "SGOV": 5000.0, "GLD": 5000.0})
        self.assertEqual(len(rows), 3)
        self.assertGreater(total_drag, 40.0)  # XLE ~3.25% * 15% = ~$48.75/yr
        self.assertGreater(bps, 20.0)

        prof = UserProfile(tfsa_usd=10000.0, tfsa_room_cad=7000.0, tfsa_contributed_ytd_cad=6700.0)
        picks = [
            {"date": "2026-02-01", "exit_date": "2026-02-03", "status": "CLOSED", "ticker": "NVDA", "entry_price": 100.0, "shares": 2}
            for _ in range(12)
        ]
        rep = compute_tfsa_safety_report(
            profile=prof,
            core_allocation=[{"ticker": "XLE", "target_weight": 0.6}, {"ticker": "CASH", "target_weight": 0.4}],
            equity_picks=picks,
            today=dt.date(2026, 11, 25),
        )
        self.assertEqual(rep.cra_status, "HIGH_TURNOVER_RISK")
        self.assertEqual(rep.room_remaining_cad, 300.0)


class TestPostMortemAndCsvSync(unittest.TestCase):
    def test_post_mortem_mistake_costs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "picks.db")
            p1 = log_equity_pick("NVDA", 100.0, shares=10, playbook_id="equity_pullback", db_path=db)
            p2 = log_equity_pick("AAPL", 200.0, shares=5, playbook_id="discretionary", db_path=db)
            close_equity_pick(p1, 112.0, close_tag="PLAN_WORKED", db_path=db)
            close_equity_pick(p2, 180.0, close_tag="BROKE_RULES", db_path=db)
            rep = build_mistake_cost_report(db_path=db)
            self.assertEqual(rep["total_closed"], 2)
            by_t = {r["tag"]: r for r in rep["by_tag"]}
            self.assertAlmostEqual(by_t["PLAN_WORKED"]["pnl_usd"], 120.0)
            self.assertAlmostEqual(by_t["BROKE_RULES"]["pnl_usd"], -100.0)

    def test_wealthsimple_csv_text_and_downloads_finder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prof_p = os.path.join(td, "profile.json")
            picks_db = os.path.join(td, "picks.db")
            csv_txt = "date,transaction,symbol,quantity,price\n2026-11-20,BUY,XLE,5,90.00\n2026-11-21,BUY,NVDA,2,140.00\n"
            res = import_wealthsimple_csv_text(csv_txt, filename="ws_test.csv", profile_path=prof_p, own_picks_db=picks_db)
            self.assertEqual(res["core_updates"], 1)
            self.assertEqual(res["picks_logged"], 1)
            # Idempotent re-import
            res2 = import_wealthsimple_csv_text(csv_txt, filename="ws_test.csv", profile_path=prof_p, own_picks_db=picks_db)
            self.assertEqual(res2["skipped_duplicates"], 2)

            ws_file = Path(td) / "wealthsimple_activity_nov.csv"
            ws_file.write_text(csv_txt, encoding="utf-8")
            self.assertEqual(find_latest_wealthsimple_csv(td), str(ws_file))


class TestWebAndMorningRitual(unittest.TestCase):
    def test_express_sandbox_and_ritual(self) -> None:
        from src.cli.morning_ritual import build_morning_ritual, format_morning_ritual_markup
        from src.web_terminal import build_desk_express

        ritual = build_morning_ritual(refresh=False)
        self.assertEqual(len(ritual["steps"]), 5)
        self.assertIn("MORNING RITUAL", format_morning_ritual_markup(ritual))

        sb = build_desk_express(ticker="SPY", direction="BULLISH", target_pct=8.0, days=30, risk_usd=500.0, allow_live=False)
        self.assertEqual(sb["ticker"], "SPY")
        self.assertEqual(len(sb["curve"]["prices"]), 41)
        self.assertGreater(len(sb["vehicles"]), 0)
        self.assertEqual(len(sb["curve"]["vehicle_payoffs"]), len(sb["vehicles"]))

    def test_alpha_formatters_occ_and_cal(self) -> None:
        from src.alpha.formatters import format_alpha_screen

        t_occ, b_occ = format_alpha_screen("OCC")
        self.assertIn("OCC", t_occ)
        self.assertTrue(len(b_occ) > 10)
        t_cal, b_cal = format_alpha_screen("CAL")
        self.assertIn("CALENDAR", t_cal)
        self.assertTrue(len(b_cal) > 10)


if __name__ == "__main__":
    unittest.main()
