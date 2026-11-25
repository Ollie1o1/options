"""Comprehensive offline tests for Phase 1 Profit Engine & Silent Wiring Fixes."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from src.cli.main import cli
from src.equity_playbooks.rebalance import compute_tfsa_rebalance, parse_holdings_spec
from src.equity_playbooks.universe import build_expanded_universe
from src.options_screener import find_credit_spreads
from src.radar.dossier import build_dossier
from src.radar.engine import enrich_with_portfolio_and_iv, run_radar
from src.radar.playbooks import load_playbooks
from src.radar.scanners import (
    compute_regime_header,
    load_kalshi_macro_pulse,
    scan_archived_etf_wide_spreads,
    scan_cached_etf_bear_calls,
    scan_cached_insider_clusters,
    scan_covered_call_holdings,
    scan_holdings_buy_zones,
)
from src.structure.chain_data import load_archived_chain_slice
from src.structure.express import _bs_px_delta, compare_vehicles
from src.structure.types import DetailedView


def _make_chain_archive_db(path: Path, symbol: str = "SPY", spot: float = 600.0) -> None:
    """Populate a realistic chain_snapshots SQLite DB with 30d & 90d smooth-skew smiles."""
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE chain_snapshots (
            symbol TEXT, snap_date TEXT, contract TEXT, type TEXT,
            strike REAL, expiration TEXT, bid REAL, ask REAL,
            bid_size INT, ask_size INT, iv REAL, delta REAL,
            gamma REAL, theta REAL, vega REAL, rho REAL,
            open_interest INT, volume INT
        )
        """
    )
    today = dt.date.today()
    snap = today.isoformat()
    rows = []
    for dte in (35, 90):
        exp = (today + dt.timedelta(days=dte)).isoformat()
        t_yr = dte / 365.0
        for mult in (0.86, 0.89, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.11, 1.14):
            k = round(spot * mult / 5.0) * 5.0
            # Realistic equity put skew: higher IV on downside strikes
            iv = 0.20 + max(0.0, (1.0 - mult) * 0.45) + 0.08 * ((mult - 1.0) ** 2)
            for ot in ("put", "call"):
                px, dl, th = _bs_px_delta(spot, k, t_yr, 0.045, iv, ot)
                half = max(0.03, px * 0.015)
                bid = max(0.05, round(px - half, 2))
                ask = round(px + half, 2)
                rows.append(
                    (symbol, snap, f"{symbol}_{exp}_{ot[0]}_{k}", ot, k, exp, bid, ask,
                     50, 50, round(iv, 4), round(dl, 4), 0.01, round(th, 4), 0.2, 0.01, 5000, 800)
                )
    con.executemany(
        "INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()


def _make_ohlcv_db(path: Path, tickers_spots: dict[str, float]) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE ohlcv (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
    )
    d0 = dt.date(2025, 10, 1)
    dates = []
    cur = d0
    while len(dates) < 265:
        if cur.weekday() < 5:
            dates.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    rows = []
    for sym, end_px in tickers_spots.items():
        # Smooth upward path ending exactly at end_px
        pxs = np.linspace(end_px * 0.82, end_px, len(dates))
        for d, px in zip(dates, pxs):
            rows.append((sym, d, float(px), float(px * 1.01), float(px * 0.99), float(px), 10_000_000.0))
    con.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


class TestPhase1ProfitEngine(unittest.TestCase):
    def test_real_chain_loader_and_svi_rnd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdb = Path(td) / "chain_archive.db"
            _make_chain_archive_db(cdb, "SPY", spot=600.0)

            sl = load_archived_chain_slice("SPY", target_dte=35, spot=600.0, chain_db=str(cdb))
            self.assertIsNotNone(sl)
            self.assertEqual(sl.source, "chain_archive")
            self.assertIsNotNone(sl.atm_iv)
            self.assertAlmostEqual(sl.atm_iv, 0.20, delta=0.03)
            self.assertIsNotNone(sl.skew_25d_vp)
            self.assertGreater(sl.skew_25d_vp, 0.0)  # downside put skew > 0
            self.assertIsNotNone(sl.rnd_density)
            self.assertIn(sl.rnd_source, ("svi", "lognormal_fallback"))

            c = sl.match_contract("put", 565.0)
            self.assertIsNotNone(c)
            self.assertGreater(c.bid, 0.0)
            self.assertGreaterEqual(c.ask, c.bid)

    def test_workbench_uses_real_chains_and_does_not_over_extrapolate_90d_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdb = Path(td) / "chain_archive.db"
            _make_chain_archive_db(cdb, "SPY", spot=600.0)

            view = DetailedView(
                symbol="SPY",
                direction="BULLISH",
                target_move_pct=0.06,
                horizon_days=30,
                risk_budget_usd=600.0,
                account_type="tfsa",
                owns_100_shares=True,
            )
            rows = compare_vehicles(view, spot=600.0, iv=0.20, chain_db=str(cdb))
            by_name = {r.vehicle: r for r in rows}
            self.assertIn("Wide Credit Spread", by_name)
            self.assertIn("Debit Vertical", by_name)
            self.assertIn("Covered Call", by_name)
            # Real chain tight SPY spread -> low friction on Wide Credit Spread
            self.assertLess(by_name["Wide Credit Spread"].friction_pct_of_reward, 20.0)
            # Check that 30d +6% view does NOT produce absurdly huge 90d EV (no 3x linear drift)
            self.assertLess(by_name["Wide Credit Spread"].ev_view_net_usd, by_name["Wide Credit Spread"].max_profit_usd)
            # Budget-aware recommendation: never crown a $10k+ LEAPS for a $600 risk budget
            rec = next((r for r in rows if r.recommended), None)
            self.assertIsNotNone(rec)
            self.assertLessEqual(rec.capital_at_risk_usd, view.risk_budget_usd * 1.15)

    def test_radar_wired_to_real_candidates_insider_longterm_kalshi_and_covered_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            reg = load_playbooks("configs/playbooks.json")
            ohlcv_db = td_p / "eq.db"
            _make_ohlcv_db(ohlcv_db, {"SPY": 600.0, "QQQ": 500.0, "XLK": 230.0, "SMH": 250.0, "GLD": 240.0, "AAPL": 200.0})

            from src.radar.scanners import load_local_ohlcv_safe
            ohlcv = load_local_ohlcv_safe(str(ohlcv_db))

            # 1. Production candidates.db schema (strategy_name, ts, contract_key, round_trip_pct)
            cand_db = td_p / "candidates.db"
            con = sqlite3.connect(str(cand_db))
            con.execute(
                "CREATE TABLE candidates (symbol TEXT, strategy_name TEXT, ts TEXT, expiration TEXT, "
                "strike REAL, contract_key TEXT, premium REAL, ev_net REAL, ev_cost REAL, "
                "round_trip_pct REAL, gate_passed INT, refused_by TEXT)"
            )
            exp = (dt.date.today() + dt.timedelta(days=35)).isoformat()
            ts = dt.datetime.now(dt.timezone.utc).isoformat()
            con.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("QQQ", "Bear Call", ts, exp, None, f"QQQ|{exp}|Bear Call|525/530", 1.25, 18.0, 12.0, 0.096, 1, None),
            )
            con.commit()
            con.close()
            bc_opps = scan_cached_etf_bear_calls(reg, ohlcv=ohlcv, candidates_db=str(cand_db))
            self.assertEqual(len(bc_opps), 1)
            self.assertEqual(bc_opps[0].ticker, "QQQ")
            self.assertIn("525/530", bc_opps[0].setup_summary)

            # 2. Real insider_cache.json format
            ins_json = td_p / "insider_cache.json"
            ins_json.write_text(
                json.dumps(
                    {
                        "AAPL": {
                            "as_of": "2026-11-24",
                            "summary": {"n_buyers": 3, "n_buys": 4, "buy_value": 450000.0},
                        },
                        "SPY": {
                            "as_of": "2026-11-24",
                            "summary": {"n_buyers": 0, "n_buys": 0, "buy_value": 0.0},
                        },
                    }
                )
            )
            ins_opps = scan_cached_insider_clusters(reg, ohlcv, insider_json=str(ins_json))
            self.assertEqual(len(ins_opps), 1)
            self.assertEqual(ins_opps[0].ticker, "AAPL")
            self.assertIn("INSIDERx3", ins_opps[0].badges)

            # 3. longterm_plan.json + skipping already-filled tranches
            plan_json = td_p / "longterm_plan.json"
            plan_json.write_text(
                json.dumps(
                    {
                        "cash_pool_usd": 5000.0,
                        "names": [
                            {"ticker": "AAPL", "thesis": "core", "tranches": [{"level": 205.0, "weight": 0.5}, {"level": 180.0, "weight": 0.5}]}
                        ],
                    }
                )
            )
            lt_db = td_p / "longterm.db"
            hz_opps = scan_holdings_buy_zones(ohlcv, reg, plan_path=str(plan_json), fills_db=str(lt_db))
            self.assertEqual(len(hz_opps), 1)
            self.assertIn("IN_ZONE", hz_opps[0].badges)

            # Record fill of the 205 tranche (100 shares @ $190) -> should hide 205 tranche AND trigger Covered Call!
            from src.longterm.fills import record_fill

            record_fill("AAPL", 205.0, 100.0, 190.0, db_path=str(lt_db))
            hz_after = scan_holdings_buy_zones(ohlcv, reg, plan_path=str(plan_json), fills_db=str(lt_db))
            self.assertEqual(len(hz_after), 0)  # remaining 180 tranche is >5% away from 200

            # 4. Covered Call on >=100-share holding + Archived Test O1 Wide ETF Spread
            cdb = td_p / "chain_archive.db"
            _make_chain_archive_db(cdb, "SPY", spot=600.0)
            cc_opps = scan_covered_call_holdings(
                reg, ohlcv, longterm_db=str(lt_db), own_picks_db=str(td_p / "none.db"), chain_db=str(cdb)
            )
            self.assertEqual(len(cc_opps), 1)
            self.assertEqual(cc_opps[0].ticker, "AAPL")
            self.assertEqual(cc_opps[0].playbook_id, "covered_call_holdings")
            self.assertGreaterEqual(cc_opps[0].extra["strike"], 200.0)

            wide_opps = scan_archived_etf_wide_spreads(reg, ohlcv, symbols=("SPY",), chain_db=str(cdb))
            self.assertEqual(len(wide_opps), 1)
            self.assertEqual(wide_opps[0].ticker, "SPY")
            self.assertIn("ARCHIVE_CHAIN", wide_opps[0].badges)
            self.assertIn("Min Limit", wide_opps[0].setup_summary)

            # 5. Kalshi macro pulse + verified macro calendar
            pm_db = td_p / "predmarkets.db"
            con = sqlite3.connect(str(pm_db))
            con.execute(
                "CREATE TABLE pm_quotes (ticker TEXT, archived_at TEXT, series TEXT, title TEXT, "
                "yes_bid REAL, yes_ask REAL, last REAL, open_interest REAL, close_time TEXT)"
            )
            fut_close = (dt.date.today() + dt.timedelta(days=15)).isoformat() + "T18:59:00Z"
            con.execute(
                "INSERT INTO pm_quotes VALUES (?,?,?,?,?,?,?,?,?)",
                ("KXFEDDECISION-26DEC-H25", "2026-11-24T16:00:00Z", "KXFEDDECISION",
                 "Hike 25bp", 0.91, 0.93, 0.92, 2_000_000.0, fut_close),
            )
            con.execute(
                "INSERT INTO pm_quotes VALUES (?,?,?,?,?,?,?,?,?)",
                ("KXRECSSNBER-27", "2026-11-24T16:00:00Z", "KXRECSSNBER",
                 "Recession 2027", 0.20, 0.22, 0.21, 450_000.0, "2028-01-31T13:25:00Z"),
            )
            con.commit()
            con.close()
            kp = load_kalshi_macro_pulse(str(pm_db))
            self.assertEqual(len(kp), 2)
            self.assertIn("FOMC", kp[0]["label"])
            self.assertAlmostEqual(kp[0]["prob_pct"], 92.0)

            hdr = compute_regime_header(ohlcv, predmarkets_db=str(pm_db))
            self.assertNotEqual(hdr["next_macro_event"], "none")
            self.assertEqual(len(hdr["kalshi_macro"]), 2)

            # 6. Portfolio overlap enrichment
            enriched = enrich_with_portfolio_and_iv(
                ins_opps, ohlcv, chain_db=str(cdb), longterm_db=str(lt_db),
                own_picks_db=str(td_p / "none.db"), paper_db=str(td_p / "none.db"),
            )
            self.assertIn("HOLDING_ALREADY", enriched[0].risk_flags)

            # 7. Dossier reads real `news_archive` table, `analyst_actions`, and `chain_archive.db`
            news_db = td_p / "news_archive.db"
            con = sqlite3.connect(str(news_db))
            con.execute(
                "CREATE TABLE news_archive (id INT, symbol TEXT, headline TEXT, published TEXT, sentiment REAL)"
            )
            con.execute(
                "INSERT INTO news_archive VALUES (1, 'SPY', 'SPY hits fresh quarterly volume peak', '2026-11-24', 0.45)"
            )
            con.commit()
            con.close()

            est_db = td_p / "estimates_archive.db"
            con = sqlite3.connect(str(est_db))
            con.execute(
                "CREATE TABLE analyst_actions (symbol TEXT, grade_date TEXT, firm TEXT, action TEXT, "
                "to_grade TEXT, current_pt REAL, prior_pt REAL)"
            )
            con.execute(
                "INSERT INTO analyst_actions VALUES ('SPY', '2026-11-20', 'Goldman', 'up', 'Buy', 650.0, 610.0)"
            )
            con.commit()
            con.close()

            dos = build_dossier(
                "SPY",
                ohlcv_db=str(ohlcv_db),
                cache_path=str(td_p / "no_cache.json"),
                chain_db=str(cdb),
                news_db=str(news_db),
                estimates_db=str(est_db),
                predmarkets_db=str(pm_db),
            )
            self.assertEqual(dos.chain_source, "chain_archive")
            self.assertIsNotNone(dos.atm_iv)
            self.assertEqual(len(dos.headlines), 1)
            self.assertEqual(len(dos.analyst_actions), 1)
            self.assertEqual(len(dos.kalshi_macro), 2)

    def test_legacy_find_credit_spreads_generates_test_o1_5pct_wide_spreads(self) -> None:
        # 90 DTE SPY put chain with $1-wide strikes around 565 down to 530
        rows = []
        spot = 600.0
        for k in (565.0, 564.0, 563.0, 550.0, 535.0):
            # Realistic 90 DTE put premiums (565P=$11.00, 535P=$3.50 -> $7.50 credit on $30 width = 25% c2w!)
            prem_map = {565.0: 11.0, 564.0: 10.72, 563.0: 10.45, 550.0: 6.80, 535.0: 3.50}
            p = prem_map[k]
            rows.append(
                {
                    "symbol": "SPY",
                    "type": "put",
                    "strike": k,
                    "expiration": "2027-02-19",
                    "T_years": 90.0 / 365.0,
                    "underlying": spot,
                    "delta": -0.25 if k == 565.0 else -0.10,
                    "premium": p,
                    "bid": p - 0.05,
                    "ask": p + 0.05,
                    "quality_score": 0.60,
                }
            )
        df = pd.DataFrame(rows)
        spreads = find_credit_spreads(df)
        self.assertFalse(spreads.empty)
        wide = spreads[(spreads["short_strike"] == 565.0) & (spreads["long_strike"] == 535.0)]
        self.assertEqual(len(wide), 1)
        self.assertTrue(bool(wide.iloc[0]["is_wide_o1_geometry"]))
        self.assertAlmostEqual(float(wide.iloc[0]["spread_width"]), 30.0)

    def test_tfsa_rebalance_calculator_and_cli(self) -> None:
        core = [
            {"ticker": "SMH", "rank": 1, "target_weight": 0.25, "vol_scale": 0.75, "spy_rv20": 0.186, "spot": 250.0},
            {"ticker": "XLK", "rank": 2, "target_weight": 0.25, "vol_scale": 0.75, "spy_rv20": 0.186, "spot": 200.0},
            {"ticker": "GLD", "rank": 3, "target_weight": 0.25, "vol_scale": 0.75, "spy_rv20": 0.186, "spot": 250.0},
            {"ticker": "CASH", "rank": 0, "target_weight": 0.25, "vol_scale": 0.75, "spy_rv20": 0.186, "spot": 1.0},
        ]
        cur = parse_holdings_spec("XLK:10,XLE:20")
        plan = compute_tfsa_rebalance(
            core,
            account_usd=10000.0,
            current_shares=cur,
            spot_lookup={"XLE": 90.0},
            fractional=False,
        )
        self.assertTrue(plan.rebalance_recommended)
        by_tk = {o.ticker: o for o in plan.orders}
        # XLE dropped out of top-3 -> must SELL all 20 shares
        self.assertEqual(by_tk["XLE"].action, "SELL")
        self.assertEqual(by_tk["XLE"].delta_shares, -20.0)
        # SMH target $2,500 @ $250 = BUY 10 shares
        self.assertEqual(by_tk["SMH"].action, "BUY")
        self.assertEqual(by_tk["SMH"].delta_shares, 10.0)
        # Cash reserve ~25% ($2,500+)
        self.assertGreaterEqual(plan.estimated_cash_after_usd, 2500.0)

        # CLI round-trip test
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "eq.db"
            cache = Path(td) / "radar.json"
            _make_ohlcv_db(db, {"SPY": 600.0, "XLK": 230.0, "SMH": 250.0, "GLD": 240.0, "QQQ": 500.0})
            res = runner.invoke(
                cli,
                ["rebalance", "--account", "5000", "--fractional", "--refresh", "--json"],
                obj={"ohlcv_db": str(db), "cache_path": str(cache), "own_picks_db": str(Path(td) / "picks.db")},
            )
            self.assertEqual(res.exit_code, 0, res.output)
            payload = json.loads(res.output)
            self.assertEqual(payload["account_usd"], 5000.0)
            self.assertGreaterEqual(len(payload["orders"]), 3)

    def test_expanded_universe_builder(self) -> None:
        uni = build_expanded_universe()
        self.assertGreaterEqual(len(uni), 200)
        for must_have in ("SPY", "QQQ", "SMH", "XLK", "GLD", "TLT", "AAPL", "NVDA", "RMBS", "RGTI"):
            self.assertIn(must_have, uni)


if __name__ == "__main__":
    unittest.main()
