"""Offline unit tests for Phase 0 & Phase 1 upgrades:
Alpaca options/news/corp-actions, Databento OPRA cost-guard, FINRA RegSHO/ATS flow,
FRED/ALFRED vintages, 3-tier live quote router, multi-expiry surface search,
SEC EDGAR bulk fundamentals + CIK resolution, and portfolio-aware discovery.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src import alpaca_data as ad
from src import databento_opra as dbo
from src import finra_flow as ff
from src.alpha import edgar_events as ee
from src.alpha import macro_plumbing as mp
from src.structure.chain_data import ChainContract, ChainSlice, load_live_chain_slices
from src.topdown.lenses import LENS_BY_ID, LensRow
from src.topdown.portfolio_fit import Holding, rank_rows_for_diversification
from src.topdown.synthetic import business_days
from src.tui.options_engine import _live_calendars_from_slices, scan_surface_verticals


class _FakeResp:
    def __init__(self, status_code: int = 200, json_body=None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_body
        self.text = text if text else (json.dumps(json_body) if json_body is not None else "")

    def json(self):
        return self._json


class AlpacaDataTests(unittest.TestCase):
    def test_option_snapshots_pagination_and_provenance(self) -> None:
        pages = [
            {
                "snapshots": {
                    "SPY270115P00570000": {
                        "latestQuote": {"bp": 4.10, "ap": 4.30, "bs": 25, "as": 30, "t": "2026-11-27T15:00:00Z"},
                        "latestTrade": {"t": "2026-11-27T14:58:00Z"},
                        "dailyBar": {"v": 412},
                        "impliedVolatility": 0.21,
                        "greeks": {"delta": -0.25, "gamma": 0.012, "theta": -0.05, "vega": 0.45, "rho": -0.1},
                    }
                },
                "next_page_token": "tok2",
            },
            {
                "snapshots": {
                    "SPY270115P00540000": {
                        "latestQuote": {"bp": 1.40, "ap": 1.55, "bs": 10, "as": 12, "t": "2026-11-27T15:00:00Z"},
                        "impliedVolatility": 0.24,
                        "greeks": {"delta": -0.11},
                    }
                },
                "next_page_token": None,
            },
        ]
        idx = {"n": 0}

        def _fake_get(url, headers=None, params=None, timeout=20):
            i = idx["n"]
            idx["n"] += 1
            return _FakeResp(200, pages[i])

        rows = ad.fetch_option_chain_snapshots(
            "SPY", spot=600.0, today=dt.date(2026, 11, 27), get=_fake_get, headers={"k": "v"}
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], "alpaca_indicative")
        self.assertEqual(rows[1]["strike"], 570.0)
        self.assertEqual(rows[1]["bid_size"], 25.0)
        self.assertAlmostEqual(rows[1]["delta"], -0.25)

    def test_news_archive_and_ex_div_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "news.db")
            items = [
                {
                    "id": 101,
                    "created_at": "2026-11-25T14:00:00Z",
                    "headline": "Apple guides higher",
                    "summary": "Strong quarter",
                    "source": "benzinga",
                    "symbols": ["AAPL", "SPY"],
                }
            ]
            n = ad.save_news_archive(items, db_path=db)
            self.assertEqual(n, 1)
            con = sqlite3.connect(db)
            try:
                cnt = con.execute("SELECT COUNT(*) FROM alpaca_news_symbols WHERE symbol='AAPL'").fetchone()[0]
            finally:
                con.close()
            self.assertEqual(cnt, 1)

        def _fake_ca(url, headers=None, params=None, timeout=15):
            return _FakeResp(
                200,
                {
                    "corporate_actions": {
                        "cash_dividends": [
                            {"symbol": "SPY", "ex_date": "2026-12-18", "rate": "1.75"},
                            {"symbol": "SPY", "ex_date": "2027-03-19", "rate": "1.75"},
                        ]
                    }
                },
            )

        ex = ad.upcoming_ex_dividend_dates(["SPY"], get=_fake_ca, headers={"k": "v"})
        self.assertEqual(ex["SPY"], ("2026-12-18", 1.75))


class DatabentoOpraTests(unittest.TestCase):
    def test_dry_run_never_downloads_and_budget_guard_refuses_overrun(self) -> None:
        post_called = {"n": 0}

        def _get(url, auth=None, params=None, timeout=30):
            return _FakeResp(200, text="0.75")

        def _post(url, auth=None, data=None, timeout=90):
            post_called["n"] += 1
            return _FakeResp(200, text="")

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "opra.db")
            res = dbo.fetch_opra_snapshots(
                ["SPY"], ["2025-11-03", "2025-11-04"], dry_run=True, max_usd=1.0, db_path=db,
                api_key="db-test", get=_get, post=_post
            )
            self.assertAlmostEqual(res["total_est_usd"], 1.50, places=4)
            self.assertEqual(post_called["n"], 0)

            with self.assertRaises(RuntimeError):
                dbo.fetch_opra_snapshots(
                    ["SPY"], ["2025-11-03", "2025-11-04"], dry_run=False, max_usd=1.0, db_path=db,
                    api_key="db-test", get=_get, post=_post
                )
            self.assertEqual(post_called["n"], 0)

    def test_download_parses_occ_and_logs_spend(self) -> None:
        csv_body = (
            "ts_recv,ts_event,symbol,bid_px_00,ask_px_00,bid_sz_00,ask_sz_00\n"
            "2025-11-03T17:00:00Z,2025-11-03T15:10:00Z,SPY   260116P00550000,4.20,4.35,40,35\n"
        )
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "opra.db")
            res1 = dbo.fetch_opra_snapshots(
                ["SPY"], ["2025-11-03"], hhmm_et="12:00", dry_run=False, max_usd=1.0, db_path=db, api_key="db-test",
                get=lambda *a, **k: _FakeResp(200, text="0.015"),
                post=lambda *a, **k: _FakeResp(200, text=csv_body),
            )
            # Same resting quote at 14:00 must NOT overwrite 12:00 row
            res2 = dbo.fetch_opra_snapshots(
                ["SPY"], ["2025-11-03"], hhmm_et="14:00", dry_run=False, max_usd=1.0, db_path=db, api_key="db-test",
                get=lambda *a, **k: _FakeResp(200, text="0.015"),
                post=lambda *a, **k: _FakeResp(200, text=csv_body),
            )
            self.assertEqual(res1["rows_written"], 1)
            self.assertEqual(res2["rows_written"], 1)
            self.assertAlmostEqual(dbo.cumulative_spend_usd(db), 0.030, places=4)
            con = sqlite3.connect(db)
            try:
                clocks = sorted(r[0] for r in con.execute("SELECT clock_et FROM opra_chain").fetchall())
            finally:
                con.close()
            self.assertEqual(clocks, ["12:00", "14:00"])


class FinraFlowTests(unittest.TestCase):
    def test_regsho_and_ats_aggregation_and_signals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "finra.db")
            reg_rows = []
            for d_off in range(12):
                d = (dt.date(2026, 11, 1) + dt.timedelta(days=d_off)).isoformat()
                sv = 680_000 if d_off == 11 else 400_000
                reg_rows.append(
                    {
                        "tradeReportDate": d,
                        "securitiesInformationProcessorSymbolIdentifier": "NVDA",
                        "shortParQuantity": sv,
                        "totalParQuantity": 1_000_000,
                    }
                )
            r1 = ff.fetch_regsho_for_symbols(["NVDA"], start_date="2026-11-01", poster=lambda u, p: reg_rows, db_path=db)
            self.assertEqual(r1["saved"], 12)
            ats_rows = [
                {"weekStartDate": "2026-11-09", "issueSymbolIdentifier": "NVDA", "totalWeeklyShareQuantity": 500_000, "totalWeeklyTradeCount": 1000}
            ]
            r2 = ff.fetch_ats_for_symbols(["NVDA"], start_date="2026-11-01", poster=lambda u, p: ats_rows, db_path=db)
            self.assertEqual(r2["saved"], 1)
            sigs = ff.scan_finra_flow_signals(db_path=db)
            self.assertEqual(len(sigs), 1)
            self.assertEqual(sigs[0].signal_label, "HIGH_SHORT_VOL")
            self.assertEqual(sigs[0].ats_avg_block, 500.0)


class FredAlfredTests(unittest.TestCase):
    def test_alfred_vintage_uses_api_and_requires_key(self) -> None:
        seen_urls: list[str] = []

        def _fake_get(url: str) -> str:
            seen_urls.append(url)
            return json.dumps({"observations": [{"date": "2024-01-10", "value": "7680000"}, {"date": "2024-01-17", "value": "."}]})

        pts = mp.fetch_alfred_vintage("WALCL", as_of_date="2024-01-20", api_key="k123", fetcher=_fake_get)
        self.assertEqual(pts, [("2024-01-10", 7680000.0)])
        self.assertIn("realtime_start=2024-01-20", seen_urls[0])
        self.assertIn("realtime_end=2024-01-20", seen_urls[0])
        with self.assertRaises(RuntimeError):
            mp.fetch_alfred_vintage("WALCL", as_of_date="2024-01-20", api_key="", fetcher=_fake_get)


class LiveRouterAndSurfaceTests(unittest.TestCase):
    def test_three_tier_router_falls_back_from_cboe_to_alpaca(self) -> None:
        today = dt.date.today()
        e1 = (today + dt.timedelta(days=35)).isoformat()
        e2 = (today + dt.timedelta(days=90)).isoformat()
        apca_rows = [
            {"symbol": "SPY", "expiration": e1, "type": "put", "strike": 570.0, "bid": 4.0, "ask": 4.2, "bid_size": 15, "ask_size": 20, "iv": 0.22, "delta": -0.25, "open_interest": 1000, "volume": 200, "spot": 600.0, "source": "alpaca_indicative"},
            {"symbol": "SPY", "expiration": e1, "type": "put", "strike": 540.0, "bid": 1.3, "ask": 1.45, "bid_size": 12, "ask_size": 12, "iv": 0.25, "delta": -0.10, "open_interest": 800, "volume": 100, "spot": 600.0, "source": "alpaca_indicative"},
            {"symbol": "SPY", "expiration": e1, "type": "call", "strike": 600.0, "bid": 12.0, "ask": 12.3, "bid_size": 20, "ask_size": 20, "iv": 0.21, "delta": 0.51, "open_interest": 2000, "volume": 500, "spot": 600.0, "source": "alpaca_indicative"},
            {"symbol": "SPY", "expiration": e2, "type": "call", "strike": 600.0, "bid": 18.0, "ask": 18.4, "bid_size": 15, "ask_size": 15, "iv": 0.18, "delta": 0.54, "open_interest": 1500, "volume": 300, "spot": 600.0, "source": "alpaca_indicative"},
        ]
        diag: dict = {}
        with patch("src.cboe_client.fetch_chain", return_value=[]), \
             patch("src.alpaca_data.has_credentials", return_value=True), \
             patch("src.alpaca_data.fetch_option_chain_snapshots", return_value=apca_rows):
            slices = load_live_chain_slices("SPY", min_dte=14, max_dte=125, max_expirations=2, spot=600.0, diagnostics=diag)
        self.assertEqual(len(slices), 2)
        self.assertEqual(diag["provider"], "alpaca_indicative")
        cals = _live_calendars_from_slices(slices, spot=600.0)
        self.assertEqual(len(cals), 1)
        self.assertEqual(cals[0].strategy_key, "calendar")

    def test_surface_vertical_search_flags_thin_depth_and_ex_div(self) -> None:
        sl = ChainSlice(
            symbol="SPY", expiration="2026-12-18", dte=35, snap_date="2026-11-27", source="cboe", spot=600.0,
            puts=[
                ChainContract(580.0, "put", 5.00, 5.15, 5.075, iv=0.22, delta=-0.30, open_interest=1000, bid_size=1, ask_size=1),
                ChainContract(570.0, "put", 3.50, 3.60, 3.55, iv=0.23, delta=-0.25, open_interest=1500, bid_size=25, ask_size=30),
                ChainContract(550.0, "put", 1.40, 1.50, 1.45, iv=0.26, delta=-0.12, open_interest=1200, bid_size=20, ask_size=20),
                ChainContract(540.0, "put", 0.90, 1.00, 0.95, iv=0.27, delta=-0.08, open_interest=900, bid_size=18, ask_size=18),
            ],
        )
        cands = scan_surface_verticals([sl], spot=600.0, side="bull_put", ex_div_date="2026-12-15")
        self.assertGreaterEqual(len(cands), 2)
        # Deep quotes beat 1-lot THIN_DEPTH quotes in ranking
        self.assertNotIn("THIN_DEPTH", cands[0]["flags"])
        self.assertTrue(any(f.startswith("EX_DIV:") for f in cands[0]["flags"]))
        self.assertTrue(any("THIN_DEPTH" in c["flags"] for c in cands))

    def test_h4_execution_window_guidance(self) -> None:
        from src.radar.live_options import OPRA_O1_LIQUIDITY_TIER, options_execution_window_guidance

        g_open = options_execution_window_guidance(dt.datetime(2026, 11, 25, 9, 45))
        g_mid = options_execution_window_guidance(dt.datetime(2026, 11, 25, 12, 0))
        g_fed = options_execution_window_guidance(dt.datetime(2026, 11, 25, 14, 0))
        g_close = options_execution_window_guidance(dt.datetime(2026, 11, 25, 15, 45))
        self.assertEqual(g_open["flag"], "EXEC_OPEN_WIDE_WAIT_1130")
        self.assertEqual(g_mid["window"], "OPTIMAL_OPRA_WINDOW")
        self.assertIsNone(g_mid["flag"])
        self.assertEqual(g_fed["flag"], "EXEC_1400_CHECK_FOMC")
        self.assertEqual(g_close["window"], "OPTIMAL_OPRA_WINDOW")
        self.assertEqual(OPRA_O1_LIQUIDITY_TIER["GLD"]["tier"], "TIER_1")
        self.assertEqual(OPRA_O1_LIQUIDITY_TIER["XLE"]["tier"], "TIER_2_WIDE")


class EdgarAndDiversifyTests(unittest.TestCase):
    def test_efts_cik_fallback_and_xbrl_bulk_zip(self) -> None:
        payload = {
            "hits": {
                "hits": [
                    {
                        "_id": "a1",
                        "_source": {
                            "adsh": "0001-26-000001",
                            "file_date": "2026-11-20",
                            "form": "SC 13D",
                            "display_names": ["APPLE INC (CIK 0000320193)"],
                            "ciks": ["0000320193"],
                        },
                    }
                ]
            }
        }
        evs = ee.parse_efts_hits(payload, event_type="ACTIVIST_13D", source_label="SEC", cik_to_ticker={320193: "AAPL"})
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].ticker, "AAPL")

        facts = {
            "entityName": "Apple Inc.",
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"end": "2025-09-27", "filed": "2025-11-01", "val": 1000.0},
                                {"end": "2026-09-26", "filed": "2026-11-01", "val": 950.0},
                            ]
                        }
                    }
                },
                "us-gaap": {
                    "GrossProfit": {"units": {"USD": [{"end": "2026-09-26", "form": "10-K", "val": 180.0}]}},
                    "Assets": {"units": {"USD": [{"end": "2026-09-26", "form": "10-K", "val": 360.0}]}},
                    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [{"end": "2026-09-26", "form": "10-K", "val": 120.0}]}},
                    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [{"end": "2026-09-26", "form": "10-K", "val": 12.0}]}},
                },
            },
        }
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "companyfacts.zip")
            db = os.path.join(td, "fund.db")
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("CIK0000320193.json", json.dumps(facts))
            with patch("src.alpha.edgar_events._reverse_cik_map", return_value={320193: "AAPL"}):
                res = ee.refresh_edgar_fundamentals(zip_path=zpath, db_path=db)
            self.assertEqual(res["saved"], 1)
            leaders = ee.load_quality_buyback_leaders(db_path=db)
            self.assertEqual(len(leaders), 1)
            self.assertAlmostEqual(leaders[0]["net_buyback_pct"], 5.0)
            self.assertAlmostEqual(leaders[0]["gross_profit_assets_pct"], 50.0)
            self.assertAlmostEqual(leaders[0]["fcf_assets_pct"], 30.0)
            self.assertEqual(leaders[0]["filed_date"], "2026-11-01")

    def test_portfolio_diversification_ranking_and_edgar_quality_lens_registered(self) -> None:
        self.assertIn("edgar_quality", LENS_BY_ID)
        rng = np.random.default_rng(7)
        days = business_days(180)
        f = rng.normal(0, 0.01, 180)
        g = rng.normal(0, 0.01, 180)
        vol = np.full(180, 1e6)
        ohlcv = {
            "XLE": (days, 100 * np.exp(np.cumsum(f)), vol),
            "XOM": (days, 100 * np.exp(np.cumsum(f + rng.normal(0, 0.002, 180))), vol),
            "GLD": (days, 100 * np.exp(np.cumsum(g)), vol),
        }
        rows = [
            LensRow(ticker="XOM", sector_etf="XLE", why="leader"),
            LensRow(ticker="GLD", sector_etf=None, why="gold"),
        ]
        ranked = rank_rows_for_diversification(
            rows, holdings=[Holding("XLE", "core sleeve (target)", weight=0.3)], ohlcv=ohlcv
        )
        self.assertEqual(ranked[0].ticker, "GLD")
        self.assertIn("LOW_CORR_DIV", ranked[0].flags)
        self.assertTrue(any(fl.startswith("CORR_DUP:XLE") for fl in ranked[1].flags))

    def test_xbrl_same_filing_diluted_nby_is_split_safe_and_point_in_time(self) -> None:
        # Simulate a 4-for-1 stock split between Q3-2020 (pre-split) and FY2020 10-K (post-split restated)
        facts = {
            "entityName": "SplitCo",
            "facts": {
                "us-gaap": {
                    "WeightedAverageNumberOfDilutedSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"accn": "A1", "form": "10-Q", "filed": "2020-07-31", "start": "2019-04-01", "end": "2019-06-30", "val": 1000.0},
                                {"accn": "A1", "form": "10-Q", "filed": "2020-07-31", "start": "2020-04-01", "end": "2020-06-30", "val": 950.0},
                                {"accn": "A2", "form": "10-K", "filed": "2020-10-30", "start": "2018-10-01", "end": "2019-09-30", "val": 4000.0},
                                {"accn": "A2", "form": "10-K", "filed": "2020-10-30", "start": "2019-10-01", "end": "2020-09-30", "val": 3800.0},
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {"accn": "A1", "form": "10-Q", "filed": "2020-07-31", "end": "2020-06-30", "val": 500.0},
                                {"accn": "A2", "form": "10-K", "filed": "2020-10-30", "end": "2020-09-30", "val": 600.0},
                            ]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [
                                {"accn": "A2", "form": "10-K", "filed": "2020-10-30", "start": "2019-10-01", "end": "2020-09-30", "val": 240.0},
                            ]
                        }
                    },
                }
            },
        }
        pit = ee.extract_xbrl_pit_series("SPLT", facts)
        # Before 10-K is filed on 2020-10-30, ONLY A1 is visible (zero look-ahead!)
        pre = ee.xbrl_metrics_as_of(pit, "2020-09-30")
        self.assertIsNotNone(pre)
        assert pre is not None
        self.assertAlmostEqual(pre["net_buyback_pct"], 5.0, places=2)
        self.assertIsNone(pre["gross_profit_assets_pct"])
        # After 10-K is filed on 2020-10-30, split-restated A2 still shows +5.0% buyback (not -300%!)
        post = ee.xbrl_metrics_as_of(pit, "2020-12-31")
        self.assertIsNotNone(post)
        assert post is not None
        self.assertAlmostEqual(post["net_buyback_pct"], 5.0, places=2)
        self.assertAlmostEqual(post["gross_profit_assets_pct"], 40.0, places=2)
        # Live metric helper also uses split-safe diluted NBY
        live = ee.compute_xbrl_quality_metrics("SPLT", facts)
        self.assertIsNotNone(live)
        assert live is not None
        self.assertAlmostEqual(live["net_buyback_pct"], 5.0, places=2)


if __name__ == "__main__":
    unittest.main()
