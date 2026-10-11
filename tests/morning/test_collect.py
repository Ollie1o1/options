import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
from datetime import datetime

from src.morning import collect


class TestSessionPhase(unittest.TestCase):
    def test_premarket_open_closed_weekend(self):
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 8, 0)), "pre-market")
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 10, 30)), "open")
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 17, 0)), "closed")
        self.assertEqual(collect.session_phase(datetime(2026, 7, 11, 10, 30)), "closed")  # Saturday
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 3, 0)), "closed")


class TestSafeHarness(unittest.TestCase):
    def test_failure_is_captured_not_raised(self):
        panels, failures = {}, []
        def boom():
            raise RuntimeError("dead fetch")
        collect._safe("market", boom, panels, failures)
        self.assertIsNone(panels["market"])
        self.assertEqual(len(failures), 1)
        self.assertIn("market: RuntimeError: dead fetch", failures[0])

    def test_success_stores_value(self):
        panels, failures = {}, []
        collect._safe("gate", lambda: {"ok": 1}, panels, failures)
        self.assertEqual(panels["gate"], {"ok": 1})
        self.assertEqual(failures, [])


class TestBuildSkeleton(unittest.TestCase):
    def test_build_meta_and_all_panels_present(self):
        # slow=False and stubbed fetchers: build must never hit the network in tests
        data = collect.build(now=datetime(2026, 7, 10, 9, 0), slow=False,
                             _fetchers=[("market", lambda: {"x": 1})])
        self.assertEqual(data["meta"]["schema"], 1)
        self.assertEqual(data["meta"]["date"], "2026-07-10")
        self.assertEqual(data["meta"]["session"], "pre-market")
        self.assertEqual(data["meta"]["sidecar"], "2026-07-10.json")
        for pid in collect.PANEL_IDS:
            self.assertIn(pid, data["panels"])
        self.assertEqual(data["panels"]["market"], {"x": 1})
        self.assertIsInstance(data["panels"]["notes"], list)

    def test_build_notes_name_failed_panels(self):
        def boom():
            raise ValueError("no db")
        data = collect.build(now=datetime(2026, 7, 10, 9, 0), slow=False,
                             _fetchers=[("vol", boom)])
        self.assertIsNone(data["panels"]["vol"])
        self.assertTrue(any("vol" in n for n in data["panels"]["notes"]))


class TestMarketPanel(unittest.TestCase):
    def test_panel_market_shapes_injected(self):
        regime = {"vix": 14.2, "posture": "RISK_ON", "vix_term_structure": "CONTANGO"}
        dirs = {"SPY": {"last": 620.0, "chg_1d_pct": 0.4, "chg_5d_pct": 1.2,
                        "closes": [610.0, 612.0, 615.0, 618.0, 620.0],
                        "verdict": "BULLISH"}}
        out = collect._panel_market(_regime_fn=lambda: regime,
                                    _dirs_fn=lambda: dirs,
                                    _rates_fn=lambda: {"t10y": 4.2, "t3m": 4.4})
        self.assertEqual(out["regime"]["vix"], 14.2)
        self.assertEqual(out["indexes"][0]["sym"], "SPY")
        self.assertAlmostEqual(out["rates"]["slope"], -0.2, places=6)

    def test_panel_market_rates_failure_is_partial_not_fatal(self):
        def boom():
            raise RuntimeError("fred down")
        out = collect._panel_market(_regime_fn=lambda: {"vix": 15.0},
                                    _dirs_fn=lambda: {}, _rates_fn=boom)
        self.assertEqual(out["rates"], {"t10y": None, "t3m": None, "slope": None})
        self.assertEqual(out["regime"]["vix"], 15.0)

    def test_index_rows_from_closes(self):
        closes = [100.0 + i for i in range(30)]
        rows = collect._index_rows_from_closes({"SPY": closes})
        self.assertEqual(rows["SPY"]["last"], 129.0)
        self.assertAlmostEqual(rows["SPY"]["chg_1d_pct"],
                               (129.0 / 128.0 - 1) * 100, places=6)
        self.assertAlmostEqual(rows["SPY"]["chg_5d_pct"],
                               (129.0 / 124.0 - 1) * 100, places=6)
        self.assertEqual(len(rows["SPY"]["closes"]), 30)


class TestHealthGatePanels(unittest.TestCase):
    def test_panel_health_serializes_report(self):
        class J:
            name, cadence, last_run = "auto-log", "business-daily", "2026-07-09"
            business_days_stale, severity = 1, "OK"
        class R:
            jobs, worst = [J()], "OK"
        out = collect._panel_health(_report_fn=lambda: R())
        self.assertEqual(out["worst"], "OK")
        self.assertEqual(out["jobs"][0]["name"], "auto-log")
        self.assertEqual(out["jobs"][0]["stale_days"], 1)

    def test_panel_gate_carries_target(self):
        ev = {"pooled_ic": 0.1, "p_value": 0.4, "n_oos": 30,
              "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-07-01"}
        out = collect._panel_gate(_evidence_fn=lambda: dict(ev))
        self.assertEqual(out["cohort_n"], 2)
        self.assertGreaterEqual(out["target_n"], 50)


class TestVolPanel(unittest.TestCase):
    def test_panel_vol_caps_sorts_and_passthrough(self):
        movers = [{"symbol": f"S{i}", "iv": 0.3, "d_iv": 0.001 * i} for i in range(20)]
        vrp = [{"symbol": "SPY", "iv": 0.15, "rv": 0.12, "vrp": 3.1, "label": "RICH"}]
        out = collect._panel_vol(_rows_fn=lambda: (movers, vrp))
        self.assertEqual(len(out["movers"]), 8)
        self.assertEqual(out["movers"][0]["symbol"], "S19")  # biggest |d_iv| first
        self.assertEqual(out["vrp"][0]["symbol"], "SPY")
        self.assertEqual(out["n_cov"], 1)
        self.assertIn("carry", out["crypto_note"].lower())


class TestMacroEventsPanel(unittest.TestCase):
    def test_panel_macro_events_shapes(self):
        cal = [{"date": "2026-07-15", "name": "CPI"}]
        out = collect._panel_macro_events(
            _calendar_fn=lambda: cal,
            _pulse_fn=lambda: ("risk tone neutral", ["Headline A", "Headline B"]),
            _earnings_fn=lambda: [{"sym": "NVDA", "date": "2026-08-26"}])
        self.assertEqual(out["calendar"][0]["name"], "CPI")
        self.assertEqual(out["pulse"], "risk tone neutral")
        self.assertEqual(out["earnings"][0]["sym"], "NVDA")

    def test_panel_macro_events_pulse_failure_partial(self):
        def boom():
            raise RuntimeError("rss down")
        out = collect._panel_macro_events(_calendar_fn=lambda: [],
                                          _pulse_fn=boom,
                                          _earnings_fn=lambda: [])
        self.assertIsNone(out["pulse"])
        self.assertEqual(out["headlines"], [])


class TestSignalsPanel(unittest.TestCase):
    def test_panel_signals_shapes(self):
        rows = [{"ticker": "ETF" + str(i), "composite": 1.0 - 0.1 * i,
                 "direction": "LONG"} for i in range(8)]
        out = collect._panel_signals(
            _uoa_fn=lambda: {"date": "2026-07-10", "days_available": 3,
                             "rows": [{"symbol": "NVDA", "score": 2.1,
                                       "net_call_share": 0.8, "unusual": [1, 2]}]},
            _insider_fn=lambda: [{"sym": "AAPL", "summary": "CLUSTER BUY (score 0.9)"}],
            _outlook_fn=lambda: {"rows": rows, "as_of": "2026-07-10"})
        self.assertEqual(out["uoa"][0]["symbol"], "NVDA")
        self.assertEqual(out["uoa"][0]["n_unusual"], 2)
        self.assertEqual(len(out["outlook"]["top"]), 3)
        self.assertEqual(len(out["outlook"]["bottom"]), 3)
        self.assertEqual(out["outlook"]["top"][0]["ticker"], "ETF0")

    def test_panel_signals_all_dead_is_empty_not_fatal(self):
        def boom():
            raise RuntimeError("dead")
        out = collect._panel_signals(_uoa_fn=boom, _insider_fn=boom, _outlook_fn=boom)
        self.assertEqual(out["uoa"], [])
        self.assertEqual(out["insider"], [])
        self.assertEqual(out["outlook"], {"top": [], "bottom": [], "as_of": None})


class TestPortfolioPanel(unittest.TestCase):
    def test_panel_portfolio_serializes_and_flags_exits(self):
        rows = [{"ticker": "NVDA", "strategy": "Long Call", "dte": 18.0,
                 "pnl_pct": 12.5, "delta": 0.42}]
        out = collect._panel_portfolio(
            _positions_fn=lambda: rows,
            _greeks_fn=lambda: {"portfolio_delta": 0.42, "portfolio_vega": 10.0,
                                "n_positions": 1},
            _guard_fn=lambda: ["concentrated: 1 underlying"])
        self.assertEqual(out["n_open"], 1)
        self.assertEqual(out["positions"][0]["ticker"], "NVDA")
        self.assertEqual(out["net_greeks"]["portfolio_vega"], 10.0)
        self.assertEqual(len(out["exits_due"]), 1)  # dte 18 <= 21
        self.assertIn("NVDA", out["exits_due"][0])
        self.assertIn("concentrated: 1 underlying", out["guard"])

    def test_panel_portfolio_empty_book(self):
        out = collect._panel_portfolio(_positions_fn=lambda: [],
                                       _greeks_fn=lambda: {},
                                       _guard_fn=lambda: [])
        self.assertEqual(out["n_open"], 0)
        self.assertEqual(out["exits_due"], [])


if __name__ == "__main__":
    unittest.main()


def _bars(closes, vols=None):
    vols = vols or [1_000_000] * len(closes)
    return list(zip(closes, vols))


class TestMoversPanel(unittest.TestCase):
    def test_shock_needs_both_the_move_and_the_volume(self):
        flat = [100.0] * 24
        # +20% on 6x volume: a shock
        shock = _bars(flat + [120.0], [1_000_000] * 24 + [6_000_000])
        # +20% on ordinary volume: a new 20-day high, but NOT a shock
        quiet = _bars(flat + [120.0])
        rows = {r["sym"]: r for r in collect._mover_rows({"AAA": shock, "BBB": quiet}, {})}
        self.assertEqual(rows["AAA"]["flag"], "shock")
        self.assertEqual(rows["AAA"]["shock"]["days_ago"], 0)
        self.assertAlmostEqual(rows["AAA"]["shock"]["move_pct"], 20.0)
        self.assertAlmostEqual(rows["AAA"]["shock"]["vol_mult"], 6.0)
        self.assertEqual(rows["BBB"]["flag"], "20d high")
        self.assertIsNone(rows["BBB"]["shock"])

    def test_shock_stays_listed_for_five_sessions_then_drops(self):
        base, v = [100.0] * 24, [1_000_000] * 24
        recent = _bars(base + [80.0] + [80.0] * 4, v + [6_000_000] + [1_000_000] * 4)
        stale = _bars(base + [80.0] + [80.0] * 5, v + [6_000_000] + [1_000_000] * 5)
        rows = {r["sym"]: r for r in collect._mover_rows({"NEW": recent, "OLD": stale}, {})}
        self.assertEqual(rows["NEW"]["shock"]["days_ago"], 4)
        self.assertAlmostEqual(rows["NEW"]["shock"]["move_pct"], -20.0)
        self.assertNotIn("OLD", rows)  # down 20% and flat since: neither shock nor high

    def test_quiet_names_and_short_histories_are_left_out(self):
        rows = collect._mover_rows({"FLAT": _bars([100.0] * 30),
                                    "THIN": _bars([100.0, 130.0])}, {})
        self.assertEqual(rows, [])

    def test_shocks_sort_ahead_of_highs_and_rows_are_capped(self):
        bars = {f"H{i:02d}": _bars([100.0] * 24 + [101.0 + i]) for i in range(15)}
        bars["SHK"] = _bars([100.0] * 24 + [118.0], [1_000_000] * 24 + [9_000_000])
        rows = collect._mover_rows(bars, {})
        self.assertEqual(rows[0]["sym"], "SHK")
        self.assertEqual(len(rows), collect._MOVERS_MAX_ROWS)
        self.assertEqual(rows[1]["sym"], "H14")  # biggest 20-day gain among the highs

    def test_passing_candidates_ride_along(self):
        bars = {"AAA": _bars([100.0] * 24 + [120.0], [1_000_000] * 24 + [6_000_000])}
        rows = collect._mover_rows(bars, {"AAA": [{"strategy": "Long Call", "n": 2}]})
        self.assertEqual(rows[0]["passing"], [{"strategy": "Long Call", "n": 2}])

    def test_panel_shapes_injected_and_carries_the_base_rate(self):
        scan = {"scan_date": "2026-10-09", "symbols": ["AAA", "ZZZ"],
                "passing": {"AAA": [{"strategy": "Bull Put", "n": 1}]}}
        seen = {}

        def bars_fn(symbols):
            seen["symbols"] = list(symbols)
            return {"AAA": _bars([100.0] * 24 + [120.0], [1_000_000] * 24 + [6_000_000])}

        p = collect._panel_movers(_scan_fn=lambda: scan, _bars_fn=bars_fn)
        self.assertEqual(seen["symbols"], ["AAA", "ZZZ"])
        self.assertEqual(p["scan_date"], "2026-10-09")
        self.assertEqual(p["n_priced"], 1)
        self.assertEqual(p["n_symbols"], 2)
        self.assertEqual(p["rows"][0]["sym"], "AAA")
        self.assertIn("558", p["base_rate"])

    def test_recent_scan_reads_passers_from_a_candidate_db(self):
        import sqlite3
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidates.db")
            conn = sqlite3.connect(path)
            conn.execute("create table candidates (ts text, symbol text, strategy_name text,"
                         " contract_key text, gate_passed integer)")
            conn.executemany("insert into candidates values (?,?,?,?,?)", [
                ("2026-10-09T15:00:00", "AAA", "Long Call", "k1", 1),
                ("2026-10-09T15:00:00", "AAA", "Long Call", "k1", 1),   # same contract, second board
                ("2026-10-09T15:00:00", "AAA", "Long Call", "k2", 1),
                ("2026-10-09T15:00:00", "AAA", "Bull Put", "k3", 0),    # refused
                ("2026-10-09T15:00:00", "AAA", "", "k4", 1),            # unnamed single-leg board
                ("2026-10-08T15:00:00", "BBB", "Bull Put", "k5", 1),    # passed, but not latest day
                ("2026-09-01T15:00:00", "OLD", "Bull Put", "k6", 1),    # outside the window
            ])
            conn.commit()
            conn.close()
            scan = collect._fetch_recent_scan(db_path=path)
        self.assertEqual(scan["scan_date"], "2026-10-09")
        self.assertEqual(scan["symbols"], ["AAA", "BBB"])
        self.assertEqual(scan["passing"], {"AAA": [{"strategy": "Long Call", "n": 2}]})
