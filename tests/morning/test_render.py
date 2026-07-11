import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src.morning import render as R


def _fixture():
    return {
        "meta": {"schema": 1, "date": "2026-07-10", "generated_at": "2026-07-10 08:05 EDT",
                 "session": "pre-market", "sidecar": "2026-07-10.json",
                 "title": "Morning Briefing — 2026-07-10"},
        "panels": {
            "health": {"worst": "OK", "jobs": [
                {"name": "auto-log", "cadence": "business-daily",
                 "last_run": "2026-07-09", "stale_days": 1, "severity": "OK"}]},
            "market": {"regime": {"vix": 14.2, "vix_term_structure": "CONTANGO",
                                  "posture": "RISK_ON", "options_pcr": 0.92,
                                  "posture_rationale": "calm tape"},
                       "indexes": [{"sym": "SPY", "last": 620.0, "chg_1d_pct": 0.4,
                                    "chg_5d_pct": 1.2, "closes": [610, 612, 615, 618, 620]}],
                       "rates": {"t10y": 4.2, "t3m": 4.4, "slope": -0.2}},
            "vol": {"movers": [{"symbol": "NVDA", "iv": 0.45, "d_iv": 0.03}],
                    "vrp": [{"symbol": "SPY", "iv": 0.15, "rv": 0.12,
                             "vrp": 3.1, "label": "RICH"}],
                    "n_cov": 12, "crypto_note": "BTC carry note"},
            "macro_events": {"calendar": [{"date": "2026-07-15", "name": "CPI"}],
                             "pulse": "risk tone neutral", "headlines": ["A", "B"],
                             "earnings": [{"sym": "NVDA", "date": "2026-07-14"}]},
            "signals": {"uoa": [{"symbol": "TSLA", "score": 2.4,
                                 "net_call_share": 0.81, "n_unusual": 3}],
                        "insider": [{"sym": "AAPL", "summary": "CLUSTER BUY (score 0.9)"}],
                        "outlook": {"top": [{"ticker": "XLK", "direction": "LONG"}],
                                    "bottom": [{"ticker": "XLE", "direction": "SHORT"}],
                                    "as_of": "2026-07-10 08:00 UTC"}},
            "portfolio": {"positions": [{"ticker": "NVDA", "strategy": "Long Call",
                                         "dte": 18.0, "pnl_pct": 12.5, "delta": 0.42}],
                          "net_greeks": {"portfolio_delta": 0.42, "portfolio_vega": 10.0},
                          "guard": ["one concentrated bet"],
                          "exits_due": ["NVDA: 18 DTE <= 21 — time-exit window"],
                          "n_open": 1},
            "gate": {"pooled_ic": 0.10, "p_value": 0.48, "n_oos": 30, "cohort_n": 2,
                     "gate_decision": "GATHERING", "as_of": "2026-07-01", "target_n": 50},
            "notes": ["Real money is OFF until the forward-cohort gate (n>=50) fires."],
        },
        "failures": [],
    }


class TestRender(unittest.TestCase):
    def test_deterministic_and_selfcontained(self):
        d = _fixture()
        html1, html2 = R.render(d), R.render(d)
        self.assertEqual(html1, html2)
        self.assertTrue(html1.startswith("<!DOCTYPE html>"))
        for banned in ("http://", "https://", 'src="//'):
            self.assertNotIn(banned, html1)

    def test_key_values_present(self):
        html = R.render(_fixture())
        for needle in ("Morning Briefing", "2026-07-10", "pre-market", "14.2",
                       "CONTANGO", "CPI", "GATHERING", "2/50", "Real money is OFF",
                       "NVDA", "CLUSTER BUY", "time-exit window"):
            self.assertIn(needle, html)

    def test_missing_panel_renders_placeholder(self):
        d = _fixture()
        d["panels"]["vol"] = None
        html = R.render(d)
        self.assertIn("unavailable", html.lower())
        self.assertIn("Vol Intelligence", html)

    def test_no_ansi_escapes(self):
        self.assertNotIn("\x1b[", R.render(_fixture()))


if __name__ == "__main__":
    unittest.main()
