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

    def test_kpi_strip_present(self):
        html = R.render(_fixture())
        self.assertIn("class='kpis'", html)
        for label in ("VIX", "SPY 1D", "POSTURE", "10Y–3M", "GATE"):
            self.assertIn(label, html)

    def test_callout_takeaways_bolded(self):
        html = R.render(_fixture())
        self.assertIn("What matters this morning", html)
        self.assertIn("<b>Exit due:</b>", html)
        self.assertIn("<b>Biggest IV move:</b>", html)
        self.assertIn("<b>Widest VRP:</b>", html)

    def test_portfolio_collapsed_in_details(self):
        html = R.render(_fixture())
        start = html.index("<h2>Portfolio</h2>")
        chunk = html[start:start + 400]
        self.assertIn("<details>", chunk)
        self.assertIn("1 open position", chunk)

    def test_vol_charts_are_svg(self):
        html = R.render(_fixture())
        self.assertIn("class='bars'", html)   # signed ΔIV bars + IV/RV pairs


class TestGateZoneRefusal(unittest.TestCase):
    """Important-2: a walk-forward refusal (src/walk_forward.py — too few
    folds survived purging) must be named, not rendered as 'not computed
    yet' with a trade count riding along that lends it false weight."""

    def test_refused_gate_names_the_refusal(self):
        p = {"cohort_n": 2, "target_n": 50, "gate_decision": "GATHERING",
             "pooled_ic": None, "p_value": None, "n_oos": 0, "as_of": None,
             "wf_refused": True,
             "wf_refused_reason": ("only 0 of 15 folds kept 54+ training "
                                   "trades after purging (minimum 3)")}
        html = R._zone_gate(p)
        self.assertIn("REFUSED", html)
        self.assertIn("only 0 of 15 folds", html)

    def test_refused_gate_does_not_render_a_misleading_n_oos(self):
        # Before the fix this rendered "Walk-forward pooled IC — (p=—,
        # n_oos=108) — no demonstrated edge yet" — reading as "not computed
        # yet, on 108 trades" rather than "computed and refused", with the
        # trade count lending it false weight.
        p = {"cohort_n": 2, "target_n": 50, "gate_decision": "GATHERING",
             "pooled_ic": None, "p_value": None, "n_oos": 108, "as_of": None,
             "wf_refused": True, "wf_refused_reason": "refused for testing"}
        html = R._zone_gate(p)
        self.assertNotIn("n_oos=108", html)
        self.assertNotIn("no demonstrated edge yet", html)

    def test_normal_gate_is_unaffected(self):
        p = {"cohort_n": 2, "target_n": 50, "gate_decision": "GATHERING",
             "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 30,
             "as_of": "2026-07-01", "wf_refused": False}
        html = R._zone_gate(p)
        self.assertNotIn("REFUSED", html)
        self.assertIn("n_oos=30", html)


if __name__ == "__main__":
    unittest.main()
