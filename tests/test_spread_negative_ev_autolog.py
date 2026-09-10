"""Tests for the spread/condor auto-log path's negative_ev gate.

Bull Put/Bear Call/Iron Condor auto-logging never called `pick_ranking`'s
gate at all (see options_screener.py's "DELIBERATELY NOT GATED" comment) —
a deliberate 2026-08-10 decision so the condor_universe/top_quintile research
rules keep an ungated sample to be tested against. This adds ONLY the
negative_ev/noise consistency check on top of that — never condor_universe,
never top_quintile — staged off/report/refuse the same way earnings_gate's
projection mode was, because a live measurement on 2026-09-08 found it would
refuse 89.5% of what currently gets logged.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_spread_negative_ev_autolog -v
"""
import json
import os
import tempfile
import unittest

import pandas as pd

from src.options_screener import (apply_spread_negative_ev_gate,
                                  spread_negative_ev_mode)


def _row(symbol, ev, strategy="Bull Put"):
    return {"symbol": symbol, "strategy_name": strategy,
            "ev_per_contract": ev, "ev_gross_per_contract": ev + 5.0,
            "ev_cost_per_contract": 5.0}


class ApplySpreadNegativeEvGateTest(unittest.TestCase):
    def test_off_mode_changes_nothing(self):
        df = pd.DataFrame([_row("A", -18.0), _row("B", 25.0)])
        kept, refused, would = apply_spread_negative_ev_gate(df, "off")
        self.assertEqual(len(kept), 2)
        self.assertEqual(refused, [])
        self.assertEqual(would, 0)

    def test_report_mode_counts_but_never_drops(self):
        df = pd.DataFrame([_row("A", -18.0), _row("B", 25.0)])
        kept, refused, would = apply_spread_negative_ev_gate(df, "report")
        self.assertEqual(len(kept), 2, "report mode must never shrink the frame")
        self.assertEqual(refused, [],
                        "nothing was actually refused, so nothing should be "
                        "recorded as refused")
        self.assertEqual(would, 1)

    def test_refuse_mode_actually_drops_the_negative_row(self):
        df = pd.DataFrame([_row("A", -18.0), _row("B", 25.0)])
        kept, refused, would = apply_spread_negative_ev_gate(df, "refuse")
        self.assertEqual(list(kept["symbol"]), ["B"])
        self.assertEqual(len(refused), 1)
        self.assertEqual(refused[0]["symbol"], "A")
        self.assertEqual(would, 1)

    def test_a_condor_row_is_covered_too(self):
        df = pd.DataFrame([_row("SPY", -10.0, strategy="Iron Condor")])
        kept, refused, would = apply_spread_negative_ev_gate(df, "refuse")
        self.assertEqual(len(kept), 0)
        self.assertEqual(would, 1)

    def test_missing_ev_fields_are_not_refused(self):
        df = pd.DataFrame([{"symbol": "X", "strategy_name": "Bull Put"}])
        kept, refused, would = apply_spread_negative_ev_gate(df, "refuse")
        self.assertEqual(len(kept), 1)
        self.assertEqual(would, 0)

    def test_empty_frame_is_handled(self):
        kept, refused, would = apply_spread_negative_ev_gate(pd.DataFrame(), "refuse")
        self.assertEqual(len(kept), 0)
        self.assertEqual(refused, [])
        self.assertEqual(would, 0)

    def test_none_is_handled(self):
        kept, refused, would = apply_spread_negative_ev_gate(None, "refuse")
        self.assertIsNone(kept)
        self.assertEqual(would, 0)


class SpreadNegativeEvModeConfigTest(unittest.TestCase):
    def _write_cfg(self, auto_log_block):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"auto_log": auto_log_block}, f)
        return path

    def test_missing_key_defaults_to_off(self):
        path = self._write_cfg({})
        try:
            self.assertEqual(spread_negative_ev_mode(path), "off")
        finally:
            os.remove(path)

    def test_report_mode_loads(self):
        path = self._write_cfg({"spread_negative_ev_mode": "report"})
        try:
            self.assertEqual(spread_negative_ev_mode(path), "report")
        finally:
            os.remove(path)

    def test_unrecognised_value_falls_back_to_off_never_refuse(self):
        path = self._write_cfg({"spread_negative_ev_mode": "yolo"})
        try:
            self.assertEqual(spread_negative_ev_mode(path), "off")
        finally:
            os.remove(path)

    def test_missing_file_defaults_to_off(self):
        self.assertEqual(spread_negative_ev_mode("/nonexistent/config.json"), "off")

    def test_real_config_is_in_report_mode(self):
        # Shipped 2026-09-08: watch it against real scans before enforcing.
        self.assertEqual(spread_negative_ev_mode("config.json"), "report")


if __name__ == "__main__":
    unittest.main()
