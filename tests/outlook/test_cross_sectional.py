"""outlook's factors, transferred point-in-time onto a wider (single-name)
universe than the 16 sector/asset ETFs they were validated on.

The hazard here is not the factor math (already tested in
tests/outlook/test_factors.py) — it is DATE ALIGNMENT. Two series with
different missing-date patterns must never be compared at a shared
positional index, or `relative_strength`'s benchmark lookback silently reads
the wrong calendar date.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.outlook.test_cross_sectional -v
"""
from __future__ import annotations

import unittest

from src.outlook.cross_sectional import _factor_row, _index_asof, composite_lookup


def _dates(n, start=0):
    """n consecutive business-ish day labels, sortable as strings."""
    return [f"2024-{1 + (start + i) // 28:02d}-{1 + (start + i) % 28:02d}"
           for i in range(n)]


def _flat(dates, price=100.0):
    return {d: price for d in dates}


def _trending(dates, start=100.0, daily=0.002):
    out = {}
    p = start
    for d in dates:
        out[d] = p
        p *= (1 + daily)
    return out


class IndexAsofTest(unittest.TestCase):

    def test_exact_match_returns_its_own_position(self):
        dates = _dates(10)
        self.assertEqual(_index_asof(dates, dates[5]), 5)

    def test_a_date_between_two_trading_days_uses_the_earlier_one(self):
        dates = ["2024-01-02", "2024-01-04"]  # a gap, e.g. a holiday
        self.assertEqual(_index_asof(dates, "2024-01-03"), 0)

    def test_before_every_date_on_file_is_none(self):
        dates = _dates(10, start=100)
        self.assertIsNone(_index_asof(dates, "2020-01-01"))

    def test_never_returns_a_future_index(self):
        dates = _dates(300)
        idx = _index_asof(dates, dates[100])
        self.assertLessEqual(idx, 100)


class FactorRowTest(unittest.TestCase):

    def test_none_when_the_date_predates_all_history(self):
        dates = _dates(400)
        row = _factor_row(dates, [100.0] * 400, dates, [100.0] * 400,
                          "1999-01-01")
        self.assertIsNone(row)

    def test_relative_strength_uses_each_series_own_calendar_position(self):
        # The symbol is MISSING day 200 (a data gap yfinance sometimes has)
        # while the benchmark has every day. A shared positional index would
        # read the benchmark 1 day off from this point forward; looking each
        # series up BY DATE must not.
        sdates = _dates(400)
        bdates = _dates(400)
        del sdates[200]
        scloses = _trending(sdates)
        bcloses = _flat(bdates)
        target = sdates[-1]
        row = _factor_row(sdates, [scloses[d] for d in sdates],
                          bdates, [bcloses[d] for d in bdates], target)
        self.assertIsNotNone(row["relative_strength"])
        # Flat benchmark: relative strength must equal the instrument's own
        # 63-day return exactly, not something drifted by an index-shift bug.
        st = _index_asof(sdates, target)
        inst_ret = scloses[sdates[st]] / scloses[sdates[st - 63]] - 1.0
        self.assertAlmostEqual(row["relative_strength"], inst_ret, places=9)

    def test_short_history_reports_none_not_zero(self):
        dates = _dates(30)
        row = _factor_row(dates, [100.0] * 30, dates, [100.0] * 30, dates[-1])
        self.assertIsNone(row["mom_12_1"])
        self.assertIsNone(row["trend_score"])


class CompositeLookupTest(unittest.TestCase):

    def _closes(self, n=400):
        dates = _dates(n)
        return dates, {
            "SPY": _flat(dates, 400.0),
            "UP": _trending(dates, start=100.0, daily=0.003),
            "DOWN": _trending(dates, start=100.0, daily=-0.003),
            "FLAT": _flat(dates, 50.0),
        }

    def test_a_date_with_a_thin_cross_section_is_skipped(self):
        dates, closes = self._closes()
        out = composite_lookup({"SPY": closes["SPY"]}, [dates[-1]])
        self.assertEqual(out, {})

    def test_a_consistently_outperforming_symbol_scores_above_a_laggard(self):
        dates, closes = self._closes()
        out = composite_lookup(closes, [dates[-1]])
        self.assertIn(("UP", dates[-1]), out)
        self.assertIn(("DOWN", dates[-1]), out)
        self.assertGreater(out[("UP", dates[-1])], out[("DOWN", dates[-1])])

    def test_missing_from_the_universe_is_absent_not_zero(self):
        dates, closes = self._closes()
        out = composite_lookup(closes, [dates[-1]])
        self.assertNotIn(("GHOST", dates[-1]), out)

    def test_no_benchmark_in_the_data_yields_nothing(self):
        dates, closes = self._closes()
        del closes["SPY"]
        out = composite_lookup(closes, [dates[-1]])
        self.assertEqual(out, {})

    def test_too_little_history_for_every_date_yields_nothing(self):
        # Below TRADING_MONTH (21) — not even reversal_1m, the shortest
        # lookback of the four, can compute anything here.
        dates, closes = self._closes(n=15)
        out = composite_lookup(closes, dates)
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
