"""Tests for src/uoa.py — unusual options activity from the chain archive."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import uoa
from src.chain_archive import ensure_db, _COLUMNS


def _insert(conn, symbol, snap_date, contract, type_, strike, oi, volume,
            bid=1.0, ask=1.2, spot=100.0, expiration="2026-07-17"):
    row = {
        "symbol": symbol, "snap_date": snap_date, "contract": contract,
        "type": type_, "strike": strike, "expiration": expiration,
        "bid": bid, "ask": ask, "bid_size": 1, "ask_size": 1,
        "iv": 0.3, "delta": 0.3 if type_ == "call" else -0.3,
        "gamma": 0.01, "theta": -0.02, "vega": 0.1, "rho": 0.01,
        "open_interest": oi, "volume": volume, "last_trade_time": None,
        "spot": spot, "snapshot_ts": snap_date, "source": "cboe",
    }
    conn.execute(
        f"INSERT INTO chain_snapshots ({','.join(_COLUMNS)}) "
        f"VALUES ({','.join('?' * len(_COLUMNS))})",
        [row.get(c) for c in _COLUMNS])


class UoaTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "chains.db")
        ensure_db(self.db)
        self.conn = sqlite3.connect(self.db)

    def tearDown(self):
        self.conn.close()


class OiDeltasTest(UoaTestBase):
    def test_computes_day_over_day_delta(self):
        _insert(self.conn, "SPY", "2026-06-10", "SPYC110", "call", 110, oi=1000, volume=50)
        _insert(self.conn, "SPY", "2026-06-11", "SPYC110", "call", 110, oi=1800, volume=900)
        self.conn.commit()
        deltas = uoa.oi_deltas(self.conn, "SPY", "2026-06-11")
        self.assertEqual(len(deltas), 1)
        d = deltas[0]
        self.assertEqual(d["d_oi"], 800)
        self.assertEqual(d["prev_oi"], 1000)
        self.assertEqual(d["volume"], 900)
        self.assertTrue(d["is_otm"])  # call strike 110 vs spot 100

    def test_new_contract_counts_full_oi(self):
        _insert(self.conn, "SPY", "2026-06-10", "SPYC120", "call", 120, oi=10, volume=1)
        _insert(self.conn, "SPY", "2026-06-11", "SPYC130", "call", 130, oi=600, volume=700)
        self.conn.commit()
        deltas = uoa.oi_deltas(self.conn, "SPY", "2026-06-11")
        d = {x["contract"]: x for x in deltas}
        self.assertEqual(d["SPYC130"]["d_oi"], 600)
        self.assertEqual(d["SPYC130"]["prev_oi"], 0)

    def test_no_previous_day_returns_empty(self):
        _insert(self.conn, "SPY", "2026-06-11", "SPYC110", "call", 110, oi=1000, volume=50)
        self.conn.commit()
        self.assertEqual(uoa.oi_deltas(self.conn, "SPY", "2026-06-11"), [])


class SymbolFlowTest(unittest.TestCase):
    def _d(self, type_="call", d_oi=600, prev_oi=1000, volume=100,
           mid=1.1, is_otm=True):
        return {"contract": "X", "type": type_, "strike": 110.0,
                "expiration": "2026-07-17", "d_oi": d_oi, "prev_oi": prev_oi,
                "volume": volume, "mid": mid, "is_otm": is_otm, "dte": 36}

    def test_net_call_share(self):
        flow = uoa.symbol_flow([self._d("call", d_oi=900, mid=2.0),
                                self._d("put", d_oi=100, mid=2.0)])
        self.assertGreater(flow["net_call_share"], 0.8)
        self.assertEqual(flow["call_oi_added"], 900)
        self.assertEqual(flow["put_oi_added"], 100)

    def test_unusual_flags_large_oi_jump(self):
        # +600 on prev 1000 = +60% and >=500 -> unusual
        flow = uoa.symbol_flow([self._d(d_oi=600, prev_oi=1000)])
        self.assertEqual(len(flow["unusual"]), 1)

    def test_small_changes_not_unusual(self):
        flow = uoa.symbol_flow([self._d(d_oi=50, prev_oi=1000, volume=10)])
        self.assertEqual(flow["unusual"], [])

    def test_volume_spike_vs_prior_oi_is_unusual(self):
        # volume 3x prior OI signals heavy new positioning even before OI updates
        flow = uoa.symbol_flow([self._d(d_oi=0, prev_oi=200, volume=900)])
        self.assertEqual(len(flow["unusual"]), 1)

    def test_oi_drops_do_not_count_as_added(self):
        flow = uoa.symbol_flow([self._d("call", d_oi=-500)])
        self.assertEqual(flow["call_oi_added"], 0)


class ReportTest(UoaTestBase):
    def test_graceful_with_one_day(self):
        _insert(self.conn, "SPY", "2026-06-11", "SPYC110", "call", 110, oi=1000, volume=50)
        self.conn.commit(); self.conn.close()
        r = uoa.uoa_report(self.db, date="2026-06-11", symbols=["SPY"])
        self.assertEqual(r["days_available"], 1)
        self.assertEqual(r["rows"], [])
        self.conn = sqlite3.connect(self.db)

    def test_ranks_symbols_by_unusualness(self):
        for day, oi in (("2026-06-10", 1000), ("2026-06-11", 3000)):
            _insert(self.conn, "AAA", day, "AAAC110", "call", 110, oi=oi, volume=500)
        for day, oi in (("2026-06-10", 1000), ("2026-06-11", 1010)):
            _insert(self.conn, "BBB", day, "BBBC110", "call", 110, oi=oi, volume=5)
        self.conn.commit(); self.conn.close()
        r = uoa.uoa_report(self.db, date="2026-06-11", symbols=["AAA", "BBB"])
        self.assertGreaterEqual(r["days_available"], 2)
        self.assertEqual(r["rows"][0]["symbol"], "AAA")
        self.assertEqual(len(r["rows"][0]["unusual"]), 1)
        self.conn = sqlite3.connect(self.db)


if __name__ == "__main__":
    unittest.main()
