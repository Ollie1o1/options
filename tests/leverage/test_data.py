import unittest
import os
import tempfile
import pandas as pd
from src.leverage import data as D


class TestDataCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_klines_to_frame_parses_binance_rows(self):
        raw = [[1780414200000, "67422.2", "67475.6", "67126.2", "67207.9",
                "5316.795", 1780414499999, "0", 0, "0", "0", "0"]]
        df = D._klines_to_frame(raw)
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(df.index.name, "open_time")
        self.assertAlmostEqual(df["close"].iloc[0], 67207.9)

    def test_cache_roundtrip(self):
        df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5],
                           "close": [1.5], "volume": [10.0]},
                          index=pd.DatetimeIndex([pd.Timestamp("2026-05-01", tz="UTC")],
                                                 name="open_time"))
        path = os.path.join(self.tmp, "BTCUSDT", "5m.parquet")
        D._write_cache(path, df)
        back = D._read_cache(path)
        pd.testing.assert_frame_equal(df, back)

    def test_fetch_falls_back_to_bybit_when_binance_raises(self):
        import requests
        calls = []

        def boom(*_a, **_k):
            calls.append("binance")
            raise requests.HTTPError("451 Unavailable For Legal Reasons")

        def bybit_ok(symbol, interval, start_ms, end_ms):
            calls.append("bybit")
            idx = pd.date_range("2026-05-01", periods=5,
                                freq=interval.replace("m", "min"), tz="UTC")
            return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                                 "close": 1.5, "volume": 10.0},
                                index=idx).rename_axis("open_time")
        orig_b, orig_y = D.fetch_klines_binance, D.fetch_klines_bybit
        D.fetch_klines_binance, D.fetch_klines_bybit = boom, bybit_ok
        try:
            df = D._fetch_with_fallback("BTCUSDT", "5m", 0, 1)
        finally:
            D.fetch_klines_binance, D.fetch_klines_bybit = orig_b, orig_y
        self.assertEqual(calls, ["binance", "bybit"])
        self.assertEqual(len(df), 5)

    def test_fetch_no_fallback_when_binance_succeeds(self):
        calls = []

        def binance_ok(symbol, interval, start_ms, end_ms):
            calls.append("binance")
            idx = pd.date_range("2026-05-01", periods=3,
                                freq=interval.replace("m", "min"), tz="UTC")
            return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                                 "close": 1.5, "volume": 10.0},
                                index=idx).rename_axis("open_time")

        def bybit_spy(*_a, **_k):
            calls.append("bybit")
            return pd.DataFrame()
        orig_b, orig_y = D.fetch_klines_binance, D.fetch_klines_bybit
        D.fetch_klines_binance, D.fetch_klines_bybit = binance_ok, bybit_spy
        try:
            df = D._fetch_with_fallback("BTCUSDT", "5m", 0, 1)
        finally:
            D.fetch_klines_binance, D.fetch_klines_bybit = orig_b, orig_y
        self.assertEqual(calls, ["binance"])  # bybit never touched
        self.assertEqual(len(df), 3)

    def test_load_history_default_path_recovers_via_bybit(self):
        import requests

        def boom(*_a, **_k):
            raise requests.ConnectionError("no route to host")

        def bybit_ok(symbol, interval, start_ms, end_ms):
            idx = pd.date_range("2026-05-01", periods=12,
                                freq=interval.replace("m", "min"), tz="UTC")
            return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                                 "close": 1.5, "volume": 10.0},
                                index=idx).rename_axis("open_time")
        orig_b, orig_y = D.fetch_klines_binance, D.fetch_klines_bybit
        D.fetch_klines_binance, D.fetch_klines_bybit = boom, bybit_ok
        try:
            df5, df15 = D.load_history("BTCUSDT", cache_dir=self.tmp, days=1)
        finally:
            D.fetch_klines_binance, D.fetch_klines_bybit = orig_b, orig_y
        self.assertGreater(len(df5), 0)  # recovered, did not raise

    def test_load_history_serves_cache_when_topup_fails(self):
        import requests
        # seed a cache, then make the top-up fetch fail outright
        idx = pd.date_range("2026-05-01", periods=30, freq="5min", tz="UTC")
        seed = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                             "volume": 10.0}, index=idx).rename_axis("open_time")
        D._write_cache(os.path.join(self.tmp, "BTCUSDT", "5m.parquet"), seed)

        def boom(*_a, **_k):
            raise requests.ConnectionError("offline")
        df5, df15 = D.load_history("BTCUSDT", cache_dir=self.tmp,
                                   fetcher=boom, days=1)
        self.assertEqual(len(df5), 30)  # served the cache, did not raise

    def test_load_history_cold_start_failure_propagates(self):
        import requests

        def boom(*_a, **_k):
            raise requests.ConnectionError("offline")
        with self.assertRaises(requests.ConnectionError):
            D.load_history("ETHUSDT", cache_dir=self.tmp, fetcher=boom, days=1)

    def test_load_history_uses_injected_fetcher(self):
        def fake_fetch(symbol, interval, start_ms, end_ms):
            idx = pd.date_range("2026-05-01", periods=10,
                                freq=interval.replace("m", "min"), tz="UTC")
            return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                                 "close": 1.5, "volume": 10.0},
                                index=idx).rename_axis("open_time")
        df5, df15 = D.load_history("BTCUSDT", cache_dir=self.tmp,
                                   fetcher=fake_fetch, days=1)
        self.assertEqual(df5.attrs["symbol"], "BTCUSDT")
        self.assertGreater(len(df5), 0)
        self.assertGreater(len(df15), 0)


if __name__ == "__main__":
    unittest.main()
