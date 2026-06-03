import builtins
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.leverage import __main__ as M
from src.leverage.__main__ import build_signal_ticket, gate_progress
from src.leverage.signals import Params


def _frames():
    n = 400
    start = pd.Timestamp("2026-05-01 13:30", tz="UTC")
    idx = pd.date_range(start, periods=n, freq="5min")
    base = 60000 + np.arange(n) * 2.0
    close = base.copy()
    close[-1] = base[:-1].max() + 500
    df5 = pd.DataFrame({"open": close, "high": close + 20, "low": close - 20,
                        "close": close,
                        "volume": np.r_[np.full(n - 1, 100.0), 1000.0]},
                       index=idx).rename_axis("open_time")
    df5.attrs["symbol"] = "BTCUSDT"
    idx15 = pd.date_range(start, periods=n // 3, freq="15min")
    c15 = 60000 + np.arange(len(idx15)) * 6.0
    df15 = pd.DataFrame({"open": c15, "high": c15 + 30, "low": c15 - 30,
                         "close": c15, "volume": 300.0},
                        index=idx15).rename_axis("open_time")
    df15.attrs["symbol"] = "BTCUSDT"
    return df5, df15


class TestCLI(unittest.TestCase):
    def test_build_signal_ticket_returns_string_or_none(self):
        df5, df15 = _frames()
        out = build_signal_ticket(df5, df15, Params(), equity=1500)
        # either a ticket string or a clear 'no setup' message
        self.assertIsInstance(out, str)
        self.assertTrue(("LONG" in out) or ("no" in out.lower()))

    def test_gate_progress_reports_counts(self):
        prog = gate_progress(closed_trades=[{"pnl_pct": 0.01}] * 20)
        self.assertIn("20", prog)
        self.assertIn("100", prog)


class TestMenuHelpers(unittest.TestCase):
    def test_verdict_line_generic_when_no_cache(self):
        with tempfile.TemporaryDirectory() as d:
            old = M._WF_CACHE
            M._WF_CACHE = os.path.join(d, "missing.json")
            try:
                line = M._backtest_verdict_line()
            finally:
                M._WF_CACHE = old
        # never invents numbers it does not have
        self.assertNotIn("/", line.split("validate")[0] if "validate" in line else line)
        self.assertIn("BACKTEST", line)

    def test_verdict_line_reflects_recorded_run(self):
        with tempfile.TemporaryDirectory() as d:
            old = M._WF_CACHE
            M._WF_CACHE = os.path.join(d, "wf.json")
            try:
                M._record_walk_forward("BTC", pos=0, total=6)
                line = M._backtest_verdict_line()
            finally:
                M._WF_CACHE = old
        self.assertIn("0/6", line)
        self.assertIn("NO EDGE", line)

    def test_verdict_line_edge_when_all_positive(self):
        with tempfile.TemporaryDirectory() as d:
            old = M._WF_CACHE
            M._WF_CACHE = os.path.join(d, "wf.json")
            try:
                M._record_walk_forward("BTC", pos=6, total=6)
                line = M._backtest_verdict_line()
            finally:
                M._WF_CACHE = old
        self.assertIn("6/6", line)
        self.assertNotIn("NO EDGE", line)


class TestMenuRouting(unittest.TestCase):
    """menu() must route each choice to the right handler and exit on Q —
    no network, handlers stubbed."""

    def _run_menu_with(self, choices):
        calls = []
        seq = iter(choices)
        stubs = {
            "_menu_signal": lambda: calls.append("signal"),
            "_menu_paper_log": lambda: calls.append("paper"),
            "_menu_portfolio": lambda: calls.append("portfolio"),
            "_menu_backtest": lambda: calls.append("backtest"),
            "_menu_optimize": lambda: calls.append("optimize"),
            "_menu_status": lambda: calls.append("status"),
            "_print_menu_header": lambda: None,
        }
        originals = {k: getattr(M, k) for k in stubs}
        orig_input = builtins.input
        builtins.input = lambda *_a, **_k: next(seq)
        for k, v in stubs.items():
            setattr(M, k, v)
        try:
            M.menu()
        finally:
            builtins.input = orig_input
            for k, v in originals.items():
                setattr(M, k, v)
        return calls

    def test_each_choice_routes(self):
        calls = self._run_menu_with(["1", "2", "3", "4", "5", "6", "Q"])
        self.assertEqual(calls, ["signal", "paper", "portfolio", "backtest",
                                 "optimize", "status"])

    def test_blank_and_q_exit(self):
        self.assertEqual(self._run_menu_with(["Q"]), [])
        self.assertEqual(self._run_menu_with([""]), [])

    def test_unknown_choice_does_not_crash(self):
        calls = self._run_menu_with(["9", "Q"])
        self.assertEqual(calls, [])


class TestMenuResilience(unittest.TestCase):
    """A failing or empty data source must NEVER crash the menu — every action
    returns control to the loop so navigation stays trustworthy."""

    def _drive(self, load_fn, inputs, fund_fn=None):
        import src.leverage.data as D
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        orig_load, orig_fund, orig_input = D.load_history, D.fetch_funding, builtins.input
        seq = iter(inputs)

        def fake_input(*_a, **_k):
            try:
                return next(seq)
            except StopIteration:
                raise EOFError
        D.load_history = load_fn
        D.fetch_funding = fund_fn or (lambda *_a, **_k: df.copy())
        builtins.input = fake_input
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                M.menu()  # must return, not raise
            finally:
                os.chdir(cwd)
                D.load_history, D.fetch_funding = orig_load, orig_fund
                builtins.input = orig_input

    def _raises(self, *_a, **_k):
        import requests
        raise requests.HTTPError("451 Unavailable For Legal Reasons")

    def _empty(self, *_a, **_k):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"]
                          ).rename_axis("open_time")
        df.attrs["symbol"] = "BTCUSDT"
        return df, df.copy()

    def test_signal_survives_network_failure(self):
        self._drive(self._raises, ["1", "BTC", "1500", "Q"])

    def test_signal_all_survives_network_failure(self):
        self._drive(self._raises, ["1", "ALL", "", "Q"])

    def test_backtest_survives_network_failure(self):
        self._drive(self._raises, ["4", "BTC", "Q"], fund_fn=self._raises)

    def test_backtest_survives_empty_data(self):
        self._drive(self._empty, ["4", "BTC", "Q"])

    def test_signal_survives_eof_then_network_failure(self):
        # choice 1, then stdin dies mid-flow -> defaults -> network failure
        self._drive(self._raises, ["1"])


class TestPaperLog(unittest.TestCase):
    """The PAPER LOG handler must persist a position only on a confirmed,
    actionable, safe setup — and refuse otherwise — without touching network.
    generate_signals is stubbed so the test exercises the handler's glue
    (size -> safety -> confirm -> ledger), not market-data geometry."""

    def _safe_long(self, ts):
        from src.leverage.signals import Signal
        # entry 60k, -0.5% stop -> ~4x at 2% risk (inside the 3-6x band), safe
        return Signal("BTCUSDT", "long", ts, entry=60000.0, atr=300.0,
                      stop=59700.0, target=61320.0, trail_trigger=60900.0,
                      session="us-open", confidence=0.6)

    def _drive(self, inputs, signal_factory):
        import src.leverage.data as D
        from src.leverage.paper import PaperLedger
        from src.leverage.reversion import ReversionParams
        df5, df15 = _frames()
        sig = signal_factory(df5.index[-1]) if signal_factory else None
        orig_load, orig_strat = D.load_history, M._strategy
        orig_input = builtins.input
        seq = iter(inputs)
        D.load_history = lambda *_a, **_k: (df5, df15)
        # _menu_paper_log resolves its generator via _strategy -> stub it
        M._strategy = lambda _name: (
            (lambda *_a, **_k: ([sig] if sig else [])), ReversionParams(), "reversion")
        builtins.input = lambda *_a, **_k: next(seq)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                M._menu_paper_log()
                return PaperLedger().open_positions()
            finally:
                os.chdir(cwd)
                D.load_history, M._strategy = orig_load, orig_strat
                builtins.input = orig_input

    def test_logs_on_confirmed_safe_setup(self):
        logged = self._drive(["BTC", "", "y"], self._safe_long)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["side"], "long")
        self.assertEqual(logged[0]["status"], "open")

    def test_declining_confirmation_logs_nothing(self):
        self.assertEqual(self._drive(["BTC", "", "n"], self._safe_long), [])

    def test_no_setup_logs_nothing(self):
        self.assertEqual(self._drive(["BTC", ""], None), [])


if __name__ == "__main__":
    unittest.main()
