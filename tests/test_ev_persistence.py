"""Schema 21: the EV levels a verdict was taken on survive into the ledger.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_ev_persistence -v

Before this, `ev_score` was stored and the four numbers `decide_verdict`
actually reads were discarded. That is why "did a STRONG contract beat a THIN
one" had no answer across 851 closed trades. These tests exist so the columns
cannot quietly stop being written.

Never names the real ledger: PaperManager migrates on init.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.paper_manager import PaperManager, _SCHEMA_VERSION


class SchemaTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = str(Path(self.dir.name) / "ledger.db")
        self.pm = PaperManager(db_path=self.db)

    def tearDown(self):
        self.dir.cleanup()

    def _cols(self):
        with sqlite3.connect(self.db) as c:
            return {r[1] for r in c.execute("PRAGMA table_info(trades)")}

    def test_the_four_ev_columns_exist(self):
        self.assertLessEqual(
            {"entry_ev_net", "entry_ev_gross", "entry_ev_cost", "entry_ev_noise"},
            self._cols())

    def test_the_schema_version_advanced(self):
        self.assertGreaterEqual(_SCHEMA_VERSION, 21)

    def test_ev_score_is_still_there(self):
        """A rank and a level answer different questions; 21 adds, never replaces."""
        self.assertIn("ev_score", self._cols())


class WritePathTest(unittest.TestCase):
    """The columns are useless unless something fills them."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = str(Path(self.dir.name) / "ledger.db")
        self.pm = PaperManager(db_path=self.db)

    def tearDown(self):
        self.dir.cleanup()

    def _trade(self, **over):
        t = {"date": "2026-08-10", "ticker": "SPY", "expiration": "2026-09-18",
             "strike": 500.0, "type": "call", "entry_price": 5.0,
             "quality_score": 0.5, "strategy_name": "Long Call",
             "ev_per_contract": 88.0, "ev_gross_per_contract": 120.0,
             "ev_cost_per_contract": 32.0, "ev_noise": 25.0}
        t.update(over)
        return t

    def _read(self):
        with sqlite3.connect(self.db) as c:
            return c.execute(
                "select entry_ev_net, entry_ev_gross, entry_ev_cost, "
                "entry_ev_noise from trades").fetchall()

    def test_the_ev_levels_round_trip_into_the_ledger(self):
        self.pm.log_trade(self._trade())
        self.assertEqual(self._read(), [(88.0, 120.0, 32.0, 25.0)])

    def test_a_trade_without_ev_stores_null_not_zero(self):
        """NULL means 'not recorded'. Zero would read as a zero-edge trade and
        silently corrupt any future analysis of the grade."""
        self.pm.log_trade(self._trade(ev_per_contract=None, ev_gross_per_contract=None,
                                      ev_cost_per_contract=None, ev_noise=None))
        self.assertEqual(self._read(), [(None, None, None, None)])

    def test_a_nan_ev_stores_null(self):
        self.pm.log_trade(self._trade(ev_per_contract=float("nan")))
        self.assertIsNone(self._read()[0][0])

    def test_sigma_is_reconstructable_after_the_fact(self):
        """The whole point: net_ev / noise can be recovered from the row."""
        self.pm.log_trade(self._trade())
        net, _, _, noise = self._read()[0]
        self.assertAlmostEqual(net / noise, 3.52, places=2)


class EveryLogPathCarriesTheEvLevelsTest(unittest.TestCase):
    """Every dict that reaches `log_trade` must carry the four EV levels.

    Written after the columns shipped storing NULL. Three sites build a trade
    dict in `options_screener`; two build from a variable named `row` and one
    from `top_pick_row`. The original patch was written against the text
    `row.get("ev_score")`, so it silently skipped the third — which is the one
    the auto-logger actually used. Every trade logged on 2026-08-10 stored NULL
    in all four columns.

    Parsed with `ast` and keyed on the ASSIGNED NAME rather than on the source
    text, so a fourth site built from a differently-named variable still fails.
    """

    EV_KEYS = {"ev_per_contract", "ev_gross_per_contract",
               "ev_cost_per_contract", "ev_noise"}

    def _trade_dicts(self):
        """(lineno, {keys}) for every `trade_dict = {...}` literal."""
        import ast
        import pathlib
        tree = ast.parse(pathlib.Path("src/options_screener.py").read_text())
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "trade_dict" not in names:
                continue
            keys = {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            out.append((node.lineno, keys))
        return out

    def test_at_least_the_known_sites_are_found(self):
        """Guards the guard: a parser that finds nothing would pass vacuously."""
        self.assertGreaterEqual(len(self._trade_dicts()), 2)

    def test_every_trade_dict_carries_all_four_ev_levels(self):
        for lineno, keys in self._trade_dicts():
            missing = self.EV_KEYS - keys
            self.assertEqual(
                missing, set(),
                f"trade_dict at options_screener.py:{lineno} is missing "
                f"{sorted(missing)} — trades logged from it will store NULL")

    def test_every_trade_dict_also_carries_the_rank_beside_the_levels(self):
        """`ev_score` is not redundant with them; a rank and a level differ."""
        for lineno, keys in self._trade_dicts():
            self.assertIn("ev_score", keys, f"line {lineno}")


class ScanRowCarriesNoiseTest(unittest.TestCase):
    """`ev_noise` has to be on the row before the logger can persist it."""

    def test_the_screener_attaches_ev_noise_to_the_frame(self):
        import pathlib
        src = pathlib.Path("src/options_screener.py").read_text()
        self.assertIn('df["ev_noise"]', src)

    def test_the_spread_path_carries_the_ev_levels_through(self):
        from src.spread_scoring import _SHORT_LEG_SCORE_COLS
        self.assertLessEqual(
            {"ev_per_contract", "ev_gross_per_contract",
             "ev_cost_per_contract", "ev_noise"},
            set(_SHORT_LEG_SCORE_COLS))


if __name__ == "__main__":
    unittest.main()
