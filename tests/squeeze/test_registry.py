"""Point-in-time cohort registry: frozen at formation, coverage recorded."""
import os
import tempfile
import unittest

from src.squeeze.sleeve import registry


def _m(symbol, arm="treated", decile=10):
    return {"symbol": symbol, "arm": arm, "si_decile": decile,
            "rv": 0.9, "log_mcap": 20.0, "log_price": 3.0}


class RegistryTest(unittest.TestCase):
    def setUp(self):
        fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        fd.close()
        self.db = fd.name
        registry.ensure_db(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_a_cycle_round_trips(self):
        n = registry.open_cycle("2026-08-14", [_m("NBIS"), _m("SMCI")],
                                [_m("KO", "control", 2)], db_path=self.db)
        self.assertEqual(n, 3)
        members = registry.cycle_members("2026-08-14", db_path=self.db)
        self.assertEqual(len(members), 3)
        self.assertEqual({m["arm"] for m in members}, {"treated", "control"})

    def test_reopening_a_cycle_does_not_rewrite_it(self):
        registry.open_cycle("2026-08-14", [_m("NBIS")], [_m("KO", "control", 2)],
                            db_path=self.db)
        again = registry.open_cycle("2026-08-14", [_m("TOTALLY_DIFFERENT")], [],
                                    db_path=self.db)
        self.assertEqual(again, 0)
        symbols = [m["symbol"] for m in registry.cycle_members("2026-08-14",
                                                              db_path=self.db)]
        self.assertIn("NBIS", symbols)
        self.assertNotIn("TOTALLY_DIFFERENT", symbols)

    def test_cycles_come_back_in_order(self):
        for date in ("2026-09-15", "2026-08-14", "2026-08-29"):
            registry.open_cycle(date, [_m("X")], [_m("Y", "control", 2)],
                                db_path=self.db)
        self.assertEqual(registry.cycles(db_path=self.db),
                         ["2026-08-14", "2026-08-29", "2026-09-15"])

    def test_coverage_defaults_to_missing_and_can_be_marked(self):
        registry.open_cycle("2026-08-14", [_m("NBIS")], [_m("KO", "control", 2)],
                            db_path=self.db)
        self.assertFalse(registry.coverage(db_path=self.db)["2026-08-14"])
        registry.mark_coverage("2026-08-14", True, db_path=self.db)
        self.assertTrue(registry.coverage(db_path=self.db)["2026-08-14"])

    def test_an_unknown_cycle_has_no_members(self):
        self.assertEqual(registry.cycle_members("1999-01-01", db_path=self.db), [])
