"""Event persistence and forward marks. Always against a temp database."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import store
from src.catalyst.models import CatalystEvent, Trial


def a_trial(date="2026-10-31", **kw):
    base = dict(nct_id="NCT06510816", sponsor_name="Annexon, Inc.",
                brief_title="Vonaprument in Dry AMD", phase="PHASE3",
                event_date=date, date_precision="day", date_type="ESTIMATED",
                status="ACTIVE_NOT_RECRUITING", enrollment=400,
                allocation="RANDOMIZED", masking="QUADRUPLE",
                primary_outcome="Change in GA lesion area",
                conditions=("Geographic Atrophy",))
    base.update(kw)
    return Trial(**base)


def an_event(date="2026-10-31"):
    return CatalystEvent(trial=a_trial(date), ticker="ANNX", mcap=976_332_558.0)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = store.connect(os.path.join(self._dir.name, "catalysts.db"))
        self.addCleanup(self.conn.close)


class TestUpsert(StoreCase):
    def test_inserts_with_first_seen(self):
        store.upsert_event(self.conn, an_event(), "2026-08-25")
        row = self.conn.execute(
            "SELECT ticker, event_date, first_seen, last_seen "
            "FROM catalyst_events").fetchone()
        self.assertEqual(row, ("ANNX", "2026-10-31", "2026-08-25", "2026-08-25"))

    def test_reseeing_preserves_first_seen_and_moves_last_seen(self):
        store.upsert_event(self.conn, an_event(), "2026-08-25")
        store.upsert_event(self.conn, an_event(), "2026-09-01")
        row = self.conn.execute(
            "SELECT first_seen, last_seen FROM catalyst_events").fetchone()
        self.assertEqual(row, ("2026-08-25", "2026-09-01"))

    def test_one_row_per_event_id(self):
        store.upsert_event(self.conn, an_event(), "2026-08-25")
        store.upsert_event(self.conn, an_event(), "2026-09-01")
        n = self.conn.execute("SELECT COUNT(*) FROM catalyst_events").fetchone()[0]
        self.assertEqual(n, 1)

    def test_a_moved_date_updates_the_event_row(self):
        store.upsert_event(self.conn, an_event("2026-10-31"), "2026-08-25")
        store.upsert_event(self.conn, an_event("2027-02-28"), "2026-09-01")
        date = self.conn.execute(
            "SELECT event_date FROM catalyst_events").fetchone()[0]
        self.assertEqual(date, "2027-02-28")

    def test_absent_mcap_is_null_not_zero(self):
        ev = CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=None)
        store.upsert_event(self.conn, ev, "2026-08-25")
        self.assertIsNone(self.conn.execute(
            "SELECT mcap_at_seen FROM catalyst_events").fetchone()[0])


class TestMarks(StoreCase):
    def test_each_observation_is_its_own_row(self):
        store.upsert_event(self.conn, an_event(), "2026-08-25")
        store.add_mark(self.conn, "NCT06510816:PRIMARY_COMPLETION",
                       "2026-09-01", "2026-10-31", "RECRUITING", 5.15)
        store.add_mark(self.conn, "NCT06510816:PRIMARY_COMPLETION",
                       "2026-10-01", "2027-02-28", "RECRUITING", 6.20)
        n = self.conn.execute("SELECT COUNT(*) FROM catalyst_marks").fetchone()[0]
        self.assertEqual(n, 2)

    def test_slippage_is_computed_from_marks_not_overwritten(self):
        store.upsert_event(self.conn, an_event("2026-10-31"), "2026-08-25")
        store.add_mark(self.conn, "NCT06510816:PRIMARY_COMPLETION",
                       "2026-09-01", "2026-10-31", "RECRUITING", 5.15)
        store.add_mark(self.conn, "NCT06510816:PRIMARY_COMPLETION",
                       "2026-10-01", "2027-02-28", "RECRUITING", 6.20)
        self.assertEqual(store.slippage(self.conn,
                                        "NCT06510816:PRIMARY_COMPLETION"), 120)

    def test_slippage_is_none_with_a_single_mark(self):
        store.upsert_event(self.conn, an_event(), "2026-08-25")
        store.add_mark(self.conn, "NCT06510816:PRIMARY_COMPLETION",
                       "2026-09-01", "2026-10-31", "RECRUITING", 5.15)
        self.assertIsNone(store.slippage(self.conn,
                                         "NCT06510816:PRIMARY_COMPLETION"))

    def test_month_precision_dates_do_not_crash_slippage(self):
        ev = CatalystEvent(trial=a_trial("2027-03", date_precision="month"),
                           ticker="ANNX", mcap=1.0)
        store.upsert_event(self.conn, ev, "2026-08-25")
        store.add_mark(self.conn, ev.event_id, "2026-09-01", "2027-03", "R", 1.0)
        store.add_mark(self.conn, ev.event_id, "2026-10-01", "2027-06", "R", 1.0)
        self.assertEqual(store.slippage(self.conn, ev.event_id), 92)


class TestOutstanding(StoreCase):
    def test_returns_events_whose_date_has_passed(self):
        store.upsert_event(self.conn, an_event("2026-10-31"), "2026-08-25")
        store.upsert_event(self.conn, CatalystEvent(
            trial=a_trial("2027-06-30", nct_id="NCT99999999"),
            ticker="SRPT", mcap=2e9), "2026-08-25")
        rows = store.outstanding(self.conn, "2026-12-01")
        self.assertEqual([r[0] for r in rows],
                         ["NCT06510816:PRIMARY_COMPLETION"])

    def test_empty_when_nothing_has_elapsed(self):
        store.upsert_event(self.conn, an_event("2027-10-31"), "2026-08-25")
        self.assertEqual(store.outstanding(self.conn, "2026-12-01"), [])


if __name__ == "__main__":
    unittest.main()
